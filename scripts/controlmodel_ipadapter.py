import math
import torch
import torch.nn as nn


# attention_channels of input, output, middle
SD_V12_CHANNELS = [320] * 4 + [640] * 4 + [1280] * 4 + [1280] * 6 + [640] * 6 + [320] * 6 + [1280] * 2
SD_XL_CHANNELS = [640] * 8 + [1280] * 40 + [1280] * 60 + [640] * 12 + [1280] * 20


class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(-1, self.clip_extra_context_tokens,
                                                              self.cross_attention_dim)
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


# Cross Attention to_k, to_v for IPAdapter
class To_KV(torch.nn.Module):
    def __init__(self, cross_attention_dim):
        super().__init__()

        channels = SD_XL_CHANNELS if cross_attention_dim == 2048 else SD_V12_CHANNELS
        self.to_kvs = torch.nn.ModuleList(
            [torch.nn.Linear(cross_attention_dim, channel, bias=False) for channel in channels])

    def load_state_dict(self, state_dict):
        # input -> output -> middle
        for i, key in enumerate(state_dict.keys()):
            self.to_kvs[i].weight.data = state_dict[key]


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def reshape_tensor(x, heads):
    bs, length, width = x.shape
    #(bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)


    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1) # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)


class Resampler(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
    ):
        super().__init__()

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)

        self.proj_in = nn.Linear(embedding_dim, dim)

        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x):

        latents = self.latents.repeat(x.size(0), 1, 1)

        x = self.proj_in(x)

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        latents = self.proj_out(latents)
        return self.norm_out(latents)


class IPAdapterModel(torch.nn.Module):
    def __init__(self, state_dict, clip_embeddings_dim, is_plus):
        super().__init__()
        self.device = "cpu"

        # cross_attention_dim is equal to text_encoder output
        self.cross_attention_dim = state_dict["ip_adapter"]["1.to_k_ip.weight"].shape[1]
        self.is_plus = is_plus

        if self.is_plus:
            self.clip_extra_context_tokens = 16

            self.image_proj_model = Resampler(
                dim=self.cross_attention_dim,
                depth=4,
                dim_head=64,
                heads=12,
                num_queries=self.clip_extra_context_tokens,
                embedding_dim=clip_embeddings_dim,
                output_dim=self.cross_attention_dim,
                ff_mult=4
            )
        else:
            self.clip_extra_context_tokens = state_dict["image_proj"]["proj.weight"].shape[0] // self.cross_attention_dim

            self.image_proj_model = ImageProjModel(
                cross_attention_dim=self.cross_attention_dim,
                clip_embeddings_dim=clip_embeddings_dim,
                clip_extra_context_tokens=self.clip_extra_context_tokens
            )

        self.load_ip_adapter(state_dict)

    def load_ip_adapter(self, state_dict):
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        self.ip_layers = To_KV(self.cross_attention_dim)
        self.ip_layers.load_state_dict(state_dict["ip_adapter"])

    @torch.inference_mode()
    def get_image_embeds(self, clip_vision_output):
        self.image_proj_model.cpu()

        if self.is_plus:
            from annotator.clipvision import clip_vision_h_uc
            cond = self.image_proj_model(clip_vision_output['hidden_states'][-2].to(device='cpu', dtype=torch.float32))
            uncond = self.image_proj_model(clip_vision_h_uc.to(cond))
            return cond, uncond

        clip_image_embeds = clip_vision_output['image_embeds'].to(device='cpu', dtype=torch.float32)
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        # input zero vector for unconditional.
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))
        return image_prompt_embeds, uncond_image_prompt_embeds


def get_block(model, flag):
    return {
        'input': model.input_blocks, 'middle': [model.middle_block], 'output': model.output_blocks
    }[flag]


def attn_forward_hacked(self, x, context=None, **kwargs):
    batch_size, sequence_length, inner_dim = x.shape
    h = self.heads
    head_dim = inner_dim // h

    if context is None:
        context = x

    q = self.to_q(x)
    k = self.to_k(context)
    v = self.to_v(context)

    del context

    q, k, v = map(
        lambda t: t.view(batch_size, -1, h, head_dim).transpose(1, 2),
        (q, k, v),
    )

    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
    out = out.transpose(1, 2).reshape(batch_size, -1, h * head_dim)

    del k, v

    for f in self.ipadapter_hacks:
        out = out + f(self, x, q)

    del q, x

    return self.to_out(out)


all_hacks = {}
current_model = None


def hack_blk(block, function, type):
    if not hasattr(block, 'ipadapter_hacks'):
        block.ipadapter_hacks = []

    if len(block.ipadapter_hacks) == 0:
        all_hacks[block] = block.forward
        block.forward = attn_forward_hacked.__get__(block, type)

    block.ipadapter_hacks.append(function)
    return


def set_model_attn2_replace(model, function, flag, id):
    from ldm.modules.attention import CrossAttention
    block = get_block(model, flag)[id][1].transformer_blocks[0].attn2
    hack_blk(block, function, CrossAttention)
    return


def set_model_patch_replace(model, function, flag, id, trans_id):
    from sgm.modules.attention import CrossAttention
    blk = get_block(model, flag)
    block = blk[id][1].transformer_blocks[trans_id].attn2
    hack_blk(block, function, CrossAttention)
    return


def clear_all_ip_adapter():
    global all_hacks, current_model
    for k, v in all_hacks.items():
        k.forward = v
        k.ipadapter_hacks = []
    all_hacks = {}
    current_model = None
    return


class PlugableIPAdapter(torch.nn.Module):
    def __init__(self, state_dict, clip_embeddings_dim, is_plus):
        super().__init__()
        self.sdxl = clip_embeddings_dim == 1280 and not is_plus
        self.is_plus = is_plus
        self.ipadapter = IPAdapterModel(state_dict, clip_embeddings_dim=clip_embeddings_dim, is_plus=is_plus)
        self.disable_memory_management = True
        self.dtype = None
        self.weight = 1.0
        self.cache = {}
        self.p_start = 0.0
        self.p_end = 1.0
        return

    def reset(self):
        self.cache = {}
        return

    @torch.no_grad()
    def hook(self, model, clip_vision_output, weight, start, end, dtype=torch.float32):
        global current_model
        current_model = model

        self.p_start = start
        self.p_end = end

        self.cache = {}

        self.weight = weight
        device = torch.device('cpu')
        self.dtype = dtype

        self.ipadapter.to(device, dtype=self.dtype)
        self.image_emb, self.uncond_image_emb = self.ipadapter.get_image_embeds(clip_vision_output)

        self.image_emb = self.image_emb.to(device, dtype=self.dtype)
        self.uncond_image_emb = self.uncond_image_emb.to(device, dtype=self.dtype)

        # From https://github.com/laksjdjf/IPAdapter-ComfyUI
        if not self.sdxl:
            number = 0  # index of to_kvs
            for id in [1, 2, 4, 5, 7, 8]:  # id of input_blocks that have cross attention
                set_model_attn2_replace(model, self.patch_forward(number), "input", id)
                number += 1
            for id in [3, 4, 5, 6, 7, 8, 9, 10, 11]:  # id of output_blocks that have cross attention
                set_model_attn2_replace(model, self.patch_forward(number), "output", id)
                number += 1
            set_model_attn2_replace(model, self.patch_forward(number), "middle", 0)
        else:
            number = 0
            for id in [4, 5, 7, 8]:  # id of input_blocks that have cross attention
                block_indices = range(2) if id in [4, 5] else range(10)  # transformer_depth
                for index in block_indices:
                    set_model_patch_replace(model, self.patch_forward(number), "input", id, index)
                    number += 1
            for id in range(6):  # id of output_blocks that have cross attention
                block_indices = range(2) if id in [3, 4, 5] else range(10)  # transformer_depth
                for index in block_indices:
                    set_model_patch_replace(model, self.patch_forward(number), "output", id, index)
                    number += 1
            for index in range(10):
                set_model_patch_replace(model, self.patch_forward(number), "middle", 0, index)
                number += 1

        return

    def call_ip(self, number, feat, device):
        if number in self.cache:
            return self.cache[number]
        else:
            ip = self.ipadapter.ip_layers.to_kvs[number](feat).to(device)
            self.cache[number] = ip
            return ip

    @torch.no_grad()
    def patch_forward(self, number):
        @torch.no_grad()
        def forward(attn_blk, x, q):
            batch_size, sequence_length, inner_dim = x.shape
            h = attn_blk.heads
            head_dim = inner_dim // h

            current_sampling_percent = getattr(current_model, 'current_sampling_percent', 0.5)
            if current_sampling_percent < self.p_start or current_sampling_percent > self.p_end:
                return 0

            cond_mark = current_model.cond_mark[:, :, :, 0].to(self.image_emb)
            cond_uncond_image_emb = self.image_emb * cond_mark + self.uncond_image_emb * (1 - cond_mark)
            ip_k = self.call_ip(number * 2, cond_uncond_image_emb, device=q.device)
            ip_v = self.call_ip(number * 2 + 1, cond_uncond_image_emb, device=q.device)

            ip_k, ip_v = map(
                lambda t: t.view(batch_size, -1, h, head_dim).transpose(1, 2),
                (ip_k, ip_v),
            )

            ip_out = torch.nn.functional.scaled_dot_product_attention(q, ip_k, ip_v, attn_mask=None, dropout_p=0.0, is_causal=False)
            ip_out = ip_out.transpose(1, 2).reshape(batch_size, -1, h * head_dim)

            return ip_out * self.weight
        return forward
