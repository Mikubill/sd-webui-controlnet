import torch
import einops

from modules import devices


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


class IPAdapterModel(torch.nn.Module):
    def __init__(self, state_dict, clip_embeddings_dim):
        super().__init__()
        self.device = "cpu"

        # cross_attention_dim is equal to text_encoder output
        self.cross_attention_dim = state_dict["ip_adapter"]["1.to_k_ip.weight"].shape[1]

        # number of tokens of ip_adapter embedding
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
    def get_image_embeds(self, clip_image_embeds):
        '''
        clip_image_embeds: clip_vision_output, size: batch_size, 1024(sdv1-2) or 1280(sdxl)
        return: cond, uncond for CFG, size: batch_size, num_tokens, 768(sdv1) or 1024(sdv2) or 2048(sdxl)
        '''

        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        # input zero vector for unconditional.
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))
        return image_prompt_embeds, uncond_image_prompt_embeds


def get_block(model, flag):
    return {
        'input': model.input_blocks, 'middle': [model.middle_block], 'output': model.output_blocks
    }[flag]


all_hacks = {}
current_model = None


def set_model_attn2_replace(model, function, flag, id):
    from ldm.modules.attention import CrossAttention
    block = get_block(model, flag)[id][1].transformer_blocks[0].attn2
    all_hacks[block] = block.forward
    block.forward = function.__get__(block, CrossAttention)
    return


def set_model_patch_replace(model, function, flag, id, trans_id):
    from sgm.modules.attention import CrossAttention
    blk = get_block(model, flag)
    block = blk[id][1].transformer_blocks[trans_id].attn2
    all_hacks[block] = block.forward
    block.forward = function.__get__(block, CrossAttention)
    return


def clear_all_ip_adapter():
    global all_hacks, current_model
    for k, v in all_hacks.items():
        k.forward = v
    all_hacks = {}
    current_model = None
    return


class PlugableIPAdapter(torch.nn.Module):
    def __init__(self, state_dict, clip_embeddings_dim):
        super().__init__()
        self.sdxl = clip_embeddings_dim == 1280
        self.ipadapter = IPAdapterModel(state_dict, clip_embeddings_dim=clip_embeddings_dim)
        self.control_model = self.ipadapter
        self.dtype = None
        self.weight = 1.0
        return

    def reset(self):
        return

    def hook(self, model, clip_vision_output, weight, dtype=torch.float32, lowvram=False):
        global current_model
        current_model = model

        self.weight = weight
        device = torch.device('cpu') if lowvram else devices.get_device_for("controlnet")
        self.dtype = dtype

        self.ipadapter.to(device, dtype=self.dtype)
        clip_vision_emb = clip_vision_output['image_embeds'].to(device, dtype=self.dtype)
        self.image_emb, self.uncond_image_emb = self.ipadapter.get_image_embeds(clip_vision_emb)

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

    def patch_forward(self, number):
        def forward(self_hacked, x, context=None, **kwargs):
            batch_size, sequence_length, inner_dim = x.shape
            h = self_hacked.heads
            head_dim = inner_dim // h

            if context is None:
                context = x

            q = self_hacked.to_q(x)
            k = self_hacked.to_k(context)
            v = self_hacked.to_v(context)

            cond_mark = current_model.cond_mark[:, :, :, 0]
            cond_uncond_image_emb = self.image_emb * cond_mark + self.uncond_image_emb * (1 - cond_mark)
            ip_k = self.ipadapter.ip_layers.to_kvs[number * 2](cond_uncond_image_emb)
            ip_v = self.ipadapter.ip_layers.to_kvs[number * 2 + 1](cond_uncond_image_emb)

            q, k, v, ip_k, ip_v = map(
                lambda t: t.view(batch_size, -1, h, head_dim).transpose(1, 2),
                (q, k, v, ip_k, ip_v),
            )

            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
            out = out.transpose(1, 2).reshape(batch_size, -1, h * head_dim)

            ip_out = torch.nn.functional.scaled_dot_product_attention(q, ip_k, ip_v, attn_mask=None, dropout_p=0.0, is_causal=False)
            ip_out = ip_out.transpose(1, 2).reshape(batch_size, -1, h * head_dim)

            out = out + ip_out * self.weight
            return self_hacked.to_out(out)
        return forward
