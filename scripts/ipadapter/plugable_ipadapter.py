import torch

from .ipadapter_model import ImageEmbed, IPAdapterModel


def get_block(model, flag):
    return {
        "input": model.input_blocks,
        "middle": [model.middle_block],
        "output": model.output_blocks,
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

    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
    )
    out = out.transpose(1, 2).reshape(batch_size, -1, h * head_dim)

    del k, v

    for f in self.ipadapter_hacks:
        out = out + f(self, x, q)

    del q, x

    return self.to_out(out)


all_hacks = {}
current_model = None


def hack_blk(block, function, type):
    if not hasattr(block, "ipadapter_hacks"):
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
    def __init__(self, ipadapter: IPAdapterModel):
        super().__init__()
        self.ipadapter = ipadapter
        self.disable_memory_management = True
        self.dtype = None
        self.weight = 1.0
        self.cache = None
        self.p_start = 0.0
        self.p_end = 1.0

    def reset(self):
        self.cache = {}

    @torch.no_grad()
    def hook(
        self, model, preprocessor_outputs, weight, start, end, dtype=torch.float32
    ):
        global current_model
        current_model = model

        self.p_start = start
        self.p_end = end

        self.cache = {}

        self.weight = weight
        device = torch.device("cpu")
        self.dtype = dtype

        self.ipadapter.to(device, dtype=self.dtype)
        if getattr(preprocessor_outputs, "bypass_average", False):
            self.image_emb = preprocessor_outputs
        else:
            if isinstance(preprocessor_outputs, (list, tuple)):
                preprocessor_outputs = preprocessor_outputs
            else:
                preprocessor_outputs = [preprocessor_outputs]
            self.image_emb = ImageEmbed.average_of(
                *[self.ipadapter.get_image_emb(o) for o in preprocessor_outputs]
            )
        # From https://github.com/laksjdjf/IPAdapter-ComfyUI
        if not self.ipadapter.is_sdxl:
            number = 0  # index of to_kvs
            for id in [
                1,
                2,
                4,
                5,
                7,
                8,
            ]:  # id of input_blocks that have cross attention
                set_model_attn2_replace(model, self.patch_forward(number), "input", id)
                number += 1
            for id in [
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
            ]:  # id of output_blocks that have cross attention
                set_model_attn2_replace(model, self.patch_forward(number), "output", id)
                number += 1
            set_model_attn2_replace(model, self.patch_forward(number), "middle", 0)
        else:
            number = 0
            for id in [4, 5, 7, 8]:  # id of input_blocks that have cross attention
                block_indices = (
                    range(2) if id in [4, 5] else range(10)
                )  # transformer_depth
                for index in block_indices:
                    set_model_patch_replace(
                        model, self.patch_forward(number), "input", id, index
                    )
                    number += 1
            for id in range(6):  # id of output_blocks that have cross attention
                block_indices = (
                    range(2) if id in [3, 4, 5] else range(10)
                )  # transformer_depth
                for index in block_indices:
                    set_model_patch_replace(
                        model, self.patch_forward(number), "output", id, index
                    )
                    number += 1
            for index in range(10):
                set_model_patch_replace(
                    model, self.patch_forward(number), "middle", 0, index
                )
                number += 1

        return

    def call_ip(self, key: str, feat, device):
        if key in self.cache:
            return self.cache[key]
        else:
            ip = self.ipadapter.ip_layers.to_kvs[key](feat).to(device)
            self.cache[key] = ip
            return ip

    @torch.no_grad()
    def patch_forward(self, number: int):
        @torch.no_grad()
        def forward(attn_blk, x, q):
            batch_size, sequence_length, inner_dim = x.shape
            h = attn_blk.heads
            head_dim = inner_dim // h

            current_sampling_percent = getattr(
                current_model, "current_sampling_percent", 0.5
            )
            if (
                current_sampling_percent < self.p_start
                or current_sampling_percent > self.p_end
            ):
                return 0

            k_key = f"{number * 2 + 1}_to_k_ip"
            v_key = f"{number * 2 + 1}_to_v_ip"
            cond_uncond_image_emb = self.image_emb.eval(current_model.cond_mark)
            ip_k = self.call_ip(k_key, cond_uncond_image_emb, device=q.device)
            ip_v = self.call_ip(v_key, cond_uncond_image_emb, device=q.device)

            ip_k, ip_v = map(
                lambda t: t.view(batch_size, -1, h, head_dim).transpose(1, 2),
                (ip_k, ip_v),
            )
            assert ip_k.dtype == ip_v.dtype

            # On MacOS, q can be float16 instead of float32.
            # https://github.com/Mikubill/sd-webui-controlnet/issues/2208
            if q.dtype != ip_k.dtype:
                ip_k = ip_k.to(dtype=q.dtype)
                ip_v = ip_v.to(dtype=q.dtype)

            ip_out = torch.nn.functional.scaled_dot_product_attention(
                q, ip_k, ip_v, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            ip_out = ip_out.transpose(1, 2).reshape(batch_size, -1, h * head_dim)

            return ip_out * self.weight

        return forward
