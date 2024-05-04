import itertools
import torch
import math
from typing import Union, Dict, Optional, Callable

from .pulid_attn import PuLIDAttnSetting
from .ipadapter_model import ImageEmbed, IPAdapterModel
from ..enums import StableDiffusionVersion, TransformerID


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


def set_model_attn2_replace(
    model,
    target_cls,
    function,
    transformer_id: TransformerID,
):
    block = get_block(model, transformer_id.block_type.value)
    module = (
        block[transformer_id.block_id][1]
        .transformer_blocks[transformer_id.block_index]
        .attn2
    )
    hack_blk(module, function, target_cls)


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
        self.ipadapter: IPAdapterModel = ipadapter
        self.disable_memory_management = True
        self.dtype = None
        self.weight: Union[float, Dict[int, float]] = 1.0
        self.cache = None
        self.p_start = 0.0
        self.p_end = 1.0
        self.latent_width: int = 0
        self.latent_height: int = 0
        self.effective_region_mask = None
        self.pulid_attn_setting: Optional[PuLIDAttnSetting] = None

    def reset(self):
        self.cache = {}

    @torch.no_grad()
    def hook(
        self,
        model,
        preprocessor_outputs,
        weight,
        start: float,
        end: float,
        latent_width: int,
        latent_height: int,
        effective_region_mask: Optional[torch.Tensor],
        pulid_attn_setting: Optional[PuLIDAttnSetting] = None,
        dtype=torch.float32,
    ):
        global current_model
        current_model = model

        self.p_start = start
        self.p_end = end
        self.latent_width = latent_width
        self.latent_height = latent_height
        self.effective_region_mask = effective_region_mask
        self.pulid_attn_setting = pulid_attn_setting

        self.cache = {}

        self.weight = weight
        device = torch.device("cpu")
        self.dtype = dtype

        self.ipadapter.to(device, dtype=self.dtype)
        if isinstance(preprocessor_outputs, (list, tuple)):
            preprocessor_outputs = preprocessor_outputs
        else:
            preprocessor_outputs = [preprocessor_outputs]
        self.image_emb = ImageEmbed.average_of(
            *[self.ipadapter.get_image_emb(o) for o in preprocessor_outputs]
        )

        if self.ipadapter.is_sdxl:
            sd_version = StableDiffusionVersion.SDXL
            from sgm.modules.attention import CrossAttention
        else:
            sd_version = StableDiffusionVersion.SD1x
            from ldm.modules.attention import CrossAttention

        input_ids, output_ids, middle_ids = sd_version.transformer_ids
        for i, transformer_id in enumerate(
            itertools.chain(input_ids, output_ids, middle_ids)
        ):
            set_model_attn2_replace(
                model,
                CrossAttention,
                self.patch_forward(i, transformer_id.transformer_index),
                transformer_id,
            )

    def weight_on_transformer(self, transformer_index: int) -> float:
        if isinstance(self.weight, dict):
            return self.weight.get(transformer_index, 0.0)
        else:
            assert isinstance(self.weight, (float, int))
            return self.weight

    def call_ip(self, key: str, feat, device):
        if key in self.cache:
            return self.cache[key]
        else:
            ip = self.ipadapter.ip_layers.to_kvs[key](feat).to(device)
            self.cache[key] = ip
            return ip

    def apply_effective_region_mask(self, out: torch.Tensor) -> torch.Tensor:
        if self.effective_region_mask is None:
            return out

        _, sequence_length, _ = out.shape
        # sequence_length = mask_h * mask_w
        # sequence_length = (latent_height * factor) * (latent_height * factor)
        # sequence_length = (latent_height * latent_height) * factor ^ 2
        factor = math.sqrt(sequence_length / (self.latent_width * self.latent_height))
        assert (
            factor > 0
        ), f"{factor}, {sequence_length}, {self.latent_width}, {self.latent_height}"
        mask_h = int(self.latent_height * factor)
        mask_w = int(self.latent_width * factor)

        mask = torch.nn.functional.interpolate(
            self.effective_region_mask.to(out.device),
            size=(mask_h, mask_w),
            mode="bilinear",
        ).squeeze()
        mask = mask.repeat(len(current_model.cond_mark), 1, 1)
        mask = mask.view(mask.shape[0], -1, 1).repeat(1, 1, out.shape[2])
        return out * mask

    def attn_eval(
        self,
        hidden_states: torch.Tensor,
        query: torch.Tensor,
        cond_uncond_image_emb: torch.Tensor,
        attn_heads: int,
        head_dim: int,
        emb_to_k: Callable[[torch.Tensor], torch.Tensor],
        emb_to_v: Callable[[torch.Tensor], torch.Tensor],
    ):
        if self.ipadapter.is_pulid:
            assert self.pulid_attn_setting is not None
            return self.pulid_attn_setting.eval(
                hidden_states,
                query,
                cond_uncond_image_emb,
                attn_heads,
                head_dim,
                emb_to_k,
                emb_to_v,
            )
        else:
            return self._attn_eval_ipadapter(
                hidden_states,
                query,
                cond_uncond_image_emb,
                attn_heads,
                head_dim,
                emb_to_k,
                emb_to_v,
            )

    def _attn_eval_ipadapter(
        self,
        hidden_states: torch.Tensor,
        query: torch.Tensor,
        cond_uncond_image_emb: torch.Tensor,
        attn_heads: int,
        head_dim: int,
        emb_to_k: Callable[[torch.Tensor], torch.Tensor],
        emb_to_v: Callable[[torch.Tensor], torch.Tensor],
    ):
        assert hidden_states.ndim == 3
        batch_size, sequence_length, inner_dim = hidden_states.shape
        ip_k = emb_to_k(cond_uncond_image_emb)
        ip_v = emb_to_v(cond_uncond_image_emb)

        ip_k, ip_v = map(
            lambda t: t.view(batch_size, -1, attn_heads, head_dim).transpose(1, 2),
            (ip_k, ip_v),
        )
        assert ip_k.dtype == ip_v.dtype

        # On MacOS, q can be float16 instead of float32.
        # https://github.com/Mikubill/sd-webui-controlnet/issues/2208
        if query.dtype != ip_k.dtype:
            ip_k = ip_k.to(dtype=query.dtype)
            ip_v = ip_v.to(dtype=query.dtype)

        ip_out = torch.nn.functional.scaled_dot_product_attention(
            query, ip_k, ip_v, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        ip_out = ip_out.transpose(1, 2).reshape(batch_size, -1, attn_heads * head_dim)
        return ip_out

    @torch.no_grad()
    def patch_forward(self, number: int, transformer_index: int):
        @torch.no_grad()
        def forward(attn_blk, x, q):
            batch_size, sequence_length, inner_dim = x.shape
            h = attn_blk.heads
            head_dim = inner_dim // h
            weight = self.weight_on_transformer(transformer_index)

            current_sampling_percent = getattr(
                current_model, "current_sampling_percent", 0.5
            )
            if (
                current_sampling_percent < self.p_start
                or current_sampling_percent > self.p_end
                or weight == 0.0
            ):
                return 0.0

            k_key = f"{number * 2 + 1}_to_k_ip"
            v_key = f"{number * 2 + 1}_to_v_ip"
            ip_out = self.attn_eval(
                hidden_states=x,
                query=q,
                cond_uncond_image_emb=self.image_emb.eval(current_model.cond_mark),
                attn_heads=h,
                head_dim=head_dim,
                emb_to_k=lambda emb: self.call_ip(k_key, emb, device=q.device),
                emb_to_v=lambda emb: self.call_ip(v_key, emb, device=q.device),
            )
            return self.apply_effective_region_mask(ip_out * weight)

        return forward
