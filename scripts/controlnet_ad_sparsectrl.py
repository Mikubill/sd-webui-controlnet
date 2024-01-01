from typing import Tuple, List

import torch
from torch import nn
from torch.nn import functional as F

from modules.devices import get_device_for
from scripts.cldm import PlugableControlModel, ControlNet, zero_module, conv_nd

class PlugableSparseCtrlModel(PlugableControlModel):
    def __init__(self, config, state_dict=None):
        self.config = config
        self.control_model = SparseCtrl(**self.config).cpu()
        if state_dict is not None:
            self.control_model.load_state_dict(state_dict, strict=False)
        self.gpu_component = None


class SparseControlNetConditioningEmbedding(nn.Module):
    def __init__(
        self,
        dims: int,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int] = (16, 32, 96, 256),
    ):
        super().__init__()

        self.conv_in = conv_nd(dims, conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(conv_nd(dims, channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(conv_nd(dims, channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.conv_out = zero_module(conv_nd(dims, block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1))

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding


class SparseCtrl(ControlNet):
    def __init__(self, use_simplified_condition_embedding=True, conditioning_channels=4, **kwargs):
        super().__init__(**kwargs)
        if use_simplified_condition_embedding:
            self.input_hint_block = zero_module(conv_nd(self.dims, conditioning_channels, kwargs.get("model_channels", 320), kernel_size=3, padding=1))
        else:
            self.input_hint_block = SparseControlNetConditioningEmbedding(
                self.dims,
                kwargs.get("model_channels", 320),
                conditioning_channels=conditioning_channels,
            )


    def load_state_dict(self, state_dict, strict=False):
        missing_keys_cn = super().load_state_dict(state_dict, strict=strict)
        from scripts.logging import logger # TODO: remove this
        logger.warn(f"Missing keys in CN part of SparseCtrl {missing_keys_cn}") # TODO: remove this

        from scripts.animatediff_mm import MotionWrapper, MotionModuleType
        sparsectrl_mm = MotionWrapper("", "", MotionModuleType.SparseCtrl)
        missing_keys_ad = sparsectrl_mm.load_state_dict(state_dict, strict=strict)
        logger.warn(f"Missing keys in MM part of SparseCtrl {missing_keys_ad}") # TODO: remove this

        for mm_idx, unet_idx in enumerate([1, 2, 4, 5, 7, 8, 10, 11]):
            mm_idx0, mm_idx1 = mm_idx // 2, mm_idx % 2
            mm_inject = getattr(sparsectrl_mm.down_blocks[mm_idx0], "motion_modules")[mm_idx1]
            self.input_blocks[unet_idx].append(mm_inject)


    @staticmethod
    def create_cond_mask(control_image_index: List[int], control_image_latents: torch.Tensor, video_length: int):
        hint_cond = torch.zeros((video_length, *control_image_latents.shape[1:]), device=get_device_for("controlnet"))
        hint_cond[control_image_index] = control_image_latents[:len(control_image_index)]
        hint_cond_mask = torch.zeros((hint_cond.shape[0], 1, hint_cond.shape[2:]), device=get_device_for("controlnet"))
        hint_cond_mask[control_image_index] = 1.0
        return torch.cat([hint_cond, hint_cond_mask], dim=1)
