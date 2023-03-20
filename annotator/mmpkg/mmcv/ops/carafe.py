# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Function
from torch.nn.modules.module import Module

from ..cnn import UPSAMPLE_LAYERS, normal_init, xavier_init
from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', [
    'carafe_naive_forward', 'carafe_naive_backward', 'carafe_forward',
    'carafe_backward'
])


class CARAFENaiveFunction(Function):

    @staticmethod
    def symbolic(g, features: Tensor, masks: Tensor, kernel_size: int,
                 group_size: int, scale_factor: int) -> Tensor:
        return g.op(
            'mmcv::MMCVCARAFENaive',
            features,
            masks,
            kernel_size_i=kernel_size,
            group_size_i=group_size,
            scale_factor_f=scale_factor)

    @staticmethod
    def forward(ctx, features: Tensor, masks: Tensor, kernel_size: int,
                group_size: int, scale_factor: int) -> Tensor:
        assert scale_factor >= 1
        assert masks.size(1) == kernel_size * kernel_size * group_size
        assert masks.size(-1) == features.size(-1) * scale_factor
        assert masks.size(-2) == features.size(-2) * scale_factor
        assert features.size(1) % group_size == 0
        assert (kernel_size - 1) % 2 == 0 and kernel_size >= 1
        ctx.kernel_size = kernel_size
        ctx.group_size = group_size
        ctx.scale_factor = scale_factor
        ctx.feature_size = features.size()
        ctx.mask_size = masks.size()

        n, c, h, w = features.size()
        output = features.new_zeros((n, c, h * scale_factor, w * scale_factor))
        ext_module.carafe_naive_forward(
            features,
            masks,
            output,
            kernel_size=kernel_size,
            group_size=group_size,
            scale_factor=scale_factor)

        if features.requires_grad or masks.requires_grad or \
                torch.__version__ == 'parrots':
            ctx.save_for_backward(features, masks)
        return output

    @staticmethod
    def backward(
            ctx,
            grad_output: Tensor) -> Tuple[Tensor, Tensor, None, None, None]:
        assert grad_output.is_cuda

        features, masks = ctx.saved_tensors
        kernel_size = ctx.kernel_size
        group_size = ctx.group_size
        scale_factor = ctx.scale_factor

        grad_input = torch.zeros_like(features)
        grad_masks = torch.zeros_like(masks)
        ext_module.carafe_naive_backward(
            grad_output.contiguous(),
            features,
            masks,
            grad_input,
            grad_masks,
            kernel_size=kernel_size,
            group_size=group_size,
            scale_factor=scale_factor)

        return grad_input, grad_masks, None, None, None


carafe_naive = CARAFENaiveFunction.apply


class CARAFENaive(Module):

    def __init__(self, kernel_size: int, group_size: int, scale_factor: int):
        super().__init__()

        assert isinstance(kernel_size, int) and isinstance(
            group_size, int) and isinstance(scale_factor, int)
        self.kernel_size = kernel_size
        self.group_size = group_size
        self.scale_factor = scale_factor

    def forward(self, features: Tensor, masks: Tensor) -> Tensor:
        return carafe_naive(features, masks, self.kernel_size, self.group_size,
                            self.scale_factor)


class CARAFEFunction(Function):

    @staticmethod
    def symbolic(g, features: Tensor, masks: Tensor, kernel_size: int,
                 group_size: int, scale_factor: int) -> Tensor:
        return g.op(
            'mmcv::MMCVCARAFE',
            features,
            masks,
            kernel_size_i=kernel_size,
            group_size_i=group_size,
            scale_factor_f=scale_factor)

    @staticmethod
    def forward(ctx, features: Tensor, masks: Tensor, kernel_size: int,
                group_size: int, scale_factor: int) -> Tensor:
        assert scale_factor >= 1
        assert masks.size(1) == kernel_size * kernel_size * group_size
        assert masks.size(-1) == features.size(-1) * scale_factor
        assert masks.size(-2) == features.size(-2) * scale_factor
        assert features.size(1) % group_size == 0
        assert (kernel_size - 1) % 2 == 0 and kernel_size >= 1
        ctx.kernel_size = kernel_size
        ctx.group_size = group_size
        ctx.scale_factor = scale_factor
        ctx.feature_size = features.size()
        ctx.mask_size = masks.size()

        n, c, h, w = features.size()
        output = features.new_zeros((n, c, h * scale_factor, w * scale_factor))
        routput = features.new_zeros(output.size(), requires_grad=False)
        rfeatures = features.new_zeros(features.size(), requires_grad=False)
        rmasks = masks.new_zeros(masks.size(), requires_grad=False)
        ext_module.carafe_forward(
            features,
            masks,
            rfeatures,
            routput,
            rmasks,
            output,
            kernel_size=kernel_size,
            group_size=group_size,
            scale_factor=scale_factor)

        if features.requires_grad or masks.requires_grad or \
                torch.__version__ == 'parrots':
            ctx.save_for_backward(features, masks, rfeatures)
        return output

    @staticmethod
    def backward(
            ctx,
            grad_output: Tensor) -> Tuple[Tensor, Tensor, None, None, None]:
        features, masks, rfeatures = ctx.saved_tensors
        kernel_size = ctx.kernel_size
        group_size = ctx.group_size
        scale_factor = ctx.scale_factor

        rgrad_output = torch.zeros_like(grad_output, requires_grad=False)
        rgrad_input_hs = torch.zeros_like(grad_output, requires_grad=False)
        rgrad_input = torch.zeros_like(features, requires_grad=False)
        rgrad_masks = torch.zeros_like(masks, requires_grad=False)
        grad_input = torch.zeros_like(features, requires_grad=False)
        grad_masks = torch.zeros_like(masks, requires_grad=False)
        ext_module.carafe_backward(
            grad_output.contiguous(),
            rfeatures,
            masks,
            rgrad_output,
            rgrad_input_hs,
            rgrad_input,
            rgrad_masks,
            grad_input,
            grad_masks,
            kernel_size=kernel_size,
            group_size=group_size,
            scale_factor=scale_factor)
        return grad_input, grad_masks, None, None, None


carafe = CARAFEFunction.apply


class CARAFE(Module):
    """ CARAFE: Content-Aware ReAssembly of FEatures

    Please refer to `CARAFE: Content-Aware ReAssembly of FEatures
    <https://arxiv.org/abs/1905.02188>`_ for more details.

    Args:
        kernel_size (int): reassemble kernel size
        group_size (int): reassemble group size
        scale_factor (int): upsample ratio

    Returns:
        upsampled feature map
    """

    def __init__(self, kernel_size: int, group_size: int, scale_factor: int):
        super().__init__()

        assert isinstance(kernel_size, int) and isinstance(
            group_size, int) and isinstance(scale_factor, int)
        self.kernel_size = kernel_size
        self.group_size = group_size
        self.scale_factor = scale_factor

    def forward(self, features: Tensor, masks: Tensor) -> Tensor:
        return carafe(features, masks, self.kernel_size, self.group_size,
                      self.scale_factor)


@UPSAMPLE_LAYERS.register_module(name='carafe')
class CARAFEPack(nn.Module):
    """A unified package of CARAFE upsampler that contains: 1) channel
    compressor 2) content encoder 3) CARAFE op.

    Official implementation of ICCV 2019 paper
    `CARAFE: Content-Aware ReAssembly of FEatures
    <https://arxiv.org/abs/1905.02188>`_.

    Args:
        channels (int): input feature channels
        scale_factor (int): upsample ratio
        up_kernel (int): kernel size of CARAFE op
        up_group (int): group size of CARAFE op
        encoder_kernel (int): kernel size of content encoder
        encoder_dilation (int): dilation of content encoder
        compressed_channels (int): output channels of channels compressor

    Returns:
        upsampled feature map
    """

    def __init__(self,
                 channels: int,
                 scale_factor: int,
                 up_kernel: int = 5,
                 up_group: int = 1,
                 encoder_kernel: int = 3,
                 encoder_dilation: int = 1,
                 compressed_channels: int = 64):
        super().__init__()
        self.channels = channels
        self.scale_factor = scale_factor
        self.up_kernel = up_kernel
        self.up_group = up_group
        self.encoder_kernel = encoder_kernel
        self.encoder_dilation = encoder_dilation
        self.compressed_channels = compressed_channels
        self.channel_compressor = nn.Conv2d(channels, self.compressed_channels,
                                            1)
        self.content_encoder = nn.Conv2d(
            self.compressed_channels,
            self.up_kernel * self.up_kernel * self.up_group *
            self.scale_factor * self.scale_factor,
            self.encoder_kernel,
            padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
            dilation=self.encoder_dilation,
            groups=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        normal_init(self.content_encoder, std=0.001)

    def kernel_normalizer(self, mask: Tensor) -> Tensor:
        mask = F.pixel_shuffle(mask, self.scale_factor)
        n, mask_c, h, w = mask.size()
        # use float division explicitly,
        # to void inconsistency while exporting to onnx
        mask_channel = int(mask_c / float(self.up_kernel**2))
        mask = mask.view(n, mask_channel, -1, h, w)

        mask = F.softmax(mask, dim=2, dtype=mask.dtype)
        mask = mask.view(n, mask_c, h, w).contiguous()

        return mask

    def feature_reassemble(self, x: Tensor, mask: Tensor) -> Tensor:
        x = carafe(x, mask, self.up_kernel, self.up_group, self.scale_factor)
        return x

    def forward(self, x: Tensor) -> Tensor:
        compressed_x = self.channel_compressor(x)
        mask = self.content_encoder(compressed_x)
        mask = self.kernel_normalizer(mask)

        x = self.feature_reassemble(x, mask)
        return x
