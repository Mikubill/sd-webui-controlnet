# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Sequence

import torch.nn as nn
import torch.nn.functional as F
from annotator.mmpkg.mmcv.cnn import build_conv_layer, build_norm_layer
from annotator.mmpkg.mmcv.runner.base_module import BaseModule
from annotator.mmpkg.mmcv.utils import to_2tuple


def nlc_to_nchw(x, hw_shape):
    """Convert [N, L, C] shape tensor to [N, C, H, W] shape tensor.

    Args:
        x (Tensor): The input tensor of shape [N, L, C] before conversion.
        hw_shape (Sequence[int]): The height and width of output feature map.

    Returns:
        Tensor: The output tensor of shape [N, C, H, W] after conversion.
    """
    H, W = hw_shape
    assert len(x.shape) == 3
    B, L, C = x.shape
    assert L == H * W, 'The seq_len does not match H, W'
    return x.transpose(1, 2).reshape(B, C, H, W).contiguous()


def nchw_to_nlc(x):
    """Flatten [N, C, H, W] shape tensor to [N, L, C] shape tensor.

    Args:
        x (Tensor): The input tensor of shape [N, C, H, W] before conversion.

    Returns:
        Tensor: The output tensor of shape [N, L, C] after conversion.
    """
    assert len(x.shape) == 4
    return x.flatten(2).transpose(1, 2).contiguous()


class AdaptivePadding(nn.Module):
    """Applies padding to input (if needed) so that input can get fully covered
    by filter you specified. It support two modes "same" and "corner". The
    "same" mode is same with "SAME" padding mode in TensorFlow, pad zero around
    input. The "corner"  mode would pad zero to bottom right.

    Args:
        kernel_size (int | tuple): Size of the kernel:
        stride (int | tuple): Stride of the filter. Default: 1:
        dilation (int | tuple): Spacing between kernel elements.
            Default: 1
        padding (str): Support "same" and "corner", "corner" mode
            would pad zero to bottom right, and "same" mode would
            pad zero around input. Default: "corner".
    Example:
        >>> kernel_size = 16
        >>> stride = 16
        >>> dilation = 1
        >>> input = torch.rand(1, 1, 15, 17)
        >>> adap_pad = AdaptivePadding(
        >>>     kernel_size=kernel_size,
        >>>     stride=stride,
        >>>     dilation=dilation,
        >>>     padding="corner")
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
        >>> input = torch.rand(1, 1, 16, 17)
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
    """

    def __init__(self, kernel_size=1, stride=1, dilation=1, padding='corner'):

        super(AdaptivePadding, self).__init__()

        assert padding in ('same', 'corner')

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        dilation = to_2tuple(dilation)

        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

    def get_pad_shape(self, input_shape):
        input_h, input_w = input_shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * stride_h +
                    (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * stride_w +
                    (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        return pad_h, pad_w

    def forward(self, x):
        pad_h, pad_w = self.get_pad_shape(x.size()[-2:])
        if pad_h > 0 or pad_w > 0:
            if self.padding == 'corner':
                x = F.pad(x, [0, pad_w, 0, pad_h])
            elif self.padding == 'same':
                x = F.pad(x, [
                    pad_w // 2, pad_w - pad_w // 2, pad_h // 2,
                    pad_h - pad_h // 2
                ])
        return x


class PatchEmbed(BaseModule):
    """Image to Patch Embedding.

    We use a conv layer to implement PatchEmbed.

    Args:
        in_channels (int): The num of input channels. Default: 3
        embed_dims (int): The dimensions of embedding. Default: 768
        conv_type (str): The config dict for embedding
            conv layer type selection. Default: "Conv2d.
        kernel_size (int): The kernel_size of embedding conv. Default: 16.
        stride (int): The slide stride of embedding conv.
            Default: None (Would be set as `kernel_size`).
        padding (int | tuple | string ): The padding length of
            embedding conv. When it is a string, it means the mode
            of adaptive padding, support "same" and "corner" now.
            Default: "corner".
        dilation (int): The dilation rate of embedding conv. Default: 1.
        bias (bool): Bias of embed conv. Default: True.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        input_size (int | tuple | None): The size of input, which will be
            used to calculate the out size. Only work when `dynamic_size`
            is False. Default: None.
        init_cfg (`mmcv.ConfigDict`, optional): The Config for initialization.
            Default: None.
    """

    def __init__(
        self,
        in_channels=3,
        embed_dims=768,
        conv_type='Conv2d',
        kernel_size=16,
        stride=16,
        padding='corner',
        dilation=1,
        bias=True,
        norm_cfg=None,
        input_size=None,
        init_cfg=None,
    ):
        super(PatchEmbed, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        if stride is None:
            stride = kernel_size

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        if isinstance(padding, str):
            self.adap_padding = AdaptivePadding(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding)
            # disable the padding of conv
            padding = 0
        else:
            self.adap_padding = None
        padding = to_2tuple(padding)

        self.projection = build_conv_layer(
            dict(type=conv_type),
            in_channels=in_channels,
            out_channels=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias)

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm = None

        if input_size:
            input_size = to_2tuple(input_size)
            # `init_out_size` would be used outside to
            # calculate the num_patches
            # when `use_abs_pos_embed` outside
            self.init_input_size = input_size
            if self.adap_padding:
                pad_h, pad_w = self.adap_padding.get_pad_shape(input_size)
                input_h, input_w = input_size
                input_h = input_h + pad_h
                input_w = input_w + pad_w
                input_size = (input_h, input_w)

            # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            h_out = (input_size[0] + 2 * padding[0] - dilation[0] *
                     (kernel_size[0] - 1) - 1) // stride[0] + 1
            w_out = (input_size[1] + 2 * padding[1] - dilation[1] *
                     (kernel_size[1] - 1) - 1) // stride[1] + 1
            self.init_out_size = (h_out, w_out)
        else:
            self.init_input_size = None
            self.init_out_size = None

    def forward(self, x):
        """
        Args:
            x (Tensor): Has shape (B, C, H, W). In most case, C is 3.

        Returns:
            tuple: Contains merged results and its spatial shape.

                - x (Tensor): Has shape (B, out_h * out_w, embed_dims)
                - out_size (tuple[int]): Spatial shape of x, arrange as
                    (out_h, out_w).
        """

        if self.adap_padding:
            x = self.adap_padding(x)

        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x, out_size


class PatchMerging(BaseModule):
    """Merge patch feature map.

    This layer groups feature map by kernel_size, and applies norm and linear
    layers to the grouped feature map. Our implementation uses `nn.Unfold` to
    merge patch, which is about 25% faster than original implementation.
    Instead, we need to modify pretrained models for compatibility.

    Args:
        in_channels (int): The num of input channels.
            to gets fully covered by filter and stride you specified..
            Default: True.
        out_channels (int): The num of output channels.
        kernel_size (int | tuple, optional): the kernel size in the unfold
            layer. Defaults to 2.
        stride (int | tuple, optional): the stride of the sliding blocks in the
            unfold layer. Default: None. (Would be set as `kernel_size`)
        padding (int | tuple | string ): The padding length of
            embedding conv. When it is a string, it means the mode
            of adaptive padding, support "same" and "corner" now.
            Default: "corner".
        dilation (int | tuple, optional): dilation parameter in the unfold
            layer. Default: 1.
        bias (bool, optional): Whether to add bias in linear layer or not.
            Defaults: False.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: dict(type='LN').
        init_cfg (dict, optional): The extra config for initialization.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=2,
                 stride=None,
                 padding='corner',
                 dilation=1,
                 bias=False,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        if stride:
            stride = stride
        else:
            stride = kernel_size

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        if isinstance(padding, str):
            self.adap_padding = AdaptivePadding(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding)
            # disable the padding of unfold
            padding = 0
        else:
            self.adap_padding = None

        padding = to_2tuple(padding)
        self.sampler = nn.Unfold(
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride)

        sample_dim = kernel_size[0] * kernel_size[1] * in_channels

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, sample_dim)[1]
        else:
            self.norm = None

        self.reduction = nn.Linear(sample_dim, out_channels, bias=bias)

    def forward(self, x, input_size):
        """
        Args:
            x (Tensor): Has shape (B, H*W, C_in).
            input_size (tuple[int]): The spatial shape of x, arrange as (H, W).
                Default: None.

        Returns:
            tuple: Contains merged results and its spatial shape.

                - x (Tensor): Has shape (B, Merged_H * Merged_W, C_out)
                - out_size (tuple[int]): Spatial shape of x, arrange as
                    (Merged_H, Merged_W).
        """
        B, L, C = x.shape
        assert isinstance(input_size, Sequence), f'Expect ' \
                                                 f'input_size is ' \
                                                 f'`Sequence` ' \
                                                 f'but get {input_size}'

        H, W = input_size
        assert L == H * W, 'input feature has wrong size'

        x = x.view(B, H, W, C).permute([0, 3, 1, 2])  # B, C, H, W
        # Use nn.Unfold to merge patch. About 25% faster than original method,
        # but need to modify pretrained model for compatibility

        if self.adap_padding:
            x = self.adap_padding(x)
            H, W = x.shape[-2:]

        x = self.sampler(x)
        # if kernel_size=2 and stride=2, x should has shape (B, 4*C, H/2*W/2)

        out_h = (H + 2 * self.sampler.padding[0] - self.sampler.dilation[0] *
                 (self.sampler.kernel_size[0] - 1) -
                 1) // self.sampler.stride[0] + 1
        out_w = (W + 2 * self.sampler.padding[1] - self.sampler.dilation[1] *
                 (self.sampler.kernel_size[1] - 1) -
                 1) // self.sampler.stride[1] + 1

        output_size = (out_h, out_w)
        x = x.transpose(1, 2)  # B, H/2*W/2, 4*C
        x = self.norm(x) if self.norm else x
        x = self.reduction(x)
        return x, output_size
