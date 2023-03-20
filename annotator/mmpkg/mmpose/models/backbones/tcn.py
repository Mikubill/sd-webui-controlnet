# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch.nn as nn
from annotator.mmpkg.mmcv.cnn import ConvModule, build_conv_layer, constant_init, kaiming_init
from annotator.mmpkg.mmcv.utils.parrots_wrapper import _BatchNorm

from annotator.mmpkg.mmpose.core import WeightNormClipHook
from ..builder import BACKBONES
from .base_backbone import BaseBackbone


class BasicTemporalBlock(nn.Module):
    """Basic block for VideoPose3D.

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        mid_channels (int): The output channels of conv1. Default: 1024.
        kernel_size (int): Size of the convolving kernel. Default: 3.
        dilation (int): Spacing between kernel elements. Default: 3.
        dropout (float): Dropout rate. Default: 0.25.
        causal (bool): Use causal convolutions instead of symmetric
            convolutions (for real-time applications). Default: False.
        residual (bool): Use residual connection. Default: True.
        use_stride_conv (bool): Use optimized TCN that designed
            specifically for single-frame batching, i.e. where batches have
            input length = receptive field, and output length = 1. This
            implementation replaces dilated convolutions with strided
            convolutions to avoid generating unused intermediate results.
            Default: False.
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: dict(type='Conv1d').
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN1d').
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=1024,
                 kernel_size=3,
                 dilation=3,
                 dropout=0.25,
                 causal=False,
                 residual=True,
                 use_stride_conv=False,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d')):
        # Protect mutable default arguments
        conv_cfg = copy.deepcopy(conv_cfg)
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dropout = dropout
        self.causal = causal
        self.residual = residual
        self.use_stride_conv = use_stride_conv

        self.pad = (kernel_size - 1) * dilation // 2
        if use_stride_conv:
            self.stride = kernel_size
            self.causal_shift = kernel_size // 2 if causal else 0
            self.dilation = 1
        else:
            self.stride = 1
            self.causal_shift = kernel_size // 2 * dilation if causal else 0

        self.conv1 = nn.Sequential(
            ConvModule(
                in_channels,
                mid_channels,
                kernel_size=kernel_size,
                stride=self.stride,
                dilation=self.dilation,
                bias='auto',
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg))
        self.conv2 = nn.Sequential(
            ConvModule(
                mid_channels,
                out_channels,
                kernel_size=1,
                bias='auto',
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg))

        if residual and in_channels != out_channels:
            self.short_cut = build_conv_layer(conv_cfg, in_channels,
                                              out_channels, 1)
        else:
            self.short_cut = None

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        """Forward function."""
        if self.use_stride_conv:
            assert self.causal_shift + self.kernel_size // 2 < x.shape[2]
        else:
            assert 0 <= self.pad + self.causal_shift < x.shape[2] - \
                self.pad + self.causal_shift <= x.shape[2]

        out = self.conv1(x)
        if self.dropout is not None:
            out = self.dropout(out)

        out = self.conv2(out)
        if self.dropout is not None:
            out = self.dropout(out)

        if self.residual:
            if self.use_stride_conv:
                res = x[:, :, self.causal_shift +
                        self.kernel_size // 2::self.kernel_size]
            else:
                res = x[:, :,
                        (self.pad + self.causal_shift):(x.shape[2] - self.pad +
                                                        self.causal_shift)]

            if self.short_cut is not None:
                res = self.short_cut(res)
            out = out + res

        return out


@BACKBONES.register_module()
class TCN(BaseBackbone):
    """TCN backbone.

    Temporal Convolutional Networks.
    More details can be found in the
    `paper <https://arxiv.org/abs/1811.11742>`__ .

    Args:
        in_channels (int): Number of input channels, which equals to
            num_keypoints * num_features.
        stem_channels (int): Number of feature channels. Default: 1024.
        num_blocks (int): NUmber of basic temporal convolutional blocks.
            Default: 2.
        kernel_sizes (Sequence[int]): Sizes of the convolving kernel of
            each basic block. Default: ``(3, 3, 3)``.
        dropout (float): Dropout rate. Default: 0.25.
        causal (bool): Use causal convolutions instead of symmetric
            convolutions (for real-time applications).
            Default: False.
        residual (bool): Use residual connection. Default: True.
        use_stride_conv (bool): Use TCN backbone optimized for
            single-frame batching, i.e. where batches have input length =
            receptive field, and output length = 1. This implementation
            replaces dilated convolutions with strided convolutions to avoid
            generating unused intermediate results. The weights are
            interchangeable with the reference implementation. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: dict(type='Conv1d').
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN1d').
        max_norm (float|None): if not None, the weight of convolution layers
            will be clipped to have a maximum norm of max_norm.

    Example:
        >>> from annotator.mmpkg.mmpose.models import TCN
        >>> import torch
        >>> self = TCN(in_channels=34)
        >>> self.eval()
        >>> inputs = torch.rand(1, 34, 243)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 1024, 235)
        (1, 1024, 217)
    """

    def __init__(self,
                 in_channels,
                 stem_channels=1024,
                 num_blocks=2,
                 kernel_sizes=(3, 3, 3),
                 dropout=0.25,
                 causal=False,
                 residual=True,
                 use_stride_conv=False,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 max_norm=None):
        # Protect mutable default arguments
        conv_cfg = copy.deepcopy(conv_cfg)
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__()
        self.in_channels = in_channels
        self.stem_channels = stem_channels
        self.num_blocks = num_blocks
        self.kernel_sizes = kernel_sizes
        self.dropout = dropout
        self.causal = causal
        self.residual = residual
        self.use_stride_conv = use_stride_conv
        self.max_norm = max_norm

        assert num_blocks == len(kernel_sizes) - 1
        for ks in kernel_sizes:
            assert ks % 2 == 1, 'Only odd filter widths are supported.'

        self.expand_conv = ConvModule(
            in_channels,
            stem_channels,
            kernel_size=kernel_sizes[0],
            stride=kernel_sizes[0] if use_stride_conv else 1,
            bias='auto',
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)

        dilation = kernel_sizes[0]
        self.tcn_blocks = nn.ModuleList()
        for i in range(1, num_blocks + 1):
            self.tcn_blocks.append(
                BasicTemporalBlock(
                    in_channels=stem_channels,
                    out_channels=stem_channels,
                    mid_channels=stem_channels,
                    kernel_size=kernel_sizes[i],
                    dilation=dilation,
                    dropout=dropout,
                    causal=causal,
                    residual=residual,
                    use_stride_conv=use_stride_conv,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))
            dilation *= kernel_sizes[i]

        if self.max_norm is not None:
            # Apply weight norm clip to conv layers
            weight_clip = WeightNormClipHook(self.max_norm)
            for module in self.modules():
                if isinstance(module, nn.modules.conv._ConvNd):
                    weight_clip.register(module)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        """Forward function."""
        x = self.expand_conv(x)

        if self.dropout is not None:
            x = self.dropout(x)

        outs = []
        for i in range(self.num_blocks):
            x = self.tcn_blocks[i](x)
            outs.append(x)

        return tuple(outs)

    def init_weights(self, pretrained=None):
        """Initialize the weights."""
        super().init_weights(pretrained)
        if pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.modules.conv._ConvNd):
                    kaiming_init(m, mode='fan_in', nonlinearity='relu')
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)
