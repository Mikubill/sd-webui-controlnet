# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from annotator.mmpkg.mmcv.cnn import build_conv_layer, build_norm_layer

from ..builder import BACKBONES
from .resnet import Bottleneck, ResNet


class SCConv(nn.Module):
    """SCConv (Self-calibrated Convolution)

    Args:
        in_channels (int): The input channels of the SCConv.
        out_channels (int): The output channel of the SCConv.
        stride (int): stride of SCConv.
        pooling_r (int): size of pooling for scconv.
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 pooling_r,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.1)):
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__()

        assert in_channels == out_channels

        self.k2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
            build_conv_layer(
                conv_cfg,
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            build_norm_layer(norm_cfg, in_channels)[1],
        )
        self.k3 = nn.Sequential(
            build_conv_layer(
                conv_cfg,
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            build_norm_layer(norm_cfg, in_channels)[1],
        )
        self.k4 = nn.Sequential(
            build_conv_layer(
                conv_cfg,
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Forward function."""
        identity = x

        out = torch.sigmoid(
            torch.add(identity, F.interpolate(self.k2(x),
                                              identity.size()[2:])))
        out = torch.mul(self.k3(x), out)
        out = self.k4(out)

        return out


class SCBottleneck(Bottleneck):
    """SC(Self-calibrated) Bottleneck.

    Args:
        in_channels (int): The input channels of the SCBottleneck block.
        out_channels (int): The output channel of the SCBottleneck block.
    """

    pooling_r = 4

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)
        self.mid_channels = out_channels // self.expansion // 2

        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, self.mid_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            self.norm_cfg, self.mid_channels, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            self.norm_cfg, out_channels, postfix=3)

        self.conv1 = build_conv_layer(
            self.conv_cfg,
            in_channels,
            self.mid_channels,
            kernel_size=1,
            stride=1,
            bias=False)
        self.add_module(self.norm1_name, norm1)

        self.k1 = nn.Sequential(
            build_conv_layer(
                self.conv_cfg,
                self.mid_channels,
                self.mid_channels,
                kernel_size=3,
                stride=self.stride,
                padding=1,
                bias=False),
            build_norm_layer(self.norm_cfg, self.mid_channels)[1],
            nn.ReLU(inplace=True))

        self.conv2 = build_conv_layer(
            self.conv_cfg,
            in_channels,
            self.mid_channels,
            kernel_size=1,
            stride=1,
            bias=False)
        self.add_module(self.norm2_name, norm2)

        self.scconv = SCConv(self.mid_channels, self.mid_channels, self.stride,
                             self.pooling_r, self.conv_cfg, self.norm_cfg)

        self.conv3 = build_conv_layer(
            self.conv_cfg,
            self.mid_channels * 2,
            out_channels,
            kernel_size=1,
            stride=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out_a = self.conv1(x)
            out_a = self.norm1(out_a)
            out_a = self.relu(out_a)

            out_a = self.k1(out_a)

            out_b = self.conv2(x)
            out_b = self.norm2(out_b)
            out_b = self.relu(out_b)

            out_b = self.scconv(out_b)

            out = self.conv3(torch.cat([out_a, out_b], dim=1))
            out = self.norm3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


@BACKBONES.register_module()
class SCNet(ResNet):
    """SCNet backbone.

    Improving Convolutional Networks with Self-Calibrated Convolutions,
    Jiang-Jiang Liu, Qibin Hou, Ming-Ming Cheng, Changhu Wang, Jiashi Feng,
    IEEE CVPR, 2020.
    http://mftp.mmcheng.net/Papers/20cvprSCNet.pdf

    Args:
        depth (int): Depth of scnet, from {50, 101}.
        in_channels (int): Number of input image channels. Normally 3.
        base_channels (int): Number of base channels of hidden layer.
        num_stages (int): SCNet stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity.

    Example:
        >>> from annotator.mmpkg.mmpose.models import SCNet
        >>> import torch
        >>> self = SCNet(depth=50, out_indices=(0, 1, 2, 3))
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 224, 224)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 256, 56, 56)
        (1, 512, 28, 28)
        (1, 1024, 14, 14)
        (1, 2048, 7, 7)
    """

    arch_settings = {
        50: (SCBottleneck, [3, 4, 6, 3]),
        101: (SCBottleneck, [3, 4, 23, 3])
    }

    def __init__(self, depth, **kwargs):
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for SCNet')
        super().__init__(depth, **kwargs)
