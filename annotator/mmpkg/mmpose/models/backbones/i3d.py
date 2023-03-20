# Copyright (c) OpenMMLab. All rights reserved.
# Code is modified from `Third-party pytorch implementation of i3d
# <https://github.com/hassony2/kinetics_i3d_pytorch>`.

import torch
import torch.nn as nn

from ..builder import BACKBONES
from .base_backbone import BaseBackbone


class Conv3dBlock(nn.Module):
    """Basic 3d convolution block for I3D.

    Args:
    in_channels (int): Input channels of this block.
    out_channels (int): Output channels of this block.
    expansion (float): The multiplier of in_channels and out_channels.
        Default: 1.
    kernel_size (tuple[int]): kernel size of the 3d convolution layer.
        Default: (1, 1, 1).
    stride (tuple[int]): stride of the block. Default: (1, 1, 1)
    padding (tuple[int]): padding of the input tensor. Default: (0, 0, 0)
    use_bias (bool): whether to enable bias in 3d convolution layer.
        Default: False
    use_bn (bool): whether to use Batch Normalization after 3d convolution
        layer. Default: True
    use_relu (bool): whether to use ReLU after Batch Normalization layer.
        Default: True
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=1.0,
                 kernel_size=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=(0, 0, 0),
                 use_bias=False,
                 use_bn=True,
                 use_relu=True):
        super().__init__()

        in_channels = int(in_channels * expansion)
        out_channels = int(out_channels * expansion)

        self.conv3d = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            stride=stride,
            bias=use_bias)

        self.use_bn = use_bn
        self.use_relu = use_relu

        if self.use_bn:
            self.batch3d = nn.BatchNorm3d(out_channels)

        if self.use_relu:
            self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward function."""
        out = self.conv3d(x)
        if self.use_bn:
            out = self.batch3d(out)
        if self.use_relu:
            out = self.activation(out)
        return out


class Mixed(nn.Module):
    """Inception block for I3D.

    Args:
    in_channels (int): Input channels of this block.
    out_channels (int): Output channels of this block.
    expansion (float): The multiplier of in_channels and out_channels.
        Default: 1.
    """

    def __init__(self, in_channels, out_channels, expansion=1.0):
        super(Mixed, self).__init__()
        # Branch 0
        self.branch_0 = Conv3dBlock(
            in_channels, out_channels[0], expansion, kernel_size=(1, 1, 1))

        # Branch 1
        branch_1_conv1 = Conv3dBlock(
            in_channels, out_channels[1], expansion, kernel_size=(1, 1, 1))
        branch_1_conv2 = Conv3dBlock(
            out_channels[1],
            out_channels[2],
            expansion,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1))
        self.branch_1 = nn.Sequential(branch_1_conv1, branch_1_conv2)

        # Branch 2
        branch_2_conv1 = Conv3dBlock(
            in_channels, out_channels[3], expansion, kernel_size=(1, 1, 1))
        branch_2_conv2 = Conv3dBlock(
            out_channels[3],
            out_channels[4],
            expansion,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1))
        self.branch_2 = nn.Sequential(branch_2_conv1, branch_2_conv2)

        # Branch3
        branch_3_pool = nn.MaxPool3d(
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            ceil_mode=True)
        branch_3_conv2 = Conv3dBlock(
            in_channels, out_channels[5], expansion, kernel_size=(1, 1, 1))
        self.branch_3 = nn.Sequential(branch_3_pool, branch_3_conv2)

    def forward(self, x):
        """Forward function."""
        out_0 = self.branch_0(x)
        out_1 = self.branch_1(x)
        out_2 = self.branch_2(x)
        out_3 = self.branch_3(x)
        out = torch.cat((out_0, out_1, out_2, out_3), 1)
        return out


@BACKBONES.register_module()
class I3D(BaseBackbone):
    """I3D backbone.

    Please refer to the `paper <https://arxiv.org/abs/1705.07750>`__ for
    details.

    Args:
    in_channels (int): Input channels of the backbone, which is decided
        on the input modality.
    expansion (float): The multiplier of in_channels and out_channels.
        Default: 1.
    """

    def __init__(self, in_channels=3, expansion=1.0):
        super(I3D, self).__init__()

        # expansion must be an integer multiple of 1/8
        expansion = round(8 * expansion) / 8.0

        # xut Layer
        self.conv3d_1a_7x7 = Conv3dBlock(
            out_channels=64,
            in_channels=in_channels / expansion,
            expansion=expansion,
            kernel_size=(7, 7, 7),
            stride=(2, 2, 2),
            padding=(2, 3, 3))
        self.maxPool3d_2a_3x3 = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        # Layer 2
        self.conv3d_2b_1x1 = Conv3dBlock(
            out_channels=64,
            in_channels=64,
            expansion=expansion,
            kernel_size=(1, 1, 1))
        self.conv3d_2c_3x3 = Conv3dBlock(
            out_channels=192,
            in_channels=64,
            expansion=expansion,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1))
        self.maxPool3d_3a_3x3 = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        # Mixed_3b
        self.mixed_3b = Mixed(192, [64, 96, 128, 16, 32, 32], expansion)
        self.mixed_3c = Mixed(256, [128, 128, 192, 32, 96, 64], expansion)
        self.maxPool3d_4a_3x3 = nn.MaxPool3d(
            kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        # Mixed 4
        self.mixed_4b = Mixed(480, [192, 96, 208, 16, 48, 64], expansion)
        self.mixed_4c = Mixed(512, [160, 112, 224, 24, 64, 64], expansion)
        self.mixed_4d = Mixed(512, [128, 128, 256, 24, 64, 64], expansion)
        self.mixed_4e = Mixed(512, [112, 144, 288, 32, 64, 64], expansion)
        self.mixed_4f = Mixed(528, [256, 160, 320, 32, 128, 128], expansion)

        self.maxPool3d_5a_2x2 = nn.MaxPool3d(
            kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0))

        # Mixed 5
        self.mixed_5b = Mixed(832, [256, 160, 320, 32, 128, 128], expansion)
        self.mixed_5c = Mixed(832, [384, 192, 384, 48, 128, 128], expansion)

    def forward(self, x):
        out = self.conv3d_1a_7x7(x)
        out = self.maxPool3d_2a_3x3(out)
        out = self.conv3d_2b_1x1(out)
        out = self.conv3d_2c_3x3(out)
        out = self.maxPool3d_3a_3x3(out)
        out = self.mixed_3b(out)
        out = self.mixed_3c(out)
        out = self.maxPool3d_4a_3x3(out)
        out = self.mixed_4b(out)
        out = self.mixed_4c(out)
        out = self.mixed_4d(out)
        out = self.mixed_4e(out)
        out = self.mixed_4f(out)
        out = self.maxPool3d_5a_2x2(out)
        out = self.mixed_5b(out)
        out = self.mixed_5c(out)
        return out
