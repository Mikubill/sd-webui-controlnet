# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from annotator.mmpkg.mmcv.cnn import build_conv_layer, build_norm_layer

from ..builder import BACKBONES
from .resnet import Bottleneck as _Bottleneck
from .resnet import ResLayer, ResNetV1d


class RSoftmax(nn.Module):
    """Radix Softmax module in ``SplitAttentionConv2d``.

    Args:
        radix (int): Radix of input.
        groups (int): Groups of input.
    """

    def __init__(self, radix, groups):
        super().__init__()
        self.radix = radix
        self.groups = groups

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.groups, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class SplitAttentionConv2d(nn.Module):
    """Split-Attention Conv2d.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int | tuple[int]): Same as nn.Conv2d.
        stride (int | tuple[int]): Same as nn.Conv2d.
        padding (int | tuple[int]): Same as nn.Conv2d.
        dilation (int | tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        radix (int): Radix of SpltAtConv2d. Default: 2
        reduction_factor (int): Reduction factor of SplitAttentionConv2d.
            Default: 4.
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 radix=2,
                 reduction_factor=4,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super().__init__()
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.groups = groups
        self.channels = channels
        self.conv = build_conv_layer(
            conv_cfg,
            in_channels,
            channels * radix,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups * radix,
            bias=False)
        self.norm0_name, norm0 = build_norm_layer(
            norm_cfg, channels * radix, postfix=0)
        self.add_module(self.norm0_name, norm0)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = build_conv_layer(
            None, channels, inter_channels, 1, groups=self.groups)
        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, inter_channels, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.fc2 = build_conv_layer(
            None, inter_channels, channels * radix, 1, groups=self.groups)
        self.rsoftmax = RSoftmax(radix, groups)

    @property
    def norm0(self):
        return getattr(self, self.norm0_name)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm0(x)
        x = self.relu(x)

        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            splits = x.view(batch, self.radix, -1, *x.shape[2:])
            gap = splits.sum(dim=1)
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)

        gap = self.norm1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)

        if self.radix > 1:
            attens = atten.view(batch, self.radix, -1, *atten.shape[2:])
            out = torch.sum(attens * splits, dim=1)
        else:
            out = atten * x
        return out.contiguous()


class Bottleneck(_Bottleneck):
    """Bottleneck block for ResNeSt.

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        groups (int): Groups of conv2.
        width_per_group (int): Width per group of conv2. 64x4d indicates
            ``groups=64, width_per_group=4`` and 32x8d indicates
            ``groups=32, width_per_group=8``.
        radix (int): Radix of SpltAtConv2d. Default: 2
        reduction_factor (int): Reduction factor of SplitAttentionConv2d.
            Default: 4.
        avg_down_stride (bool): Whether to use average pool for stride in
            Bottleneck. Default: True.
        stride (int): stride of the block. Default: 1
        dilation (int): dilation of convolution. Default: 1
        downsample (nn.Module): downsample operation on identity branch.
            Default: None
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 groups=1,
                 width_per_group=4,
                 base_channels=64,
                 radix=2,
                 reduction_factor=4,
                 avg_down_stride=True,
                 **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)

        self.groups = groups
        self.width_per_group = width_per_group

        # For ResNet bottleneck, middle channels are determined by expansion
        # and out_channels, but for ResNeXt bottleneck, it is determined by
        # groups and width_per_group and the stage it is located in.
        if groups != 1:
            assert self.mid_channels % base_channels == 0
            self.mid_channels = (
                groups * width_per_group * self.mid_channels // base_channels)

        self.avg_down_stride = avg_down_stride and self.conv2_stride > 1

        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, self.mid_channels, postfix=1)
        self.norm3_name, norm3 = build_norm_layer(
            self.norm_cfg, self.out_channels, postfix=3)

        self.conv1 = build_conv_layer(
            self.conv_cfg,
            self.in_channels,
            self.mid_channels,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = SplitAttentionConv2d(
            self.mid_channels,
            self.mid_channels,
            kernel_size=3,
            stride=1 if self.avg_down_stride else self.conv2_stride,
            padding=self.dilation,
            dilation=self.dilation,
            groups=groups,
            radix=radix,
            reduction_factor=reduction_factor,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg)
        delattr(self, self.norm2_name)

        if self.avg_down_stride:
            self.avd_layer = nn.AvgPool2d(3, self.conv2_stride, padding=1)

        self.conv3 = build_conv_layer(
            self.conv_cfg,
            self.mid_channels,
            self.out_channels,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)

            if self.avg_down_stride:
                out = self.avd_layer(out)

            out = self.conv3(out)
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
class ResNeSt(ResNetV1d):
    """ResNeSt backbone.

    Please refer to the `paper <https://arxiv.org/pdf/2004.08955.pdf>`__
    for details.

    Args:
        depth (int): Network depth, from {50, 101, 152, 200}.
        groups (int): Groups of conv2 in Bottleneck. Default: 32.
        width_per_group (int): Width per group of conv2 in Bottleneck.
            Default: 4.
        radix (int): Radix of SpltAtConv2d. Default: 2
        reduction_factor (int): Reduction factor of SplitAttentionConv2d.
            Default: 4.
        avg_down_stride (bool): Whether to use average pool for stride in
            Bottleneck. Default: True.
        in_channels (int): Number of input image channels. Default: 3.
        stem_channels (int): Output channels of the stem layer. Default: 64.
        num_stages (int): Stages of the network. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
            Default: ``(1, 2, 2, 2)``.
        dilations (Sequence[int]): Dilation of each stage.
            Default: ``(1, 1, 1, 1)``.
        out_indices (Sequence[int]): Output from which stages. If only one
            stage is specified, a single tensor (feature map) is returned,
            otherwise multiple stages are specified, a tuple of tensors will
            be returned. Default: ``(3, )``.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv.
            Default: False.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.
        conv_cfg (dict | None): The config dict for conv layers. Default: None.
        norm_cfg (dict): The config dict for norm layers.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity. Default: True.
    """

    arch_settings = {
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3)),
        200: (Bottleneck, (3, 24, 36, 3)),
        269: (Bottleneck, (3, 30, 48, 8))
    }

    def __init__(self,
                 depth,
                 groups=1,
                 width_per_group=4,
                 radix=2,
                 reduction_factor=4,
                 avg_down_stride=True,
                 **kwargs):
        self.groups = groups
        self.width_per_group = width_per_group
        self.radix = radix
        self.reduction_factor = reduction_factor
        self.avg_down_stride = avg_down_stride
        super().__init__(depth=depth, **kwargs)

    def make_res_layer(self, **kwargs):
        return ResLayer(
            groups=self.groups,
            width_per_group=self.width_per_group,
            base_channels=self.base_channels,
            radix=self.radix,
            reduction_factor=self.reduction_factor,
            avg_down_stride=self.avg_down_stride,
            **kwargs)
