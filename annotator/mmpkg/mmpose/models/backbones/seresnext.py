# Copyright (c) OpenMMLab. All rights reserved.
from annotator.mmpkg.mmcv.cnn import build_conv_layer, build_norm_layer

from ..builder import BACKBONES
from .resnet import ResLayer
from .seresnet import SEBottleneck as _SEBottleneck
from .seresnet import SEResNet


class SEBottleneck(_SEBottleneck):
    """SEBottleneck block for SEResNeXt.

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        base_channels (int): Middle channels of the first stage. Default: 64.
        groups (int): Groups of conv2.
        width_per_group (int): Width per group of conv2. 64x4d indicates
            ``groups=64, width_per_group=4`` and 32x8d indicates
            ``groups=32, width_per_group=8``.
        stride (int): stride of the block. Default: 1
        dilation (int): dilation of convolution. Default: 1
        downsample (nn.Module): downsample operation on identity branch.
            Default: None
        se_ratio (int): Squeeze ratio in SELayer. Default: 16
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
                 base_channels=64,
                 groups=32,
                 width_per_group=4,
                 se_ratio=16,
                 **kwargs):
        super().__init__(in_channels, out_channels, se_ratio, **kwargs)
        self.groups = groups
        self.width_per_group = width_per_group

        # We follow the same rational of ResNext to compute mid_channels.
        # For SEResNet bottleneck, middle channels are determined by expansion
        # and out_channels, but for SEResNeXt bottleneck, it is determined by
        # groups and width_per_group and the stage it is located in.
        if groups != 1:
            assert self.mid_channels % base_channels == 0
            self.mid_channels = (
                groups * width_per_group * self.mid_channels // base_channels)

        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, self.mid_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            self.norm_cfg, self.mid_channels, postfix=2)
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
        self.conv2 = build_conv_layer(
            self.conv_cfg,
            self.mid_channels,
            self.mid_channels,
            kernel_size=3,
            stride=self.conv2_stride,
            padding=self.dilation,
            dilation=self.dilation,
            groups=groups,
            bias=False)

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            self.conv_cfg,
            self.mid_channels,
            self.out_channels,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)


@BACKBONES.register_module()
class SEResNeXt(SEResNet):
    """SEResNeXt backbone.

    Please refer to the `paper <https://arxiv.org/abs/1709.01507>`__ for
    details.

    Args:
        depth (int): Network depth, from {50, 101, 152}.
        groups (int): Groups of conv2 in Bottleneck. Default: 32.
        width_per_group (int): Width per group of conv2 in Bottleneck.
            Default: 4.
        se_ratio (int): Squeeze ratio in SELayer. Default: 16.
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

    Example:
        >>> from annotator.mmpkg.mmpose.models import SEResNeXt
        >>> import torch
        >>> self = SEResNet(depth=50, out_indices=(0, 1, 2, 3))
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
        50: (SEBottleneck, (3, 4, 6, 3)),
        101: (SEBottleneck, (3, 4, 23, 3)),
        152: (SEBottleneck, (3, 8, 36, 3))
    }

    def __init__(self, depth, groups=32, width_per_group=4, **kwargs):
        self.groups = groups
        self.width_per_group = width_per_group
        super().__init__(depth, **kwargs)

    def make_res_layer(self, **kwargs):
        return ResLayer(
            groups=self.groups,
            width_per_group=self.width_per_group,
            base_channels=self.base_channels,
            **kwargs)
