# Copyright (c) OpenMMLab. All rights reserved.
import torch.utils.checkpoint as cp

from ..builder import BACKBONES
from .resnet import Bottleneck, ResLayer, ResNet
from .utils.se_layer import SELayer


class SEBottleneck(Bottleneck):
    """SEBottleneck block for SEResNet.

    Args:
        in_channels (int): The input channels of the SEBottleneck block.
        out_channels (int): The output channel of the SEBottleneck block.
        se_ratio (int): Squeeze ratio in SELayer. Default: 16
    """

    def __init__(self, in_channels, out_channels, se_ratio=16, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)
        self.se_layer = SELayer(out_channels, ratio=se_ratio)

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.norm3(out)

            out = self.se_layer(out)

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
class SEResNet(ResNet):
    """SEResNet backbone.

    Please refer to the `paper <https://arxiv.org/abs/1709.01507>`__ for
    details.

    Args:
        depth (int): Network depth, from {50, 101, 152}.
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
        >>> from annotator.mmpkg.mmpose.models import SEResNet
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

    def __init__(self, depth, se_ratio=16, **kwargs):
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for SEResNet')
        self.se_ratio = se_ratio
        super().__init__(depth, **kwargs)

    def make_res_layer(self, **kwargs):
        return ResLayer(se_ratio=self.se_ratio, **kwargs)
