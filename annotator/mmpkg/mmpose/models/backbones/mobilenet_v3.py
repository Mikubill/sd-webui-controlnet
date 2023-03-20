# Copyright (c) OpenMMLab. All rights reserved.
import copy
import logging

import torch.nn as nn
from annotator.mmpkg.mmcv.cnn import ConvModule, constant_init, kaiming_init
from torch.nn.modules.batchnorm import _BatchNorm

from ..builder import BACKBONES
from .base_backbone import BaseBackbone
from .utils import InvertedResidual, load_checkpoint


@BACKBONES.register_module()
class MobileNetV3(BaseBackbone):
    """MobileNetV3 backbone.

    Args:
        arch (str): Architecture of mobilnetv3, from {small, big}.
            Default: small.
        conv_cfg (dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        out_indices (None or Sequence[int]): Output from which stages.
            Default: (-1, ), which means output tensors from final stage.
        frozen_stages (int): Stages to be frozen (all param fixed).
            Default: -1, which means not freezing any parameters.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed.
            Default: False.
    """
    # Parameters to build each block:
    #     [kernel size, mid channels, out channels, with_se, act type, stride]
    arch_settings = {
        'small': [[3, 16, 16, True, 'ReLU', 2],
                  [3, 72, 24, False, 'ReLU', 2],
                  [3, 88, 24, False, 'ReLU', 1],
                  [5, 96, 40, True, 'HSwish', 2],
                  [5, 240, 40, True, 'HSwish', 1],
                  [5, 240, 40, True, 'HSwish', 1],
                  [5, 120, 48, True, 'HSwish', 1],
                  [5, 144, 48, True, 'HSwish', 1],
                  [5, 288, 96, True, 'HSwish', 2],
                  [5, 576, 96, True, 'HSwish', 1],
                  [5, 576, 96, True, 'HSwish', 1]],
        'big': [[3, 16, 16, False, 'ReLU', 1],
                [3, 64, 24, False, 'ReLU', 2],
                [3, 72, 24, False, 'ReLU', 1],
                [5, 72, 40, True, 'ReLU', 2],
                [5, 120, 40, True, 'ReLU', 1],
                [5, 120, 40, True, 'ReLU', 1],
                [3, 240, 80, False, 'HSwish', 2],
                [3, 200, 80, False, 'HSwish', 1],
                [3, 184, 80, False, 'HSwish', 1],
                [3, 184, 80, False, 'HSwish', 1],
                [3, 480, 112, True, 'HSwish', 1],
                [3, 672, 112, True, 'HSwish', 1],
                [5, 672, 160, True, 'HSwish', 1],
                [5, 672, 160, True, 'HSwish', 2],
                [5, 960, 160, True, 'HSwish', 1]]
    }  # yapf: disable

    def __init__(self,
                 arch='small',
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 out_indices=(-1, ),
                 frozen_stages=-1,
                 norm_eval=False,
                 with_cp=False):
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__()
        assert arch in self.arch_settings
        for index in out_indices:
            if index not in range(-len(self.arch_settings[arch]),
                                  len(self.arch_settings[arch])):
                raise ValueError('the item in out_indices must in '
                                 f'range(0, {len(self.arch_settings[arch])}). '
                                 f'But received {index}')

        if frozen_stages not in range(-1, len(self.arch_settings[arch])):
            raise ValueError('frozen_stages must be in range(-1, '
                             f'{len(self.arch_settings[arch])}). '
                             f'But received {frozen_stages}')
        self.arch = arch
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.with_cp = with_cp

        self.in_channels = 16
        self.conv1 = ConvModule(
            in_channels=3,
            out_channels=self.in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='HSwish'))

        self.layers = self._make_layer()
        self.feat_dim = self.arch_settings[arch][-1][2]

    def _make_layer(self):
        layers = []
        layer_setting = self.arch_settings[self.arch]
        for i, params in enumerate(layer_setting):
            (kernel_size, mid_channels, out_channels, with_se, act,
             stride) = params
            if with_se:
                se_cfg = dict(
                    channels=mid_channels,
                    ratio=4,
                    act_cfg=(dict(type='ReLU'),
                             dict(type='HSigmoid', bias=1.0, divisor=2.0)))
            else:
                se_cfg = None

            layer = InvertedResidual(
                in_channels=self.in_channels,
                out_channels=out_channels,
                mid_channels=mid_channels,
                kernel_size=kernel_size,
                stride=stride,
                se_cfg=se_cfg,
                with_expand_conv=True,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=dict(type=act),
                with_cp=self.with_cp)
            self.in_channels = out_channels
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, layer)
            layers.append(layer_name)
        return layers

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.conv1(x)

        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices or \
                    i - len(self.layers) in self.out_indices:
                outs.append(x)

        if len(outs) == 1:
            return outs[0]
        return tuple(outs)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self, f'layer{i}')
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
