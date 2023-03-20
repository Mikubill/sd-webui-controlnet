# Copyright (c) OpenMMLab. All rights reserved.
import copy
import logging

import torch.nn as nn
from annotator.mmpkg.mmcv.cnn import ConvModule
from torch.nn.modules.batchnorm import _BatchNorm

from ..builder import BACKBONES
from .base_backbone import BaseBackbone
from .utils import InvertedResidual, load_checkpoint


@BACKBONES.register_module()
class ViPNAS_MobileNetV3(BaseBackbone):
    """ViPNAS_MobileNetV3 backbone.

    "ViPNAS: Efficient Video Pose Estimation via Neural Architecture Search"
    More details can be found in the `paper
    <https://arxiv.org/abs/2105.10154>`__ .

    Args:
        wid (list(int)): Searched width config for each stage.
        expan (list(int)): Searched expansion ratio config for each stage.
        dep (list(int)): Searched depth config for each stage.
        ks (list(int)): Searched kernel size config for each stage.
        group (list(int)): Searched group number config for each stage.
        att (list(bool)): Searched attention config for each stage.
        stride (list(int)): Stride config for each stage.
        act (list(dict)): Activation config for each stage.
        conv_cfg (dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        frozen_stages (int): Stages to be frozen (all param fixed).
            Default: -1, which means not freezing any parameters.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed.
            Default: False.
    """

    def __init__(self,
                 wid=[16, 16, 24, 40, 80, 112, 160],
                 expan=[None, 1, 5, 4, 5, 5, 6],
                 dep=[None, 1, 4, 4, 4, 4, 4],
                 ks=[3, 3, 7, 7, 5, 7, 5],
                 group=[None, 8, 120, 20, 100, 280, 240],
                 att=[None, True, True, False, True, True, True],
                 stride=[2, 1, 2, 2, 2, 1, 2],
                 act=[
                     'HSwish', 'ReLU', 'ReLU', 'ReLU', 'HSwish', 'HSwish',
                     'HSwish'
                 ],
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 frozen_stages=-1,
                 norm_eval=False,
                 with_cp=False):
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__()
        self.wid = wid
        self.expan = expan
        self.dep = dep
        self.ks = ks
        self.group = group
        self.att = att
        self.stride = stride
        self.act = act
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.with_cp = with_cp

        self.conv1 = ConvModule(
            in_channels=3,
            out_channels=self.wid[0],
            kernel_size=self.ks[0],
            stride=self.stride[0],
            padding=self.ks[0] // 2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=dict(type=self.act[0]))

        self.layers = self._make_layer()

    def _make_layer(self):
        layers = []
        layer_index = 0
        for i, dep in enumerate(self.dep[1:]):
            mid_channels = self.wid[i + 1] * self.expan[i + 1]

            if self.att[i + 1]:
                se_cfg = dict(
                    channels=mid_channels,
                    ratio=4,
                    act_cfg=(dict(type='ReLU'),
                             dict(type='HSigmoid', bias=1.0, divisor=2.0)))
            else:
                se_cfg = None

            if self.expan[i + 1] == 1:
                with_expand_conv = False
            else:
                with_expand_conv = True

            for j in range(dep):
                if j == 0:
                    stride = self.stride[i + 1]
                    in_channels = self.wid[i]
                else:
                    stride = 1
                    in_channels = self.wid[i + 1]

                layer = InvertedResidual(
                    in_channels=in_channels,
                    out_channels=self.wid[i + 1],
                    mid_channels=mid_channels,
                    kernel_size=self.ks[i + 1],
                    groups=self.group[i + 1],
                    stride=stride,
                    se_cfg=se_cfg,
                    with_expand_conv=with_expand_conv,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=dict(type=self.act[i + 1]),
                    with_cp=self.with_cp)
                layer_index += 1
                layer_name = f'layer{layer_index}'
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
                    nn.init.normal_(m.weight, std=0.001)
                    for name, _ in m.named_parameters():
                        if name in ['bias']:
                            nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.conv1(x)

        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)

        return x

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
