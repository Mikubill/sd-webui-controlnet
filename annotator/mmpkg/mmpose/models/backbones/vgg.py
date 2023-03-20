# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from annotator.mmpkg.mmcv.cnn import ConvModule, constant_init, kaiming_init, normal_init
from annotator.mmpkg.mmcv.utils.parrots_wrapper import _BatchNorm

from ..builder import BACKBONES
from .base_backbone import BaseBackbone


def make_vgg_layer(in_channels,
                   out_channels,
                   num_blocks,
                   conv_cfg=None,
                   norm_cfg=None,
                   act_cfg=dict(type='ReLU'),
                   dilation=1,
                   with_norm=False,
                   ceil_mode=False):
    layers = []
    for _ in range(num_blocks):
        layer = ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            dilation=dilation,
            padding=dilation,
            bias=True,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        layers.append(layer)
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil_mode))

    return layers


@BACKBONES.register_module()
class VGG(BaseBackbone):
    """VGG backbone.

    Args:
        depth (int): Depth of vgg, from {11, 13, 16, 19}.
        with_norm (bool): Use BatchNorm or not.
        num_classes (int): number of classes for classification.
        num_stages (int): VGG stages, normally 5.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages. If only one
            stage is specified, a single tensor (feature map) is returned,
            otherwise multiple stages are specified, a tuple of tensors will
            be returned. When it is None, the default behavior depends on
            whether num_classes is specified. If num_classes <= 0, the default
            value is (4, ), outputting the last feature map before classifier.
            If num_classes > 0, the default value is (5, ), outputting the
            classification score. Default: None.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        ceil_mode (bool): Whether to use ceil_mode of MaxPool. Default: False.
        with_last_pool (bool): Whether to keep the last pooling before
            classifier. Default: True.
    """

    # Parameters to build layers. Each element specifies the number of conv in
    # each stage. For example, VGG11 contains 11 layers with learnable
    # parameters. 11 is computed as 11 = (1 + 1 + 2 + 2 + 2) + 3,
    # where 3 indicates the last three fully-connected layers.
    arch_settings = {
        11: (1, 1, 2, 2, 2),
        13: (2, 2, 2, 2, 2),
        16: (2, 2, 3, 3, 3),
        19: (2, 2, 4, 4, 4)
    }

    def __init__(self,
                 depth,
                 num_classes=-1,
                 num_stages=5,
                 dilations=(1, 1, 1, 1, 1),
                 out_indices=None,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 norm_eval=False,
                 ceil_mode=False,
                 with_last_pool=True):
        super().__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for vgg')
        assert num_stages >= 1 and num_stages <= 5
        stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        assert len(dilations) == num_stages

        self.num_classes = num_classes
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        with_norm = norm_cfg is not None

        if out_indices is None:
            out_indices = (5, ) if num_classes > 0 else (4, )
        assert max(out_indices) <= num_stages
        self.out_indices = out_indices

        self.in_channels = 3
        start_idx = 0
        vgg_layers = []
        self.range_sub_modules = []
        for i, num_blocks in enumerate(self.stage_blocks):
            num_modules = num_blocks + 1
            end_idx = start_idx + num_modules
            dilation = dilations[i]
            out_channels = 64 * 2**i if i < 4 else 512
            vgg_layer = make_vgg_layer(
                self.in_channels,
                out_channels,
                num_blocks,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                dilation=dilation,
                with_norm=with_norm,
                ceil_mode=ceil_mode)
            vgg_layers.extend(vgg_layer)
            self.in_channels = out_channels
            self.range_sub_modules.append([start_idx, end_idx])
            start_idx = end_idx
        if not with_last_pool:
            vgg_layers.pop(-1)
            self.range_sub_modules[-1][1] -= 1
        self.module_name = 'features'
        self.add_module(self.module_name, nn.Sequential(*vgg_layers))

        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )

    def init_weights(self, pretrained=None):
        super().init_weights(pretrained)
        if pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)

    def forward(self, x):
        outs = []
        vgg_layers = getattr(self, self.module_name)
        for i in range(len(self.stage_blocks)):
            for j in range(*self.range_sub_modules[i]):
                vgg_layer = vgg_layers[j]
                x = vgg_layer(x)
            if i in self.out_indices:
                outs.append(x)
        if self.num_classes > 0:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            outs.append(x)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def _freeze_stages(self):
        vgg_layers = getattr(self, self.module_name)
        for i in range(self.frozen_stages):
            for j in range(*self.range_sub_modules[i]):
                m = vgg_layers[j]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
