# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
import torch.nn as nn
from annotator.mmpkg.mmcv.cnn import ConvModule, constant_init, normal_init
from torch.nn.modules.batchnorm import _BatchNorm

from annotator.mmpkg.mmpose.utils import get_root_logger
from ..builder import BACKBONES
from .base_backbone import BaseBackbone
from .utils import load_checkpoint


class CpmBlock(nn.Module):
    """CpmBlock for Convolutional Pose Machine.

    Args:
        in_channels (int): Input channels of this block.
        channels (list): Output channels of each conv module.
        kernels (list): Kernel sizes of each conv module.
    """

    def __init__(self,
                 in_channels,
                 channels=(128, 128, 128),
                 kernels=(11, 11, 11),
                 norm_cfg=None):
        super().__init__()

        assert len(channels) == len(kernels)
        layers = []
        for i in range(len(channels)):
            if i == 0:
                input_channels = in_channels
            else:
                input_channels = channels[i - 1]
            layers.append(
                ConvModule(
                    input_channels,
                    channels[i],
                    kernels[i],
                    padding=(kernels[i] - 1) // 2,
                    norm_cfg=norm_cfg))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """Model forward function."""
        out = self.model(x)
        return out


@BACKBONES.register_module()
class CPM(BaseBackbone):
    """CPM backbone.

    Convolutional Pose Machines.
    More details can be found in the `paper
    <https://arxiv.org/abs/1602.00134>`__ .

    Args:
        in_channels (int): The input channels of the CPM.
        out_channels (int): The output channels of the CPM.
        feat_channels (int): Feature channel of each CPM stage.
        middle_channels (int): Feature channel of conv after the middle stage.
        num_stages (int): Number of stages.
        norm_cfg (dict): Dictionary to construct and config norm layer.

    Example:
        >>> from annotator.mmpkg.mmpose.models import CPM
        >>> import torch
        >>> self = CPM(3, 17)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 368, 368)
        >>> level_outputs = self.forward(inputs)
        >>> for level_output in level_outputs:
        ...     print(tuple(level_output.shape))
        (1, 17, 46, 46)
        (1, 17, 46, 46)
        (1, 17, 46, 46)
        (1, 17, 46, 46)
        (1, 17, 46, 46)
        (1, 17, 46, 46)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 feat_channels=128,
                 middle_channels=32,
                 num_stages=6,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__()

        assert in_channels == 3

        self.num_stages = num_stages
        assert self.num_stages >= 1

        self.stem = nn.Sequential(
            ConvModule(in_channels, 128, 9, padding=4, norm_cfg=norm_cfg),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvModule(128, 128, 9, padding=4, norm_cfg=norm_cfg),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvModule(128, 128, 9, padding=4, norm_cfg=norm_cfg),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvModule(128, 32, 5, padding=2, norm_cfg=norm_cfg),
            ConvModule(32, 512, 9, padding=4, norm_cfg=norm_cfg),
            ConvModule(512, 512, 1, padding=0, norm_cfg=norm_cfg),
            ConvModule(512, out_channels, 1, padding=0, act_cfg=None))

        self.middle = nn.Sequential(
            ConvModule(in_channels, 128, 9, padding=4, norm_cfg=norm_cfg),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvModule(128, 128, 9, padding=4, norm_cfg=norm_cfg),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvModule(128, 128, 9, padding=4, norm_cfg=norm_cfg),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.cpm_stages = nn.ModuleList([
            CpmBlock(
                middle_channels + out_channels,
                channels=[feat_channels, feat_channels, feat_channels],
                kernels=[11, 11, 11],
                norm_cfg=norm_cfg) for _ in range(num_stages - 1)
        ])

        self.middle_conv = nn.ModuleList([
            nn.Sequential(
                ConvModule(
                    128, middle_channels, 5, padding=2, norm_cfg=norm_cfg))
            for _ in range(num_stages - 1)
        ])

        self.out_convs = nn.ModuleList([
            nn.Sequential(
                ConvModule(
                    feat_channels,
                    feat_channels,
                    1,
                    padding=0,
                    norm_cfg=norm_cfg),
                ConvModule(feat_channels, out_channels, 1, act_cfg=None))
            for _ in range(num_stages - 1)
        ])

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Model forward function."""
        stage1_out = self.stem(x)
        middle_out = self.middle(x)
        out_feats = []

        out_feats.append(stage1_out)

        for ind in range(self.num_stages - 1):
            single_stage = self.cpm_stages[ind]
            out_conv = self.out_convs[ind]

            inp_feat = torch.cat(
                [out_feats[-1], self.middle_conv[ind](middle_out)], 1)
            cpm_feat = single_stage(inp_feat)
            out_feat = out_conv(cpm_feat)
            out_feats.append(out_feat)

        return out_feats
