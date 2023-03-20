# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch.nn as nn
from annotator.mmpkg.mmcv.cnn import ConvModule, MaxPool2d, constant_init, normal_init
from torch.nn.modules.batchnorm import _BatchNorm

from annotator.mmpkg.mmpose.utils import get_root_logger
from ..builder import BACKBONES
from .base_backbone import BaseBackbone
from .utils import load_checkpoint


class HourglassAEModule(nn.Module):
    """Modified Hourglass Module for HourglassNet_AE backbone.

    Generate module recursively and use BasicBlock as the base unit.

    Args:
        depth (int): Depth of current HourglassModule.
        stage_channels (list[int]): Feature channels of sub-modules in current
            and follow-up HourglassModule.
        norm_cfg (dict): Dictionary to construct and config norm layer.
    """

    def __init__(self,
                 depth,
                 stage_channels,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__()

        self.depth = depth

        cur_channel = stage_channels[0]
        next_channel = stage_channels[1]

        self.up1 = ConvModule(
            cur_channel, cur_channel, 3, padding=1, norm_cfg=norm_cfg)

        self.pool1 = MaxPool2d(2, 2)

        self.low1 = ConvModule(
            cur_channel, next_channel, 3, padding=1, norm_cfg=norm_cfg)

        if self.depth > 1:
            self.low2 = HourglassAEModule(depth - 1, stage_channels[1:])
        else:
            self.low2 = ConvModule(
                next_channel, next_channel, 3, padding=1, norm_cfg=norm_cfg)

        self.low3 = ConvModule(
            next_channel, cur_channel, 3, padding=1, norm_cfg=norm_cfg)

        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, x):
        """Model forward function."""
        up1 = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        return up1 + up2


@BACKBONES.register_module()
class HourglassAENet(BaseBackbone):
    """Hourglass-AE Network proposed by Newell et al.

    Associative Embedding: End-to-End Learning for Joint
    Detection and Grouping.

    More details can be found in the `paper
    <https://arxiv.org/abs/1611.05424>`__ .

    Args:
        downsample_times (int): Downsample times in a HourglassModule.
        num_stacks (int): Number of HourglassModule modules stacked,
            1 for Hourglass-52, 2 for Hourglass-104.
        stage_channels (list[int]): Feature channel of each sub-module in a
            HourglassModule.
        stage_blocks (list[int]): Number of sub-modules stacked in a
            HourglassModule.
        feat_channels (int): Feature channel of conv after a HourglassModule.
        norm_cfg (dict): Dictionary to construct and config norm layer.

    Example:
        >>> from annotator.mmpkg.mmpose.models import HourglassAENet
        >>> import torch
        >>> self = HourglassAENet()
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 512, 512)
        >>> level_outputs = self.forward(inputs)
        >>> for level_output in level_outputs:
        ...     print(tuple(level_output.shape))
        (1, 34, 128, 128)
    """

    def __init__(self,
                 downsample_times=4,
                 num_stacks=1,
                 out_channels=34,
                 stage_channels=(256, 384, 512, 640, 768),
                 feat_channels=256,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__()

        self.num_stacks = num_stacks
        assert self.num_stacks >= 1
        assert len(stage_channels) > downsample_times

        cur_channels = stage_channels[0]

        self.stem = nn.Sequential(
            ConvModule(3, 64, 7, padding=3, stride=2, norm_cfg=norm_cfg),
            ConvModule(64, 128, 3, padding=1, norm_cfg=norm_cfg),
            MaxPool2d(2, 2),
            ConvModule(128, 128, 3, padding=1, norm_cfg=norm_cfg),
            ConvModule(128, feat_channels, 3, padding=1, norm_cfg=norm_cfg),
        )

        self.hourglass_modules = nn.ModuleList([
            nn.Sequential(
                HourglassAEModule(
                    downsample_times, stage_channels, norm_cfg=norm_cfg),
                ConvModule(
                    feat_channels,
                    feat_channels,
                    3,
                    padding=1,
                    norm_cfg=norm_cfg),
                ConvModule(
                    feat_channels,
                    feat_channels,
                    3,
                    padding=1,
                    norm_cfg=norm_cfg)) for _ in range(num_stacks)
        ])

        self.out_convs = nn.ModuleList([
            ConvModule(
                cur_channels,
                out_channels,
                1,
                padding=0,
                norm_cfg=None,
                act_cfg=None) for _ in range(num_stacks)
        ])

        self.remap_out_convs = nn.ModuleList([
            ConvModule(
                out_channels,
                feat_channels,
                1,
                norm_cfg=norm_cfg,
                act_cfg=None) for _ in range(num_stacks - 1)
        ])

        self.remap_feature_convs = nn.ModuleList([
            ConvModule(
                feat_channels,
                feat_channels,
                1,
                norm_cfg=norm_cfg,
                act_cfg=None) for _ in range(num_stacks - 1)
        ])

        self.relu = nn.ReLU(inplace=True)

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
        inter_feat = self.stem(x)
        out_feats = []

        for ind in range(self.num_stacks):
            single_hourglass = self.hourglass_modules[ind]
            out_conv = self.out_convs[ind]

            hourglass_feat = single_hourglass(inter_feat)
            out_feat = out_conv(hourglass_feat)
            out_feats.append(out_feat)

            if ind < self.num_stacks - 1:
                inter_feat = inter_feat + self.remap_out_convs[ind](
                    out_feat) + self.remap_feature_convs[ind](
                        hourglass_feat)

        return out_feats
