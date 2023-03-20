# Copyright (c) OpenMMLab. All rights reserved.
import copy as cp
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
from annotator.mmpkg.mmcv.cnn import (ConvModule, MaxPool2d, constant_init, kaiming_init,
                      normal_init)
from annotator.mmpkg.mmcv.runner.checkpoint import load_state_dict

from annotator.mmpkg.mmpose.utils import get_root_logger
from ..builder import BACKBONES
from .base_backbone import BaseBackbone
from .resnet import Bottleneck as _Bottleneck
from .utils.utils import get_state_dict


class Bottleneck(_Bottleneck):
    expansion = 4
    """Bottleneck block for MSPN.

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        stride (int): stride of the block. Default: 1
        downsample (nn.Module): downsample operation on identity branch.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels, out_channels * 4, **kwargs)


class DownsampleModule(nn.Module):
    """Downsample module for MSPN.

    Args:
        block (nn.Module): Downsample block.
        num_blocks (list): Number of blocks in each downsample unit.
        num_units (int): Numbers of downsample units. Default: 4
        has_skip (bool): Have skip connections from prior upsample
            module or not. Default:False
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        in_channels (int): Number of channels of the input feature to
            downsample module. Default: 64
    """

    def __init__(self,
                 block,
                 num_blocks,
                 num_units=4,
                 has_skip=False,
                 norm_cfg=dict(type='BN'),
                 in_channels=64):
        # Protect mutable default arguments
        norm_cfg = cp.deepcopy(norm_cfg)
        super().__init__()
        self.has_skip = has_skip
        self.in_channels = in_channels
        assert len(num_blocks) == num_units
        self.num_blocks = num_blocks
        self.num_units = num_units
        self.norm_cfg = norm_cfg
        self.layer1 = self._make_layer(block, in_channels, num_blocks[0])
        for i in range(1, num_units):
            module_name = f'layer{i + 1}'
            self.add_module(
                module_name,
                self._make_layer(
                    block, in_channels * pow(2, i), num_blocks[i], stride=2))

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = ConvModule(
                self.in_channels,
                out_channels * block.expansion,
                kernel_size=1,
                stride=stride,
                padding=0,
                norm_cfg=self.norm_cfg,
                act_cfg=None,
                inplace=True)

        units = list()
        units.append(
            block(
                self.in_channels,
                out_channels,
                stride=stride,
                downsample=downsample,
                norm_cfg=self.norm_cfg))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            units.append(block(self.in_channels, out_channels))

        return nn.Sequential(*units)

    def forward(self, x, skip1, skip2):
        out = list()
        for i in range(self.num_units):
            module_name = f'layer{i + 1}'
            module_i = getattr(self, module_name)
            x = module_i(x)
            if self.has_skip:
                x = x + skip1[i] + skip2[i]
            out.append(x)
        out.reverse()

        return tuple(out)


class UpsampleUnit(nn.Module):
    """Upsample unit for upsample module.

    Args:
        ind (int): Indicates whether to interpolate (>0) and whether to
           generate feature map for the next hourglass-like module.
        num_units (int): Number of units that form a upsample module. Along
            with ind and gen_cross_conv, nm_units is used to decide whether
            to generate feature map for the next hourglass-like module.
        in_channels (int): Channel number of the skip-in feature maps from
            the corresponding downsample unit.
        unit_channels (int): Channel number in this unit. Default:256.
        gen_skip: (bool): Whether or not to generate skips for the posterior
            downsample module. Default:False
        gen_cross_conv (bool): Whether to generate feature map for the next
            hourglass-like module. Default:False
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        out_channels (int): Number of channels of feature output by upsample
            module. Must equal to in_channels of downsample module. Default:64
    """

    def __init__(self,
                 ind,
                 num_units,
                 in_channels,
                 unit_channels=256,
                 gen_skip=False,
                 gen_cross_conv=False,
                 norm_cfg=dict(type='BN'),
                 out_channels=64):
        # Protect mutable default arguments
        norm_cfg = cp.deepcopy(norm_cfg)
        super().__init__()
        self.num_units = num_units
        self.norm_cfg = norm_cfg
        self.in_skip = ConvModule(
            in_channels,
            unit_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_cfg=self.norm_cfg,
            act_cfg=None,
            inplace=True)
        self.relu = nn.ReLU(inplace=True)

        self.ind = ind
        if self.ind > 0:
            self.up_conv = ConvModule(
                unit_channels,
                unit_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                norm_cfg=self.norm_cfg,
                act_cfg=None,
                inplace=True)

        self.gen_skip = gen_skip
        if self.gen_skip:
            self.out_skip1 = ConvModule(
                in_channels,
                in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                norm_cfg=self.norm_cfg,
                inplace=True)

            self.out_skip2 = ConvModule(
                unit_channels,
                in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                norm_cfg=self.norm_cfg,
                inplace=True)

        self.gen_cross_conv = gen_cross_conv
        if self.ind == num_units - 1 and self.gen_cross_conv:
            self.cross_conv = ConvModule(
                unit_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                norm_cfg=self.norm_cfg,
                inplace=True)

    def forward(self, x, up_x):
        out = self.in_skip(x)

        if self.ind > 0:
            up_x = F.interpolate(
                up_x,
                size=(x.size(2), x.size(3)),
                mode='bilinear',
                align_corners=True)
            up_x = self.up_conv(up_x)
            out = out + up_x
        out = self.relu(out)

        skip1 = None
        skip2 = None
        if self.gen_skip:
            skip1 = self.out_skip1(x)
            skip2 = self.out_skip2(out)

        cross_conv = None
        if self.ind == self.num_units - 1 and self.gen_cross_conv:
            cross_conv = self.cross_conv(out)

        return out, skip1, skip2, cross_conv


class UpsampleModule(nn.Module):
    """Upsample module for MSPN.

    Args:
        unit_channels (int): Channel number in the upsample units.
            Default:256.
        num_units (int): Numbers of upsample units. Default: 4
        gen_skip (bool): Whether to generate skip for posterior downsample
            module or not. Default:False
        gen_cross_conv (bool): Whether to generate feature map for the next
            hourglass-like module. Default:False
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        out_channels (int): Number of channels of feature output by upsample
            module. Must equal to in_channels of downsample module. Default:64
    """

    def __init__(self,
                 unit_channels=256,
                 num_units=4,
                 gen_skip=False,
                 gen_cross_conv=False,
                 norm_cfg=dict(type='BN'),
                 out_channels=64):
        # Protect mutable default arguments
        norm_cfg = cp.deepcopy(norm_cfg)
        super().__init__()
        self.in_channels = list()
        for i in range(num_units):
            self.in_channels.append(Bottleneck.expansion * out_channels *
                                    pow(2, i))
        self.in_channels.reverse()
        self.num_units = num_units
        self.gen_skip = gen_skip
        self.gen_cross_conv = gen_cross_conv
        self.norm_cfg = norm_cfg
        for i in range(num_units):
            module_name = f'up{i + 1}'
            self.add_module(
                module_name,
                UpsampleUnit(
                    i,
                    self.num_units,
                    self.in_channels[i],
                    unit_channels,
                    self.gen_skip,
                    self.gen_cross_conv,
                    norm_cfg=self.norm_cfg,
                    out_channels=64))

    def forward(self, x):
        out = list()
        skip1 = list()
        skip2 = list()
        cross_conv = None
        for i in range(self.num_units):
            module_i = getattr(self, f'up{i + 1}')
            if i == 0:
                outi, skip1_i, skip2_i, _ = module_i(x[i], None)
            elif i == self.num_units - 1:
                outi, skip1_i, skip2_i, cross_conv = module_i(x[i], out[i - 1])
            else:
                outi, skip1_i, skip2_i, _ = module_i(x[i], out[i - 1])
            out.append(outi)
            skip1.append(skip1_i)
            skip2.append(skip2_i)
        skip1.reverse()
        skip2.reverse()

        return out, skip1, skip2, cross_conv


class SingleStageNetwork(nn.Module):
    """Single_stage Network.

    Args:
        unit_channels (int): Channel number in the upsample units. Default:256.
        num_units (int): Numbers of downsample/upsample units. Default: 4
        gen_skip (bool): Whether to generate skip for posterior downsample
            module or not. Default:False
        gen_cross_conv (bool): Whether to generate feature map for the next
            hourglass-like module. Default:False
        has_skip (bool): Have skip connections from prior upsample
            module or not. Default:False
        num_blocks (list): Number of blocks in each downsample unit.
            Default: [2, 2, 2, 2] Note: Make sure num_units==len(num_blocks)
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        in_channels (int): Number of channels of the feature from ResNetTop.
            Default: 64.
    """

    def __init__(self,
                 has_skip=False,
                 gen_skip=False,
                 gen_cross_conv=False,
                 unit_channels=256,
                 num_units=4,
                 num_blocks=[2, 2, 2, 2],
                 norm_cfg=dict(type='BN'),
                 in_channels=64):
        # Protect mutable default arguments
        norm_cfg = cp.deepcopy(norm_cfg)
        num_blocks = cp.deepcopy(num_blocks)
        super().__init__()
        assert len(num_blocks) == num_units
        self.has_skip = has_skip
        self.gen_skip = gen_skip
        self.gen_cross_conv = gen_cross_conv
        self.num_units = num_units
        self.unit_channels = unit_channels
        self.num_blocks = num_blocks
        self.norm_cfg = norm_cfg

        self.downsample = DownsampleModule(Bottleneck, num_blocks, num_units,
                                           has_skip, norm_cfg, in_channels)
        self.upsample = UpsampleModule(unit_channels, num_units, gen_skip,
                                       gen_cross_conv, norm_cfg, in_channels)

    def forward(self, x, skip1, skip2):
        mid = self.downsample(x, skip1, skip2)
        out, skip1, skip2, cross_conv = self.upsample(mid)

        return out, skip1, skip2, cross_conv


class ResNetTop(nn.Module):
    """ResNet top for MSPN.

    Args:
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        channels (int): Number of channels of the feature output by ResNetTop.
    """

    def __init__(self, norm_cfg=dict(type='BN'), channels=64):
        # Protect mutable default arguments
        norm_cfg = cp.deepcopy(norm_cfg)
        super().__init__()
        self.top = nn.Sequential(
            ConvModule(
                3,
                channels,
                kernel_size=7,
                stride=2,
                padding=3,
                norm_cfg=norm_cfg,
                inplace=True), MaxPool2d(kernel_size=3, stride=2, padding=1))

    def forward(self, img):
        return self.top(img)


@BACKBONES.register_module()
class MSPN(BaseBackbone):
    """MSPN backbone. Paper ref: Li et al. "Rethinking on Multi-Stage Networks
    for Human Pose Estimation" (CVPR 2020).

    Args:
        unit_channels (int): Number of Channels in an upsample unit.
            Default: 256
        num_stages (int): Number of stages in a multi-stage MSPN. Default: 4
        num_units (int): Number of downsample/upsample units in a single-stage
            network. Default: 4
            Note: Make sure num_units == len(self.num_blocks)
        num_blocks (list): Number of bottlenecks in each
            downsample unit. Default: [2, 2, 2, 2]
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        res_top_channels (int): Number of channels of feature from ResNetTop.
            Default: 64.

    Example:
        >>> from annotator.mmpkg.mmpose.models import MSPN
        >>> import torch
        >>> self = MSPN(num_stages=2,num_units=2,num_blocks=[2,2])
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 511, 511)
        >>> level_outputs = self.forward(inputs)
        >>> for level_output in level_outputs:
        ...     for feature in level_output:
        ...         print(tuple(feature.shape))
        ...
        (1, 256, 64, 64)
        (1, 256, 128, 128)
        (1, 256, 64, 64)
        (1, 256, 128, 128)
    """

    def __init__(self,
                 unit_channels=256,
                 num_stages=4,
                 num_units=4,
                 num_blocks=[2, 2, 2, 2],
                 norm_cfg=dict(type='BN'),
                 res_top_channels=64):
        # Protect mutable default arguments
        norm_cfg = cp.deepcopy(norm_cfg)
        num_blocks = cp.deepcopy(num_blocks)
        super().__init__()
        self.unit_channels = unit_channels
        self.num_stages = num_stages
        self.num_units = num_units
        self.num_blocks = num_blocks
        self.norm_cfg = norm_cfg

        assert self.num_stages > 0
        assert self.num_units > 1
        assert self.num_units == len(self.num_blocks)
        self.top = ResNetTop(norm_cfg=norm_cfg)
        self.multi_stage_mspn = nn.ModuleList([])
        for i in range(self.num_stages):
            if i == 0:
                has_skip = False
            else:
                has_skip = True
            if i != self.num_stages - 1:
                gen_skip = True
                gen_cross_conv = True
            else:
                gen_skip = False
                gen_cross_conv = False
            self.multi_stage_mspn.append(
                SingleStageNetwork(has_skip, gen_skip, gen_cross_conv,
                                   unit_channels, num_units, num_blocks,
                                   norm_cfg, res_top_channels))

    def forward(self, x):
        """Model forward function."""
        out_feats = []
        skip1 = None
        skip2 = None
        x = self.top(x)
        for i in range(self.num_stages):
            out, skip1, skip2, x = self.multi_stage_mspn[i](x, skip1, skip2)
            out_feats.append(out)

        return out_feats

    def init_weights(self, pretrained=None):
        """Initialize model weights."""
        if isinstance(pretrained, str):
            logger = get_root_logger()
            state_dict_tmp = get_state_dict(pretrained)
            state_dict = OrderedDict()
            state_dict['top'] = OrderedDict()
            state_dict['bottlenecks'] = OrderedDict()
            for k, v in state_dict_tmp.items():
                if k.startswith('layer'):
                    if 'downsample.0' in k:
                        state_dict['bottlenecks'][k.replace(
                            'downsample.0', 'downsample.conv')] = v
                    elif 'downsample.1' in k:
                        state_dict['bottlenecks'][k.replace(
                            'downsample.1', 'downsample.bn')] = v
                    else:
                        state_dict['bottlenecks'][k] = v
                elif k.startswith('conv1'):
                    state_dict['top'][k.replace('conv1', 'top.0.conv')] = v
                elif k.startswith('bn1'):
                    state_dict['top'][k.replace('bn1', 'top.0.bn')] = v

            load_state_dict(
                self.top, state_dict['top'], strict=False, logger=logger)
            for i in range(self.num_stages):
                load_state_dict(
                    self.multi_stage_mspn[i].downsample,
                    state_dict['bottlenecks'],
                    strict=False,
                    logger=logger)
        else:
            for m in self.multi_stage_mspn.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)

            for m in self.top.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
