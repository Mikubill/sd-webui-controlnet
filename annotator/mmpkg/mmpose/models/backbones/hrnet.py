# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch.nn as nn
from annotator.mmpkg.mmcv.cnn import (build_conv_layer, build_norm_layer, constant_init,
                      normal_init)
from torch.nn.modules.batchnorm import _BatchNorm

from annotator.mmpkg.mmpose.utils import get_root_logger
from ..builder import BACKBONES
from .resnet import BasicBlock, Bottleneck, get_expansion
from .utils import load_checkpoint


class HRModule(nn.Module):
    """High-Resolution Module for HRNet.

    In this module, every branch has 4 BasicBlocks/Bottlenecks. Fusion/Exchange
    is in this module.
    """

    def __init__(self,
                 num_branches,
                 blocks,
                 num_blocks,
                 in_channels,
                 num_channels,
                 multiscale_output=False,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 upsample_cfg=dict(mode='nearest', align_corners=None)):

        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__()
        self._check_branches(num_branches, num_blocks, in_channels,
                             num_channels)

        self.in_channels = in_channels
        self.num_branches = num_branches

        self.multiscale_output = multiscale_output
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.upsample_cfg = upsample_cfg
        self.with_cp = with_cp
        self.branches = self._make_branches(num_branches, blocks, num_blocks,
                                            num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    @staticmethod
    def _check_branches(num_branches, num_blocks, in_channels, num_channels):
        """Check input to avoid ValueError."""
        if num_branches != len(num_blocks):
            error_msg = f'NUM_BRANCHES({num_branches}) ' \
                f'!= NUM_BLOCKS({len(num_blocks)})'
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = f'NUM_BRANCHES({num_branches}) ' \
                f'!= NUM_CHANNELS({len(num_channels)})'
            raise ValueError(error_msg)

        if num_branches != len(in_channels):
            error_msg = f'NUM_BRANCHES({num_branches}) ' \
                f'!= NUM_INCHANNELS({len(in_channels)})'
            raise ValueError(error_msg)

    def _make_one_branch(self,
                         branch_index,
                         block,
                         num_blocks,
                         num_channels,
                         stride=1):
        """Make one branch."""
        downsample = None
        if stride != 1 or \
                self.in_channels[branch_index] != \
                num_channels[branch_index] * get_expansion(block):
            downsample = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    self.in_channels[branch_index],
                    num_channels[branch_index] * get_expansion(block),
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                build_norm_layer(
                    self.norm_cfg,
                    num_channels[branch_index] * get_expansion(block))[1])

        layers = []
        layers.append(
            block(
                self.in_channels[branch_index],
                num_channels[branch_index] * get_expansion(block),
                stride=stride,
                downsample=downsample,
                with_cp=self.with_cp,
                norm_cfg=self.norm_cfg,
                conv_cfg=self.conv_cfg))
        self.in_channels[branch_index] = \
            num_channels[branch_index] * get_expansion(block)
        for _ in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.in_channels[branch_index],
                    num_channels[branch_index] * get_expansion(block),
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        """Make branches."""
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        """Make fuse layer."""
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        in_channels = self.in_channels
        fuse_layers = []
        num_out_branches = num_branches if self.multiscale_output else 1

        for i in range(num_out_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                in_channels[j],
                                in_channels[i],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False),
                            build_norm_layer(self.norm_cfg, in_channels[i])[1],
                            nn.Upsample(
                                scale_factor=2**(j - i),
                                mode=self.upsample_cfg['mode'],
                                align_corners=self.
                                upsample_cfg['align_corners'])))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv_downsamples = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_downsamples.append(
                                nn.Sequential(
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[i],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        bias=False),
                                    build_norm_layer(self.norm_cfg,
                                                     in_channels[i])[1]))
                        else:
                            conv_downsamples.append(
                                nn.Sequential(
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        bias=False),
                                    build_norm_layer(self.norm_cfg,
                                                     in_channels[j])[1],
                                    nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv_downsamples))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        """Forward function."""
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = 0
            for j in range(self.num_branches):
                if i == j:
                    y += x[j]
                else:
                    y += self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


@BACKBONES.register_module()
class HRNet(nn.Module):
    """HRNet backbone.

    `High-Resolution Representations for Labeling Pixels and Regions
    <https://arxiv.org/abs/1904.04514>`__

    Args:
        extra (dict): detailed configuration for each stage of HRNet.
        in_channels (int): Number of input image channels. Default: 3.
        conv_cfg (dict): dictionary to construct and config conv layer.
        norm_cfg (dict): dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.

    Example:
        >>> from annotator.mmpkg.mmpose.models import HRNet
        >>> import torch
        >>> extra = dict(
        >>>     stage1=dict(
        >>>         num_modules=1,
        >>>         num_branches=1,
        >>>         block='BOTTLENECK',
        >>>         num_blocks=(4, ),
        >>>         num_channels=(64, )),
        >>>     stage2=dict(
        >>>         num_modules=1,
        >>>         num_branches=2,
        >>>         block='BASIC',
        >>>         num_blocks=(4, 4),
        >>>         num_channels=(32, 64)),
        >>>     stage3=dict(
        >>>         num_modules=4,
        >>>         num_branches=3,
        >>>         block='BASIC',
        >>>         num_blocks=(4, 4, 4),
        >>>         num_channels=(32, 64, 128)),
        >>>     stage4=dict(
        >>>         num_modules=3,
        >>>         num_branches=4,
        >>>         block='BASIC',
        >>>         num_blocks=(4, 4, 4, 4),
        >>>         num_channels=(32, 64, 128, 256)))
        >>> self = HRNet(extra, in_channels=1)
        >>> self.eval()
        >>> inputs = torch.rand(1, 1, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 32, 8, 8)
    """

    blocks_dict = {'BASIC': BasicBlock, 'BOTTLENECK': Bottleneck}

    def __init__(self,
                 extra,
                 in_channels=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 norm_eval=False,
                 with_cp=False,
                 zero_init_residual=False,
                 frozen_stages=-1):
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__()
        self.extra = extra
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual
        self.frozen_stages = frozen_stages

        # stem net
        self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, 64, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(self.norm_cfg, 64, postfix=2)

        self.conv1 = build_conv_layer(
            self.conv_cfg,
            in_channels,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)

        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            self.conv_cfg,
            64,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)

        self.add_module(self.norm2_name, norm2)
        self.relu = nn.ReLU(inplace=True)

        self.upsample_cfg = self.extra.get('upsample', {
            'mode': 'nearest',
            'align_corners': None
        })

        # stage 1
        self.stage1_cfg = self.extra['stage1']
        num_channels = self.stage1_cfg['num_channels'][0]
        block_type = self.stage1_cfg['block']
        num_blocks = self.stage1_cfg['num_blocks'][0]

        block = self.blocks_dict[block_type]
        stage1_out_channels = num_channels * get_expansion(block)
        self.layer1 = self._make_layer(block, 64, stage1_out_channels,
                                       num_blocks)

        # stage 2
        self.stage2_cfg = self.extra['stage2']
        num_channels = self.stage2_cfg['num_channels']
        block_type = self.stage2_cfg['block']

        block = self.blocks_dict[block_type]
        num_channels = [
            channel * get_expansion(block) for channel in num_channels
        ]
        self.transition1 = self._make_transition_layer([stage1_out_channels],
                                                       num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        # stage 3
        self.stage3_cfg = self.extra['stage3']
        num_channels = self.stage3_cfg['num_channels']
        block_type = self.stage3_cfg['block']

        block = self.blocks_dict[block_type]
        num_channels = [
            channel * get_expansion(block) for channel in num_channels
        ]
        self.transition2 = self._make_transition_layer(pre_stage_channels,
                                                       num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        # stage 4
        self.stage4_cfg = self.extra['stage4']
        num_channels = self.stage4_cfg['num_channels']
        block_type = self.stage4_cfg['block']

        block = self.blocks_dict[block_type]
        num_channels = [
            channel * get_expansion(block) for channel in num_channels
        ]
        self.transition3 = self._make_transition_layer(pre_stage_channels,
                                                       num_channels)

        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg,
            num_channels,
            multiscale_output=self.stage4_cfg.get('multiscale_output', False))

        self._freeze_stages()

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: the normalization layer named "norm2" """
        return getattr(self, self.norm2_name)

    def _make_transition_layer(self, num_channels_pre_layer,
                               num_channels_cur_layer):
        """Make transition layer."""
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False),
                            build_norm_layer(self.norm_cfg,
                                             num_channels_cur_layer[i])[1],
                            nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv_downsamples = []
                for j in range(i + 1 - num_branches_pre):
                    in_channels = num_channels_pre_layer[-1]
                    out_channels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else in_channels
                    conv_downsamples.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                in_channels,
                                out_channels,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                bias=False),
                            build_norm_layer(self.norm_cfg, out_channels)[1],
                            nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv_downsamples))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        """Make layer."""
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                build_norm_layer(self.norm_cfg, out_channels)[1])

        layers = []
        layers.append(
            block(
                in_channels,
                out_channels,
                stride=stride,
                downsample=downsample,
                with_cp=self.with_cp,
                norm_cfg=self.norm_cfg,
                conv_cfg=self.conv_cfg))
        for _ in range(1, blocks):
            layers.append(
                block(
                    out_channels,
                    out_channels,
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, in_channels, multiscale_output=True):
        """Make stage."""
        num_modules = layer_config['num_modules']
        num_branches = layer_config['num_branches']
        num_blocks = layer_config['num_blocks']
        num_channels = layer_config['num_channels']
        block = self.blocks_dict[layer_config['block']]

        hr_modules = []
        for i in range(num_modules):
            # multi_scale_output is only used for the last module
            if not multiscale_output and i == num_modules - 1:
                reset_multiscale_output = False
            else:
                reset_multiscale_output = True

            hr_modules.append(
                HRModule(
                    num_branches,
                    block,
                    num_blocks,
                    in_channels,
                    num_channels,
                    reset_multiscale_output,
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg,
                    upsample_cfg=self.upsample_cfg))

            in_channels = hr_modules[-1].in_channels

        return nn.Sequential(*hr_modules), in_channels

    def _freeze_stages(self):
        """Freeze parameters."""
        if self.frozen_stages >= 0:
            self.norm1.eval()
            self.norm2.eval()

            for m in [self.conv1, self.norm1, self.conv2, self.norm2]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            if i == 1:
                m = getattr(self, 'layer1')
            else:
                m = getattr(self, f'stage{i}')

            m.eval()
            for param in m.parameters():
                param.requires_grad = False

            if i < 4:
                m = getattr(self, f'transition{i}')
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

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

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['num_branches']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['num_branches']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['num_branches']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        return y_list

    def train(self, mode=True):
        """Convert the model into training mode."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
