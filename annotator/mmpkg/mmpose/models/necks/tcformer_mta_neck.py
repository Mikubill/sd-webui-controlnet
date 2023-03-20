# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch.nn as nn
import torch.nn.functional as F
from annotator.mmpkg.mmcv.cnn import ConvModule, constant_init, normal_init, trunc_normal_init
from annotator.mmpkg.mmcv.runner import BaseModule

from ..builder import NECKS
from ..utils import TCFormerDynamicBlock, token2map, token_interp


@NECKS.register_module()
class MTA(BaseModule):
    """Multi-stage Token feature Aggregation (MTA) module in TCFormer.

    Args:
        in_channels (list[int]): Number of input channels per stage.
            Default: [64, 128, 256, 512].
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales. Default: 4.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed
            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer in ConvModule.
        num_heads (Sequence[int]): The attention heads of each transformer
            block. Default: [2, 2, 2, 2].
        mlp_ratios (Sequence[int]): The ratio of the mlp hidden dim to the
            embedding dim of each transformer block.
        sr_ratios (Sequence[int]): The spatial reduction rate of each
            transformer block. Default: [8, 4, 2, 1].
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0.
        drop_path_rate (float): stochastic depth rate. Default 0.
        transformer_norm_cfg (dict): Config dict for normalization layer
            in transformer blocks. Default: dict(type='LN').
        use_sr_conv (bool): If True, use a conv layer for spatial reduction.
            If False, use a pooling process for spatial reduction. Defaults:
            False.
    """

    def __init__(
            self,
            in_channels=[64, 128, 256, 512],
            out_channels=128,
            num_outs=4,
            start_level=0,
            end_level=-1,
            add_extra_convs=False,
            relu_before_extra_convs=False,
            no_norm_on_lateral=False,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None,
            num_heads=[2, 2, 2, 2],
            mlp_ratios=[4, 4, 4, 4],
            sr_ratios=[8, 4, 2, 1],
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            transformer_norm_cfg=dict(type='LN'),
            use_sr_conv=False,
    ):
        super().__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.act_cfg = act_cfg
        self.mlp_ratios = mlp_ratios

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level

        self.lateral_convs = nn.ModuleList()
        self.merge_blocks = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)

        for i in range(self.start_level, self.backbone_end_level - 1):
            merge_block = TCFormerDynamicBlock(
                dim=out_channels,
                num_heads=num_heads[i],
                mlp_ratio=mlp_ratios[i],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate,
                norm_cfg=transformer_norm_cfg,
                sr_ratio=sr_ratios[i],
                use_sr_conv=use_sr_conv)
            self.merge_blocks.append(merge_block)

        # add extra conv layers (e.g., RetinaNet)
        self.relu_before_extra_convs = relu_before_extra_convs

        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_output'
            assert add_extra_convs in ('on_input', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.extra_convs = nn.ModuleList()
        extra_levels = num_outs - (self.end_level + 1 - self.start_level)
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.end_level]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.extra_convs.append(extra_fpn_conv)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0.)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, 1.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                normal_init(m, 0, math.sqrt(2.0 / fan_out))

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build lateral tokens
        input_dicts = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            tmp = inputs[i + self.start_level].copy()
            tmp['x'] = lateral_conv(tmp['x'].unsqueeze(2).permute(
                0, 3, 1, 2)).permute(0, 2, 3, 1).squeeze(2)
            input_dicts.append(tmp)

        # merge from high level to low level
        for i in range(len(input_dicts) - 2, -1, -1):
            input_dicts[i]['x'] = input_dicts[i]['x'] + token_interp(
                input_dicts[i], input_dicts[i + 1])
            input_dicts[i] = self.merge_blocks[i](input_dicts[i])

        # transform to feature map
        outs = [token2map(token_dict) for token_dict in input_dicts]

        # part 2: add extra levels
        used_backbone_levels = len(outs)
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps
            else:
                if self.add_extra_convs == 'on_input':
                    tmp = inputs[self.backbone_end_level - 1]
                    extra_source = token2map(tmp)
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError

                outs.append(self.extra_convs[0](extra_source))
                for i in range(1, self.num_outs - used_backbone_levels):
                    if self.relu_before_extra_convs:
                        outs.append(self.extra_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.extra_convs[i](outs[-1]))
        return outs
