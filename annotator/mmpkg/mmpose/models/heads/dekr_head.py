# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
import torch.nn as nn
from annotator.mmpkg.mmcv.cnn import (ConvModule, build_activation_layer, build_conv_layer,
                      build_norm_layer, constant_init, normal_init)

from annotator.mmpkg.mmpose.models.builder import build_loss
from ..backbones.resnet import BasicBlock
from ..builder import HEADS
from .deconv_head import DeconvHead

try:
    from annotator.mmpkg.mmcv.ops import DeformConv2d
    has_mmcv_full = True
except (ImportError, ModuleNotFoundError):
    has_mmcv_full = False


class AdaptiveActivationBlock(nn.Module):
    """Adaptive activation convolution block. "Bottom-up human pose estimation
    via disentangled keypoint regression", CVPR'2021.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        groups (int): Number of groups. Generally equal to the
            number of joints.
        norm_cfg (dict): Config for normalization layers.
        act_cfg (dict): Config for activation layers.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 groups=1,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):

        super(AdaptiveActivationBlock, self).__init__()

        assert in_channels % groups == 0 and out_channels % groups == 0
        self.groups = groups

        regular_matrix = torch.tensor([[-1, -1, -1, 0, 0, 0, 1, 1, 1],
                                       [-1, 0, 1, -1, 0, 1, -1, 0, 1],
                                       [1, 1, 1, 1, 1, 1, 1, 1, 1]])
        self.register_buffer('regular_matrix', regular_matrix.float())

        self.transform_matrix_conv = build_conv_layer(
            dict(type='Conv2d'),
            in_channels=in_channels,
            out_channels=6 * groups,
            kernel_size=3,
            padding=1,
            groups=groups,
            bias=True)

        if has_mmcv_full:
            self.adapt_conv = DeformConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                groups=groups,
                deform_groups=groups)
        else:
            raise ImportError('Please install the full version of mmcv '
                              'to use `DeformConv2d`.')

        self.norm = build_norm_layer(norm_cfg, out_channels)[1]
        self.act = build_activation_layer(act_cfg)

    def forward(self, x):
        B, _, H, W = x.size()
        residual = x

        affine_matrix = self.transform_matrix_conv(x)
        affine_matrix = affine_matrix.permute(0, 2, 3, 1).contiguous()
        affine_matrix = affine_matrix.view(B, H, W, self.groups, 2, 3)
        offset = torch.matmul(affine_matrix, self.regular_matrix)
        offset = offset.transpose(4, 5).reshape(B, H, W, self.groups * 18)
        offset = offset.permute(0, 3, 1, 2).contiguous()

        x = self.adapt_conv(x, offset)
        x = self.norm(x)
        x = self.act(x + residual)

        return x


@HEADS.register_module()
class DEKRHead(DeconvHead):
    """DisEntangled Keypoint Regression head. "Bottom-up human pose estimation
    via disentangled keypoint regression", CVPR'2021.

    Args:
        in_channels (int): Number of input channels.
        num_joints (int): Number of joints.
        num_heatmap_filters (int): Number of filters for heatmap branch.
        num_offset_filters_per_joint (int): Number of filters for each joint.
        in_index (int|Sequence[int]): Input feature index. Default: 0
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            Default: None.

            - 'resize_concat': Multiple feature maps will be resized to the
                same size as the first one and then concat together.
                Usually used in FCN head of HRNet.
            - 'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            - None: Only one select feature map is allowed.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        heatmap_loss (dict): Config for heatmap loss. Default: None.
        offset_loss (dict): Config for offset loss. Default: None.
    """

    def __init__(self,
                 in_channels,
                 num_joints,
                 num_heatmap_filters=32,
                 num_offset_filters_per_joint=15,
                 in_index=0,
                 input_transform=None,
                 num_deconv_layers=0,
                 num_deconv_filters=None,
                 num_deconv_kernels=None,
                 extra=dict(final_conv_kernel=0),
                 align_corners=False,
                 heatmap_loss=None,
                 offset_loss=None):

        super().__init__(
            in_channels,
            out_channels=in_channels,
            num_deconv_layers=num_deconv_layers,
            num_deconv_filters=num_deconv_filters,
            num_deconv_kernels=num_deconv_kernels,
            align_corners=align_corners,
            in_index=in_index,
            input_transform=input_transform,
            extra=extra,
            loss_keypoint=heatmap_loss)

        # set up filters for heatmap
        self.heatmap_conv_layers = nn.Sequential(
            ConvModule(
                in_channels=self.in_channels,
                out_channels=num_heatmap_filters,
                kernel_size=1,
                norm_cfg=dict(type='BN')),
            BasicBlock(num_heatmap_filters, num_heatmap_filters),
            build_conv_layer(
                dict(type='Conv2d'),
                in_channels=num_heatmap_filters,
                out_channels=1 + num_joints,
                kernel_size=1))

        # set up filters for offset map
        groups = num_joints
        num_offset_filters = num_joints * num_offset_filters_per_joint

        self.offset_conv_layers = nn.Sequential(
            ConvModule(
                in_channels=self.in_channels,
                out_channels=num_offset_filters,
                kernel_size=1,
                norm_cfg=dict(type='BN')),
            AdaptiveActivationBlock(
                num_offset_filters, num_offset_filters, groups=groups),
            AdaptiveActivationBlock(
                num_offset_filters, num_offset_filters, groups=groups),
            build_conv_layer(
                dict(type='Conv2d'),
                in_channels=num_offset_filters,
                out_channels=2 * num_joints,
                kernel_size=1,
                groups=groups))

        # set up offset losses
        self.offset_loss = build_loss(copy.deepcopy(offset_loss))

    def get_loss(self, outputs, heatmaps, masks, offsets, offset_weights):
        """Calculate the dekr loss.

        Note:
            - batch_size: N
            - num_channels: C
            - num_joints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            outputs (List(torch.Tensor[N,C,H,W])): Multi-scale outputs.
            heatmaps (List(torch.Tensor[N,K+1,H,W])): Multi-scale heatmap
                targets.
            masks (List(torch.Tensor[N,K+1,H,W])): Weights of multi-scale
                heatmap targets.
            offsets (List(torch.Tensor[N,K*2,H,W])): Multi-scale offset
                targets.
            offset_weights (List(torch.Tensor[N,K*2,H,W])): Weights of
                multi-scale offset targets.
        """

        losses = dict()

        for idx in range(len(outputs)):
            pred_heatmap, pred_offset = outputs[idx]
            heatmap_weight = masks[idx].view(masks[idx].size(0),
                                             masks[idx].size(1), -1)
            losses['loss_hms'] = losses.get('loss_hms', 0) + self.loss(
                pred_heatmap, heatmaps[idx], heatmap_weight)
            losses['loss_ofs'] = losses.get('loss_ofs', 0) + self.offset_loss(
                pred_offset, offsets[idx], offset_weights[idx])

        return losses

    def forward(self, x):
        """Forward function."""
        x = self._transform_inputs(x)
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        heatmap = self.heatmap_conv_layers(x)
        offset = self.offset_conv_layers(x)
        return [[heatmap, offset]]

    def init_weights(self):
        """Initialize model weights."""
        super().init_weights()
        for name, m in self.heatmap_conv_layers.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        for name, m in self.offset_conv_layers.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'transform_matrix_conv' in name:
                    normal_init(m, std=1e-8, bias=0)
                else:
                    normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
