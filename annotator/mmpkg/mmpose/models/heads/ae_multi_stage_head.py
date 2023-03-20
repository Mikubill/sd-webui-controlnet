# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from annotator.mmpkg.mmcv.cnn import (build_conv_layer, build_upsample_layer, constant_init,
                      normal_init)

from annotator.mmpkg.mmpose.models.builder import build_loss
from ..builder import HEADS


@HEADS.register_module()
class AEMultiStageHead(nn.Module):
    """Associative embedding multi-stage head.
    paper ref: Alejandro Newell et al. "Associative
    Embedding: End-to-end Learning for Joint Detection
    and Grouping"

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_deconv_layers (int): Number of deconv layers.
            num_deconv_layers should >= 0. Note that 0 means
            no deconv layers.
        num_deconv_filters (list|tuple): Number of filters.
            If num_deconv_layers > 0, the length of
        num_deconv_kernels (list|tuple): Kernel sizes.
        loss_keypoint (dict): Config for loss. Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_stages=1,
                 num_deconv_layers=3,
                 num_deconv_filters=(256, 256, 256),
                 num_deconv_kernels=(4, 4, 4),
                 extra=None,
                 loss_keypoint=None):
        super().__init__()

        self.loss = build_loss(loss_keypoint)

        self.in_channels = in_channels
        self.num_stages = num_stages

        if extra is not None and not isinstance(extra, dict):
            raise TypeError('extra should be dict or None.')

        # build multi-stage deconv layers
        self.multi_deconv_layers = nn.ModuleList([])
        for _ in range(self.num_stages):
            if num_deconv_layers > 0:
                deconv_layers = self._make_deconv_layer(
                    num_deconv_layers,
                    num_deconv_filters,
                    num_deconv_kernels,
                )
            elif num_deconv_layers == 0:
                deconv_layers = nn.Identity()
            else:
                raise ValueError(
                    f'num_deconv_layers ({num_deconv_layers}) should >= 0.')
            self.multi_deconv_layers.append(deconv_layers)

        identity_final_layer = False
        if extra is not None and 'final_conv_kernel' in extra:
            assert extra['final_conv_kernel'] in [0, 1, 3]
            if extra['final_conv_kernel'] == 3:
                padding = 1
            elif extra['final_conv_kernel'] == 1:
                padding = 0
            else:
                # 0 for Identity mapping.
                identity_final_layer = True
            kernel_size = extra['final_conv_kernel']
        else:
            kernel_size = 1
            padding = 0

        # build multi-stage final layers
        self.multi_final_layers = nn.ModuleList([])
        for i in range(self.num_stages):
            if identity_final_layer:
                final_layer = nn.Identity()
            else:
                final_layer = build_conv_layer(
                    cfg=dict(type='Conv2d'),
                    in_channels=num_deconv_filters[-1]
                    if num_deconv_layers > 0 else in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding)
            self.multi_final_layers.append(final_layer)

    def get_loss(self, output, targets, masks, joints):
        """Calculate bottom-up keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (List(torch.Tensor[NxKxHxW])): Output heatmaps.
            targets(List(List(torch.Tensor[NxKxHxW]))):
                Multi-stage and multi-scale target heatmaps.
            masks(List(List(torch.Tensor[NxHxW]))):
                Masks of multi-stage and multi-scale target heatmaps
            joints(List(List(torch.Tensor[NxMxKx2]))):
                Joints of multi-stage multi-scale target heatmaps for ae loss
        """

        losses = dict()

        # Flatten list:
        # [stage_1_scale_1, stage_1_scale_2, ... , stage_1_scale_m,
        # ...
        # stage_n_scale_1, stage_n_scale_2, ... , stage_n_scale_m]
        targets = [target for _targets in targets for target in _targets]
        masks = [mask for _masks in masks for mask in _masks]
        joints = [joint for _joints in joints for joint in _joints]

        heatmaps_losses, push_losses, pull_losses = self.loss(
            output, targets, masks, joints)

        for idx in range(len(targets)):
            if heatmaps_losses[idx] is not None:
                heatmaps_loss = heatmaps_losses[idx].mean(dim=0)
                if 'heatmap_loss' not in losses:
                    losses['heatmap_loss'] = heatmaps_loss
                else:
                    losses['heatmap_loss'] += heatmaps_loss
            if push_losses[idx] is not None:
                push_loss = push_losses[idx].mean(dim=0)
                if 'push_loss' not in losses:
                    losses['push_loss'] = push_loss
                else:
                    losses['push_loss'] += push_loss
            if pull_losses[idx] is not None:
                pull_loss = pull_losses[idx].mean(dim=0)
                if 'pull_loss' not in losses:
                    losses['pull_loss'] = pull_loss
                else:
                    losses['pull_loss'] += pull_loss

        return losses

    def forward(self, x):
        """Forward function.

        Returns:
            out (list[Tensor]): a list of heatmaps from multiple stages.
        """
        out = []
        assert isinstance(x, list)
        for i in range(self.num_stages):
            y = self.multi_deconv_layers[i](x[i])
            y = self.multi_final_layers[i](y)
            out.append(y)
        return out

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""
        if num_layers != len(num_filters):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_filters({len(num_filters)})'
            raise ValueError(error_msg)
        if num_layers != len(num_kernels):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_kernels({len(num_kernels)})'
            raise ValueError(error_msg)

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                build_upsample_layer(
                    dict(type='deconv'),
                    in_channels=self.in_channels,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = planes

        return nn.Sequential(*layers)

    @staticmethod
    def _get_deconv_cfg(deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

        return deconv_kernel, padding, output_padding

    def init_weights(self):
        """Initialize model weights."""
        for _, m in self.multi_deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        for m in self.multi_final_layers.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
