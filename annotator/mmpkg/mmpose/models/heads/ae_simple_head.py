# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import HEADS
from .deconv_head import DeconvHead


@HEADS.register_module()
class AESimpleHead(DeconvHead):
    """Associative embedding simple head.
    paper ref: Alejandro Newell et al. "Associative
    Embedding: End-to-end Learning for Joint Detection
    and Grouping"

    Args:
        in_channels (int): Number of input channels.
        num_joints (int): Number of joints.
        num_deconv_layers (int): Number of deconv layers.
            num_deconv_layers should >= 0. Note that 0 means
            no deconv layers.
        num_deconv_filters (list|tuple): Number of filters.
            If num_deconv_layers > 0, the length of
        num_deconv_kernels (list|tuple): Kernel sizes.
        tag_per_joint (bool): If tag_per_joint is True,
            the dimension of tags equals to num_joints,
            else the dimension of tags is 1. Default: True
        with_ae_loss (list[bool]): Option to use ae loss or not.
        loss_keypoint (dict): Config for loss. Default: None.
    """

    def __init__(self,
                 in_channels,
                 num_joints,
                 num_deconv_layers=3,
                 num_deconv_filters=(256, 256, 256),
                 num_deconv_kernels=(4, 4, 4),
                 tag_per_joint=True,
                 with_ae_loss=None,
                 extra=None,
                 loss_keypoint=None):

        dim_tag = num_joints if tag_per_joint else 1
        if with_ae_loss[0]:
            out_channels = num_joints + dim_tag
        else:
            out_channels = num_joints

        super().__init__(
            in_channels,
            out_channels,
            num_deconv_layers=num_deconv_layers,
            num_deconv_filters=num_deconv_filters,
            num_deconv_kernels=num_deconv_kernels,
            extra=extra,
            loss_keypoint=loss_keypoint)

    def get_loss(self, outputs, targets, masks, joints):
        """Calculate bottom-up keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K
            - num_outputs: O
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            outputs (list(torch.Tensor[N,K,H,W])): Multi-scale output heatmaps.
            targets (List(torch.Tensor[N,K,H,W])): Multi-scale target heatmaps.
            masks (List(torch.Tensor[N,H,W])): Masks of multi-scale target
                heatmaps
            joints(List(torch.Tensor[N,M,K,2])): Joints of multi-scale target
                heatmaps for ae loss
        """

        losses = dict()

        heatmaps_losses, push_losses, pull_losses = self.loss(
            outputs, targets, masks, joints)

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
