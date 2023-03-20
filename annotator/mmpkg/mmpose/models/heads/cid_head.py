# Copyright (c) OpenMMLab. All rights reserved.
import math
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import HEADS, build_loss


def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
    return y


@HEADS.register_module()
class CIDHead(nn.Module):
    """CID head. paper ref: Dongkai Wang et al. "Contextual Instance Decouple
    for Robust Multi-Person Pose Estimation".

    Args:
        in_channels (int): Number of input channels.
        gfd_channels (int): Number of instance feature map channels
        num_joints (int): Number of joints
        multi_hm_loss_factor (float): loss weight for multi-person heatmap
        single_hm_loss_factor (float): loss weight for single person heatmap
        contrastive_loss_factor (float): loss weight for contrastive loss
        max_train_instances (int): limit the number of instances
        during training to avoid
        prior_prob (float): focal loss bias initialization
    """

    def __init__(self,
                 in_channels,
                 gfd_channels,
                 num_joints,
                 multi_hm_loss_factor=1.0,
                 single_hm_loss_factor=4.0,
                 contrastive_loss_factor=1.0,
                 max_train_instances=200,
                 prior_prob=0.01):
        super().__init__()
        self.multi_hm_loss_factor = multi_hm_loss_factor
        self.single_hm_loss_factor = single_hm_loss_factor
        self.contrastive_loss_factor = contrastive_loss_factor
        self.max_train_instances = max_train_instances
        self.prior_prob = prior_prob

        # iia module
        self.keypoint_center_conv = nn.Conv2d(in_channels, num_joints + 1, 1,
                                              1, 0)
        # gfd module
        self.conv_down = nn.Conv2d(in_channels, gfd_channels, 1, 1, 0)
        self.c_attn = ChannelAtten(in_channels, gfd_channels)
        self.s_attn = SpatialAtten(in_channels, gfd_channels)
        self.fuse_attn = nn.Conv2d(gfd_channels * 2, gfd_channels, 1, 1, 0)
        self.heatmap_conv = nn.Conv2d(gfd_channels, num_joints, 1, 1, 0)

        # loss
        self.heatmap_loss = build_loss(dict(type='FocalHeatmapLoss'))
        self.contrastive_loss = ContrastiveLoss()

        # initialize
        self.init_weights()

    def forward(self, features, forward_info=None):
        """Forward function."""
        assert isinstance(features, list)
        x0_h, x0_w = features[0].size(2), features[0].size(3)

        features = torch.cat([
            features[0],
            F.interpolate(
                features[1],
                size=(x0_h, x0_w),
                mode='bilinear',
                align_corners=False),
            F.interpolate(
                features[2],
                size=(x0_h, x0_w),
                mode='bilinear',
                align_corners=False),
            F.interpolate(
                features[3],
                size=(x0_h, x0_w),
                mode='bilinear',
                align_corners=False)
        ], 1)

        if self.training:
            return self.forward_train(features, forward_info)
        else:
            return self.forward_test(features, forward_info)

    def forward_train(self, features, labels):
        gt_multi_heatmap, gt_multi_mask, gt_instance_coord,\
            gt_instance_heatmap, gt_instance_mask, gt_instance_valid = labels

        pred_multi_heatmap = _sigmoid(self.keypoint_center_conv(features))

        # multi-person heatmap loss
        multi_heatmap_loss = self.heatmap_loss(pred_multi_heatmap,
                                               gt_multi_heatmap, gt_multi_mask)

        contrastive_loss = 0
        total_instances = 0
        instances = defaultdict(list)
        for i in range(features.size(0)):
            if torch.sum(gt_instance_valid[i]) < 0.5:
                continue
            instance_coord = gt_instance_coord[i][
                gt_instance_valid[i] > 0.5].long()
            instance_heatmap = gt_instance_heatmap[i][
                gt_instance_valid[i] > 0.5]
            instance_mask = gt_instance_mask[i][gt_instance_valid[i] > 0.5]
            instance_imgid = i * torch.ones(
                instance_coord.size(0),
                dtype=torch.long,
                device=features.device)
            instance_param = self._sample_feats(features[i], instance_coord)
            contrastive_loss += self.contrastive_loss(instance_param)
            total_instances += instance_coord.size(0)

            instances['instance_coord'].append(instance_coord)
            instances['instance_imgid'].append(instance_imgid)
            instances['instance_param'].append(instance_param)
            instances['instance_heatmap'].append(instance_heatmap)
            instances['instance_mask'].append(instance_mask)

        if total_instances <= 0:
            losses = dict()
            losses['multi_heatmap_loss'] = multi_heatmap_loss * \
                self.multi_hm_loss_factor
            losses['single_heatmap_loss'] = torch.zeros_like(
                multi_heatmap_loss)
            losses['contrastive_loss'] = torch.zeros_like(multi_heatmap_loss)
            return losses

        contrastive_loss = contrastive_loss / total_instances

        for k, v in instances.items():
            instances[k] = torch.cat(v, dim=0)

        # limit max instances in training
        if 0 <= self.max_train_instances < instances['instance_param'].size(0):
            inds = torch.randperm(
                instances['instance_param'].size(0),
                device=features.device).long()
            for k, v in instances.items():
                instances[k] = v[inds[:self.max_train_instances]]

        # single person heatmap loss
        global_features = self.conv_down(features)
        instance_features = global_features[instances['instance_imgid']]
        instance_params = instances['instance_param']
        c_instance_feats = self.c_attn(instance_features, instance_params)
        s_instance_feats = self.s_attn(instance_features, instance_params,
                                       instances['instance_coord'])
        cond_instance_feats = torch.cat((c_instance_feats, s_instance_feats),
                                        dim=1)
        cond_instance_feats = self.fuse_attn(cond_instance_feats)
        cond_instance_feats = F.relu(cond_instance_feats)

        pred_instance_heatmaps = _sigmoid(
            self.heatmap_conv(cond_instance_feats))

        gt_instance_heatmaps = instances['instance_heatmap']
        gt_instance_masks = instances['instance_mask']
        single_heatmap_loss = self.heatmap_loss(pred_instance_heatmaps,
                                                gt_instance_heatmaps,
                                                gt_instance_masks)

        losses = dict()
        losses['multi_heatmap_loss'] = multi_heatmap_loss *\
            self.multi_hm_loss_factor
        losses['single_heatmap_loss'] = single_heatmap_loss *\
            self.single_hm_loss_factor
        losses['contrastive_loss'] = contrastive_loss *\
            self.contrastive_loss_factor

        return losses

    def forward_test(self, features, test_cfg):
        flip_test = test_cfg.get('flip_test', False)
        center_pool_kernel = test_cfg.get('center_pool_kernel', 3)
        max_proposals = test_cfg.get('max_num_people', 30)
        keypoint_thre = test_cfg.get('detection_threshold', 0.01)

        # flip back feature map
        if flip_test:
            features[1, :, :, :] = features[1, :, :, :].flip([2])

        instances = {}
        pred_multi_heatmap = _sigmoid(self.keypoint_center_conv(features))
        W = pred_multi_heatmap.size()[-1]

        if flip_test:
            center_heatmap = pred_multi_heatmap[:, -1, :, :].mean(
                dim=0, keepdim=True)
        else:
            center_heatmap = pred_multi_heatmap[:, -1, :, :]

        center_pool = F.avg_pool2d(center_heatmap, center_pool_kernel, 1,
                                   (center_pool_kernel - 1) // 2)
        center_heatmap = (center_heatmap + center_pool) / 2.0
        maxm = self.hierarchical_pool(center_heatmap)
        maxm = torch.eq(maxm, center_heatmap).float()
        center_heatmap = center_heatmap * maxm
        scores = center_heatmap.view(-1)
        scores, pos_ind = scores.topk(max_proposals, dim=0)
        select_ind = (scores > (keypoint_thre)).nonzero()

        if len(select_ind) == 0:
            return [], []

        scores = scores[select_ind].squeeze(1)
        pos_ind = pos_ind[select_ind].squeeze(1)
        x = pos_ind % W
        y = pos_ind // W
        instance_coord = torch.stack((y, x), dim=1)
        instance_param = self._sample_feats(features[0], instance_coord)
        instance_imgid = torch.zeros(
            instance_coord.size(0), dtype=torch.long).to(features.device)
        if flip_test:
            instance_param_flip = self._sample_feats(features[1],
                                                     instance_coord)
            instance_imgid_flip = torch.ones(
                instance_coord.size(0), dtype=torch.long).to(features.device)
            instance_coord = torch.cat((instance_coord, instance_coord), dim=0)
            instance_param = torch.cat((instance_param, instance_param_flip),
                                       dim=0)
            instance_imgid = torch.cat((instance_imgid, instance_imgid_flip),
                                       dim=0)

        instances['instance_coord'] = instance_coord
        instances['instance_imgid'] = instance_imgid
        instances['instance_param'] = instance_param

        global_features = self.conv_down(features)
        instance_features = global_features[instances['instance_imgid']]
        instance_params = instances['instance_param']
        c_instance_feats = self.c_attn(instance_features, instance_params)
        s_instance_feats = self.s_attn(instance_features, instance_params,
                                       instances['instance_coord'])
        cond_instance_feats = torch.cat((c_instance_feats, s_instance_feats),
                                        dim=1)
        cond_instance_feats = self.fuse_attn(cond_instance_feats)
        cond_instance_feats = F.relu(cond_instance_feats)

        instance_heatmaps = _sigmoid(self.heatmap_conv(cond_instance_feats))

        if flip_test:
            instance_heatmaps, instance_heatmaps_flip = torch.chunk(
                instance_heatmaps, 2, dim=0)
            instance_heatmaps_flip = instance_heatmaps_flip[:, test_cfg[
                'flip_index'], :, :]
            instance_heatmaps = (instance_heatmaps +
                                 instance_heatmaps_flip) / 2.0

        return instance_heatmaps, scores

    def _sample_feats(self, features, pos_ind):
        feats = features[:, pos_ind[:, 0], pos_ind[:, 1]]
        return feats.permute(1, 0)

    def hierarchical_pool(self, heatmap):
        map_size = (heatmap.shape[1] + heatmap.shape[2]) / 2.0
        if map_size > 300:
            maxm = F.max_pool2d(heatmap, 7, 1, 3)
        elif map_size > 200:
            maxm = F.max_pool2d(heatmap, 5, 1, 2)
        else:
            maxm = F.max_pool2d(heatmap, 3, 1, 1)
        return maxm

    def init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
        # focal loss init
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        torch.nn.init.constant_(self.keypoint_center_conv.bias, bias_value)
        torch.nn.init.constant_(self.heatmap_conv.bias, bias_value)


class ChannelAtten(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ChannelAtten, self).__init__()
        self.atn = nn.Linear(in_channels, out_channels)

    def forward(self, global_features, instance_params):
        B, C, H, W = global_features.size()
        instance_params = self.atn(instance_params).reshape(B, C, 1, 1)
        return global_features * instance_params.expand_as(global_features)


class SpatialAtten(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(SpatialAtten, self).__init__()
        self.atn = nn.Linear(in_channels, out_channels)
        self.feat_stride = 4
        conv_in = 3
        self.conv = nn.Conv2d(conv_in, 1, 5, 1, 2)

    def forward(self, global_features, instance_params, instance_inds):
        B, C, H, W = global_features.size()
        instance_params = self.atn(instance_params).reshape(B, C, 1, 1)
        feats = global_features * instance_params.expand_as(global_features)
        fsum = torch.sum(feats, dim=1, keepdim=True)
        input_feats = fsum
        locations = compute_locations(
            global_features.size(2),
            global_features.size(3),
            stride=1,
            device=global_features.device)
        n_inst = instance_inds.size(0)
        H, W = global_features.size()[2:]
        instance_locations = torch.flip(instance_inds, [1])
        instance_locations = instance_locations
        relative_coords = instance_locations.reshape(
            -1, 1, 2) - locations.reshape(1, -1, 2)
        relative_coords = relative_coords.permute(0, 2, 1).float()
        relative_coords = (relative_coords /
                           32).to(dtype=global_features.dtype)
        relative_coords = relative_coords.reshape(n_inst, 2, H, W)
        input_feats = torch.cat((input_feats, relative_coords), dim=1)
        mask = self.conv(input_feats).sigmoid()
        return global_features * mask


class ContrastiveLoss(nn.Module):

    def __init__(self, temperature=0.05):
        super(ContrastiveLoss, self).__init__()
        self.temp = temperature

    def forward(self, features):
        n = features.size(0)
        features_norm = F.normalize(features, dim=1)
        logits = features_norm.mm(features_norm.t()) / self.temp
        targets = torch.arange(n, dtype=torch.long, device=features.device)
        loss = F.cross_entropy(logits, targets, reduction='sum')
        return loss


def compute_locations(h, w, stride, device):
    shifts_x = torch.arange(
        0, w * stride, step=stride, dtype=torch.float32, device=device)
    shifts_y = torch.arange(
        0, h * stride, step=stride, dtype=torch.float32, device=device)
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations
