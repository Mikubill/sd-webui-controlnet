# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import annotator.mmpkg.mmcv as mmcv
import numpy as np
import torch
from annotator.mmpkg.mmcv.image import imwrite
from annotator.mmpkg.mmcv.utils.misc import deprecated_api_warning
from annotator.mmpkg.mmcv.visualization.image import imshow

from annotator.mmpkg.mmpose.core.evaluation import (aggregate_scale, aggregate_stage_flip,
                                    flip_feature_maps, get_group_preds)
from annotator.mmpkg.mmpose.core.post_processing import nearby_joints_nms
from annotator.mmpkg.mmpose.core.post_processing.group import HeatmapOffsetParser
from annotator.mmpkg.mmpose.core.visualization import imshow_keypoints
from .. import builder
from ..builder import POSENETS
from ..utils import DekrRescoreNet
from .base import BasePose

try:
    from annotator.mmpkg.mmcv.runner import auto_fp16
except ImportError:
    warnings.warn('auto_fp16 from annotator.mmpkg.mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from annotator.mmpkg.mmpose.core import auto_fp16


@POSENETS.register_module()
class DisentangledKeypointRegressor(BasePose):
    """Disentangled keypoint regression pose detector.

    Args:
        backbone (dict): Backbone modules to extract feature.
        keypoint_head (dict): Keypoint head to process feature.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path to the pretrained models.
    """

    def __init__(self,
                 backbone,
                 keypoint_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__()
        self.fp16_enabled = False

        self.backbone = builder.build_backbone(backbone)

        if keypoint_head is not None:
            self.keypoint_head = builder.build_head(keypoint_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.use_udp = test_cfg.get('use_udp', False)
        self.parser = HeatmapOffsetParser(self.test_cfg)
        self.pretrained = pretrained

        rescore_cfg = test_cfg.get('rescore_cfg', None)
        if rescore_cfg is not None:
            self.rescore_net = DekrRescoreNet(**rescore_cfg)

        self.init_weights()

    @property
    def with_keypoint(self):
        """Check if has keypoint_head."""
        return hasattr(self, 'keypoint_head')

    def init_weights(self, pretrained=None):
        """Weight initialization for model."""
        if pretrained is not None:
            self.pretrained = pretrained
        self.backbone.init_weights(self.pretrained)
        if self.with_keypoint:
            self.keypoint_head.init_weights()
        if hasattr(self, 'rescore_net'):
            self.rescore_net.init_weight()

    @auto_fp16(apply_to=('img', ))
    def forward(self,
                img=None,
                heatmaps=None,
                masks=None,
                offsets=None,
                offset_weights=None,
                img_metas=None,
                return_loss=True,
                return_heatmap=False,
                **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss is True.

        Note:
            - batch_size: N
            - num_keypoints: K
            - num_img_channel: C
            - img_width: imgW
            - img_height: imgH
            - heatmaps weight: W
            - heatmaps height: H
            - max_num_people: M

        Args:
            img (torch.Tensor[N,C,imgH,imgW]): # input image.
            targets (list(torch.Tensor[N,K,H,W])): Multi-scale target heatmaps.
            masks (list(torch.Tensor[N,H,W])): Masks of multi-scale target
                heatmaps
            joints (list(torch.Tensor[N,M,K,2])): Joints of multi-scale target
                heatmaps for ae loss
            img_metas (dict): Information about val & test.
                By default it includes:

                - "image_file": image path
                - "aug_data": # input
                - "test_scale_factor": test scale factor
                - "base_size": base size of # input
                - "center": center of image
                - "scale": scale of image
                - "flip_index": flip index of keypoints
            return loss (bool): ``return_loss=True`` for training,
                ``return_loss=False`` for validation & test.
            return_heatmap (bool) : Option to return heatmap.

        Returns:
            dict|tuple: if 'return_loss' is true, then return losses. \
                Otherwise, return predicted poses, scores, image \
                paths and heatmaps.
        """

        if return_loss:
            return self.forward_train(img, heatmaps, masks, offsets,
                                      offset_weights, img_metas, **kwargs)
        return self.forward_test(
            img, img_metas, return_heatmap=return_heatmap, **kwargs)

    def forward_train(self, img, heatmaps, masks, offsets, offset_weights,
                      img_metas, **kwargs):
        """Forward the bottom-up model and calculate the loss.

        Note:
            batch_size: N
            num_keypoints: K
            num_img_channel: C
            img_width: imgW
            img_height: imgH
            heatmaps weight: W
            heatmaps height: H
            max_num_people: M

        Args:
            img (torch.Tensor[N,C,imgH,imgW]): # input image.
            targets (List(torch.Tensor[N,K,H,W])): Multi-scale target heatmaps.
            masks (List(torch.Tensor[N,H,W])): Masks of multi-scale target
                                              heatmaps
            joints (List(torch.Tensor[N,M,K,2])): Joints of multi-scale target
                                                 heatmaps for ae loss
            img_metas (dict):Information about val&test
                By default this includes:
                - "image_file": image path
                - "aug_data": # input
                - "test_scale_factor": test scale factor
                - "base_size": base size of # input
                - "center": center of image
                - "scale": scale of image
                - "flip_index": flip index of keypoints

        Returns:
            dict: The total loss for bottom-up
        """

        output = self.backbone(img)

        if self.with_keypoint:
            output = self.keypoint_head(output)

        # if return loss
        losses = dict()
        if self.with_keypoint:
            keypoint_losses = self.keypoint_head.get_loss(
                output,
                heatmaps,
                masks,
                offsets,
                offset_weights,
            )
            losses.update(keypoint_losses)

        return losses

    def forward_dummy(self, img):
        """Used for computing network FLOPs.

        See ``tools/get_flops.py``.

        Args:
            img (torch.Tensor): # input image.

        Returns:
            Tensor: Outputs.
        """
        output = self.backbone(img)
        if self.with_keypoint:
            output = self.keypoint_head(output)
        return output

    def forward_test(self, img, img_metas, return_heatmap=False, **kwargs):
        """Inference the one-stage model.

        Note:
            - Batchsize: N (currently support batchsize = 1)
            - num_img_channel: C
            - img_width: imgW
            - img_height: imgH

        Args:
            flip_index (List(int)):
            aug_data (List(Tensor[NxCximgHximgW])): Multi-scale image
            num_joints (int): Number of joints of an instsance.\
            test_scale_factor (List(float)): Multi-scale factor
            base_size (Tuple(int)): Base size of image when scale is 1
            image_size (int): Short edge of images when scale is 1
            heatmap_size (int): Short edge of outputs when scale is 1
            center (np.ndarray): center of image
            scale (np.ndarray): the scale of image
            skeleton (List(List(int))): Links of joints
        """
        assert img.size(0) == 1
        assert len(img_metas) == 1

        img_metas = img_metas[0]

        flip_index = img_metas['flip_index']
        aug_data = img_metas['aug_data']
        num_joints = img_metas['num_joints']
        test_scale_factor = img_metas['test_scale_factor']
        base_size = img_metas['base_size']
        image_size = img_metas['image_size']
        heatmap_size = img_metas['heatmap_size'][0]
        center = img_metas['center']
        scale = img_metas['scale']
        skeleton = img_metas['skeleton']

        result = {}

        scale_heatmaps_list = []
        scale_poses_dict = dict()

        for idx, s in enumerate(sorted(test_scale_factor, reverse=True)):
            image_resized = aug_data[idx].to(img.device)

            features = self.backbone(image_resized)
            if self.with_keypoint:
                outputs = self.keypoint_head(features)
            heatmaps, offsets = outputs[0]

            if self.test_cfg.get('flip_test', True):
                # use flip test
                image_flipped = torch.flip(image_resized, [3])
                features_flipped = self.backbone(image_flipped)
                if self.with_keypoint:
                    outputs_flipped = self.keypoint_head(features_flipped)
                heatmaps_flipped, offsets_flipped = outputs_flipped[0]

                # compute heatmaps for flipped input image
                center_heatmaps_flipped = flip_feature_maps(
                    [heatmaps_flipped[:, :1]], None)[0]
                keypoint_heatmaps_flipped = flip_feature_maps(
                    [heatmaps_flipped[:, 1:]], flip_index=flip_index)[0]
                heatmaps_flipped = torch.cat(
                    [center_heatmaps_flipped, keypoint_heatmaps_flipped],
                    dim=1)

                # compute offsets for flipped input image
                h, w = offsets_flipped.shape[2], offsets_flipped.shape[3]
                offsets_flipped = offsets_flipped.view(num_joints, 2, h, w)
                offsets_flipped = offsets_flipped.transpose(1, 0).contiguous()
                offsets_flipped[0] = -offsets_flipped[0] - 1
                offsets_flipped = flip_feature_maps([offsets_flipped],
                                                    flip_index=flip_index)[0]
                offsets_flipped = offsets_flipped.transpose(1, 0).reshape(
                    1, -1, h, w)

                heatmaps_flipped = [heatmaps_flipped]
                offsets_flipped = [offsets_flipped]

            else:
                heatmaps_flipped = None
                offsets_flipped = None

            # aggregate heatmaps and offsets
            aggregated_heatmaps = aggregate_stage_flip(
                [heatmaps],
                heatmaps_flipped,
                index=-1,
                project2image=self.test_cfg['project2image'],
                size_projected=base_size,
                align_corners=self.test_cfg.get('align_corners', True),
                aggregate_stage='average',
                aggregate_flip='average')[0]
            scale_heatmaps_list.append(aggregated_heatmaps)

            aggregated_offsets = aggregate_stage_flip(
                [offsets],
                offsets_flipped,
                index=-1,
                project2image=self.test_cfg['project2image'],
                size_projected=base_size,
                align_corners=self.test_cfg.get('align_corners', True),
                aggregate_stage='average',
                aggregate_flip='average')[0]

            poses = self.parser.decode(aggregated_heatmaps, aggregated_offsets)
            # rescale pose coordinates to a unified scale
            poses[..., :2] *= (image_size * 1.0 / heatmap_size) / s
            scale_poses_dict[s] = poses

        # aggregate multi-scale heatmaps
        aggregated_heatmaps = aggregate_scale(
            scale_heatmaps_list,
            align_corners=self.test_cfg.get('align_corners', True),
            aggregate_scale='average',
            size_projected=base_size)

        # rescale the score of instances inferred from difference scales
        max_score_ref = 1
        if len(scale_poses_dict.get(1, [])) > 0:
            max_score_ref = scale_poses_dict[1][..., 2].max()

        for s, poses in scale_poses_dict.items():
            if s != 1.0 and poses.shape[0]:
                rescale_factor = max_score_ref / poses[..., 2].max()
                poses[..., 2] *= rescale_factor * self.test_cfg.get(
                    'multi_scale_score_decrease', 1.0)

        poses = torch.cat(tuple(scale_poses_dict.values()))
        # refine keypoint scores using keypoint heatmaps
        poses = self.parser.refine_score(aggregated_heatmaps, poses)
        poses = poses.cpu().numpy()

        # nms
        if poses.shape[0] and self.test_cfg.get('use_nms', False):
            kpts_db = []
            for i in range(len(poses)):
                kpts_db.append(
                    dict(keypoints=poses[i, :, :2], score=poses[i, :, 3]))

            keep_pose_inds = nearby_joints_nms(
                kpts_db,
                self.test_cfg['nms_dist_thr'],
                self.test_cfg['nms_joints_thr'],
                score_per_joint=True,
                max_dets=self.test_cfg['max_num_people'])
            poses = poses[keep_pose_inds]
        scores = poses[..., 2].mean(axis=1)

        # recover the pose to match the size of original image
        preds = get_group_preds(
            poses[None], center, scale, base_size, use_udp=self.use_udp)

        image_paths = []
        image_paths.append(img_metas['image_file'])

        if return_heatmap:
            output_heatmap = aggregated_heatmaps.detach().cpu().numpy()
        else:
            output_heatmap = None

        # rescore each instance with a pretrained rescore net
        if hasattr(self, 'rescore_net') and len(preds) > 0:
            re_scores = self.rescore_net(np.stack(preds, axis=0), skeleton)
            re_scores = re_scores.cpu().numpy()
            re_scores[np.isnan(re_scores)] = 0
            scores *= re_scores

        result['preds'] = preds
        result['scores'] = scores
        result['image_paths'] = image_paths
        result['output_heatmap'] = output_heatmap

        return result

    @deprecated_api_warning({'pose_limb_color': 'pose_link_color'},
                            cls_name='AssociativeEmbedding')
    def show_result(self,
                    img,
                    result,
                    skeleton=None,
                    kpt_score_thr=0.3,
                    bbox_color=None,
                    pose_kpt_color=None,
                    pose_link_color=None,
                    radius=4,
                    thickness=1,
                    font_scale=0.5,
                    win_name='',
                    show=False,
                    show_keypoint_weight=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
            skeleton (list[list]): The connection of keypoints.
                skeleton is 0-based indexing.
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            pose_kpt_color (np.array[Nx3]`): Color of N keypoints.
                If None, do not draw keypoints.
            pose_link_color (np.array[Mx3]): Color of M links.
                If None, do not draw links.
            radius (int): Radius of circles.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            show (bool): Whether to show the image. Default: False.
            show_keypoint_weight (bool): Whether to change the transparency
                using the predicted confidence scores of keypoints.
            wait_time (int): Value of waitKey param.
                Default: 0.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            Tensor: Visualized image only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        img_h, img_w, _ = img.shape

        pose_result = []
        for res in result:
            pose_result.append(res['keypoints'])

        imshow_keypoints(img, pose_result, skeleton, kpt_score_thr,
                         pose_kpt_color, pose_link_color, radius, thickness)

        if show:
            imshow(img, win_name, wait_time)

        if out_file is not None:
            imwrite(img, out_file)

        return img
