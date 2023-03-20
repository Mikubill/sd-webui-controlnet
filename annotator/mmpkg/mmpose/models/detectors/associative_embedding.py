# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import annotator.mmpkg.mmcv as mmcv
import torch
from annotator.mmpkg.mmcv.image import imwrite
from annotator.mmpkg.mmcv.utils.misc import deprecated_api_warning
from annotator.mmpkg.mmcv.visualization.image import imshow

from annotator.mmpkg.mmpose.core.evaluation import (aggregate_scale, aggregate_stage_flip,
                                    flip_feature_maps, get_group_preds,
                                    split_ae_outputs)
from annotator.mmpkg.mmpose.core.post_processing.group import HeatmapParser
from annotator.mmpkg.mmpose.core.visualization import imshow_keypoints
from .. import builder
from ..builder import POSENETS
from .base import BasePose

try:
    from annotator.mmpkg.mmcv.runner import auto_fp16
except ImportError:
    warnings.warn('auto_fp16 from annotator.mmpkg.mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from annotator.mmpkg.mmpose.core import auto_fp16


@POSENETS.register_module()
class AssociativeEmbedding(BasePose):
    """Associative embedding pose detectors.

    Args:
        backbone (dict): Backbone modules to extract feature.
        keypoint_head (dict): Keypoint head to process feature.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path to the pretrained models.
        loss_pose (None): Deprecated arguments. Please use
            ``loss_keypoint`` for heads instead.
    """

    def __init__(self,
                 backbone,
                 keypoint_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 loss_pose=None):
        super().__init__()
        self.fp16_enabled = False

        self.backbone = builder.build_backbone(backbone)

        if keypoint_head is not None:
            if 'loss_keypoint' not in keypoint_head and loss_pose is not None:
                warnings.warn(
                    '`loss_pose` for BottomUp is deprecated, '
                    'use `loss_keypoint` for heads instead. See '
                    'https://github.com/open-mmlab/mmpose/pull/382'
                    ' for more information.', DeprecationWarning)
                keypoint_head['loss_keypoint'] = loss_pose

            self.keypoint_head = builder.build_head(keypoint_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.use_udp = test_cfg.get('use_udp', False)
        self.parser = HeatmapParser(self.test_cfg)
        self.pretrained = pretrained
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

    @auto_fp16(apply_to=('img', ))
    def forward(self,
                img=None,
                targets=None,
                masks=None,
                joints=None,
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
            img (torch.Tensor[N,C,imgH,imgW]): Input image.
            targets (list(torch.Tensor[N,K,H,W])): Multi-scale target heatmaps.
            masks (list(torch.Tensor[N,H,W])): Masks of multi-scale target
                heatmaps
            joints (list(torch.Tensor[N,M,K,2])): Joints of multi-scale target
                heatmaps for ae loss
            img_metas (dict): Information about val & test.
                By default it includes:

                - "image_file": image path
                - "aug_data": input
                - "test_scale_factor": test scale factor
                - "base_size": base size of input
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
            return self.forward_train(img, targets, masks, joints, img_metas,
                                      **kwargs)
        return self.forward_test(
            img, img_metas, return_heatmap=return_heatmap, **kwargs)

    def forward_train(self, img, targets, masks, joints, img_metas, **kwargs):
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
            img (torch.Tensor[N,C,imgH,imgW]): Input image.
            targets (List(torch.Tensor[N,K,H,W])): Multi-scale target heatmaps.
            masks (List(torch.Tensor[N,H,W])): Masks of multi-scale target
                                              heatmaps
            joints (List(torch.Tensor[N,M,K,2])): Joints of multi-scale target
                                                 heatmaps for ae loss
            img_metas (dict):Information about val&test
                By default this includes:
                - "image_file": image path
                - "aug_data": input
                - "test_scale_factor": test scale factor
                - "base_size": base size of input
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
                output, targets, masks, joints)
            losses.update(keypoint_losses)

        return losses

    def forward_dummy(self, img):
        """Used for computing network FLOPs.

        See ``tools/get_flops.py``.

        Args:
            img (torch.Tensor): Input image.

        Returns:
            Tensor: Outputs.
        """
        output = self.backbone(img)
        if self.with_keypoint:
            output = self.keypoint_head(output)
        return output

    def forward_test(self, img, img_metas, return_heatmap=False, **kwargs):
        """Inference the bottom-up model.

        Note:
            - Batchsize: N (currently support batchsize = 1)
            - num_img_channel: C
            - img_width: imgW
            - img_height: imgH

        Args:
            flip_index (List(int)):
            aug_data (List(Tensor[NxCximgHximgW])): Multi-scale image
            test_scale_factor (List(float)): Multi-scale factor
            base_size (Tuple(int)): Base size of image when scale is 1
            center (np.ndarray): center of image
            scale (np.ndarray): the scale of image
        """
        assert img.size(0) == 1
        assert len(img_metas) == 1

        img_metas = img_metas[0]

        aug_data = img_metas['aug_data']

        test_scale_factor = img_metas['test_scale_factor']
        base_size = img_metas['base_size']
        center = img_metas['center']
        scale = img_metas['scale']

        result = {}

        scale_heatmaps_list = []
        scale_tags_list = []

        for idx, s in enumerate(sorted(test_scale_factor, reverse=True)):
            image_resized = aug_data[idx].to(img.device)

            features = self.backbone(image_resized)
            if self.with_keypoint:
                outputs = self.keypoint_head(features)

            heatmaps, tags = split_ae_outputs(
                outputs, self.test_cfg['num_joints'],
                self.test_cfg['with_heatmaps'], self.test_cfg['with_ae'],
                self.test_cfg.get('select_output_index', range(len(outputs))))

            if self.test_cfg.get('flip_test', True):
                # use flip test
                features_flipped = self.backbone(
                    torch.flip(image_resized, [3]))
                if self.with_keypoint:
                    outputs_flipped = self.keypoint_head(features_flipped)

                heatmaps_flipped, tags_flipped = split_ae_outputs(
                    outputs_flipped, self.test_cfg['num_joints'],
                    self.test_cfg['with_heatmaps'], self.test_cfg['with_ae'],
                    self.test_cfg.get('select_output_index',
                                      range(len(outputs))))

                heatmaps_flipped = flip_feature_maps(
                    heatmaps_flipped, flip_index=img_metas['flip_index'])
                if self.test_cfg['tag_per_joint']:
                    tags_flipped = flip_feature_maps(
                        tags_flipped, flip_index=img_metas['flip_index'])
                else:
                    tags_flipped = flip_feature_maps(
                        tags_flipped, flip_index=None, flip_output=True)

            else:
                heatmaps_flipped = None
                tags_flipped = None

            aggregated_heatmaps = aggregate_stage_flip(
                heatmaps,
                heatmaps_flipped,
                index=-1,
                project2image=self.test_cfg['project2image'],
                size_projected=base_size,
                align_corners=self.test_cfg.get('align_corners', True),
                aggregate_stage='average',
                aggregate_flip='average')

            aggregated_tags = aggregate_stage_flip(
                tags,
                tags_flipped,
                index=-1,
                project2image=self.test_cfg['project2image'],
                size_projected=base_size,
                align_corners=self.test_cfg.get('align_corners', True),
                aggregate_stage='concat',
                aggregate_flip='concat')

            if s == 1 or len(test_scale_factor) == 1:
                if isinstance(aggregated_tags, list):
                    scale_tags_list.extend(aggregated_tags)
                else:
                    scale_tags_list.append(aggregated_tags)

            if isinstance(aggregated_heatmaps, list):
                scale_heatmaps_list.extend(aggregated_heatmaps)
            else:
                scale_heatmaps_list.append(aggregated_heatmaps)

        aggregated_heatmaps = aggregate_scale(
            scale_heatmaps_list,
            align_corners=self.test_cfg.get('align_corners', True),
            aggregate_scale='average')

        aggregated_tags = aggregate_scale(
            scale_tags_list,
            align_corners=self.test_cfg.get('align_corners', True),
            aggregate_scale='unsqueeze_concat')

        heatmap_size = aggregated_heatmaps.shape[2:4]
        tag_size = aggregated_tags.shape[2:4]
        if heatmap_size != tag_size:
            tmp = []
            for idx in range(aggregated_tags.shape[-1]):
                tmp.append(
                    torch.nn.functional.interpolate(
                        aggregated_tags[..., idx],
                        size=heatmap_size,
                        mode='bilinear',
                        align_corners=self.test_cfg.get('align_corners',
                                                        True)).unsqueeze(-1))
            aggregated_tags = torch.cat(tmp, dim=-1)

        # perform grouping
        grouped, scores = self.parser.parse(aggregated_heatmaps,
                                            aggregated_tags,
                                            self.test_cfg['adjust'],
                                            self.test_cfg['refine'])

        preds = get_group_preds(
            grouped,
            center,
            scale, [aggregated_heatmaps.size(3),
                    aggregated_heatmaps.size(2)],
            use_udp=self.use_udp)

        image_paths = []
        image_paths.append(img_metas['image_file'])

        if return_heatmap:
            output_heatmap = aggregated_heatmaps.detach().cpu().numpy()
        else:
            output_heatmap = None

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
