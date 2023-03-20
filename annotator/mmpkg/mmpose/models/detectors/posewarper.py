# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import numpy as np
import torch

from ..builder import POSENETS
from .top_down import TopDown

try:
    from annotator.mmpkg.mmcv.runner import auto_fp16
except ImportError:
    warnings.warn('auto_fp16 from annotator.mmpkg.mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from annotator.mmpkg.mmpose.core import auto_fp16


@POSENETS.register_module()
class PoseWarper(TopDown):
    """Top-down pose detectors for multi-frame settings for video inputs.

    `"Learning temporal pose estimation from sparsely-labeled videos"
    <https://arxiv.org/abs/1906.04016>`_.

    A child class of TopDown detector. The main difference between PoseWarper
    and TopDown lies in that the former takes a list of tensors as input image
    while the latter takes a single tensor as input image in forward method.

    Args:
        backbone (dict): Backbone modules to extract features.
        neck (dict): intermediate modules to transform features.
        keypoint_head (dict): Keypoint head to process feature.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path to the pretrained models.
        loss_pose (None): Deprecated arguments. Please use
            `loss_keypoint` for heads instead.
        concat_tensors (bool): Whether to concat the tensors on the batch dim,
            which can speed up, Default: True
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 keypoint_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 loss_pose=None,
                 concat_tensors=True):
        super().__init__(
            backbone=backbone,
            neck=neck,
            keypoint_head=keypoint_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            loss_pose=loss_pose)
        self.concat_tensors = concat_tensors

    @auto_fp16(apply_to=('img', ))
    def forward(self,
                img,
                target=None,
                target_weight=None,
                img_metas=None,
                return_loss=True,
                return_heatmap=False,
                **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=True`, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=False`, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.

        Note:
            - number of frames: F
            - batch_size: N
            - num_keypoints: K
            - num_img_channel: C (Default: 3)
            - img height: imgH
            - img width: imgW
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            imgs (list[F,torch.Tensor[N,C,imgH,imgW]]): multiple input frames
            target (torch.Tensor[N,K,H,W]): Target heatmaps for one frame.
            target_weight (torch.Tensor[N,K,1]): Weights across
                different joint types.
            img_metas (list(dict)): Information about data augmentation
                By default this includes:

                - "image_file: paths to multiple video frames
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
            return_loss (bool): Option to `return loss`. `return loss=True`
                for training, `return loss=False` for validation & test.
            return_heatmap (bool) : Option to return heatmap.

        Returns:
            dict|tuple: if `return loss` is true, then return losses. \
                Otherwise, return predicted poses, boxes, image paths \
                and heatmaps.
        """
        if return_loss:
            return self.forward_train(img, target, target_weight, img_metas,
                                      **kwargs)
        return self.forward_test(
            img, img_metas, return_heatmap=return_heatmap, **kwargs)

    def forward_train(self, imgs, target, target_weight, img_metas, **kwargs):
        """Defines the computation performed at every call when training."""
        # imgs (list[Fxtorch.Tensor[NxCximgHximgW]]): multiple input frames
        assert imgs[0].size(0) == len(img_metas)
        num_frames = len(imgs)
        frame_weight = img_metas[0]['frame_weight']

        assert num_frames == len(frame_weight), f'The number of frames ' \
            f'({num_frames}) and the length of weights for each frame ' \
            f'({len(frame_weight)}) must match'

        if self.concat_tensors:
            features = [self.backbone(torch.cat(imgs, 0))]
        else:
            features = [self.backbone(img) for img in imgs]

        if self.with_neck:
            features = self.neck(features, frame_weight=frame_weight)

        if self.with_keypoint:
            output = self.keypoint_head(features)

        # if return loss
        losses = dict()
        if self.with_keypoint:
            keypoint_losses = self.keypoint_head.get_loss(
                output, target, target_weight)
            losses.update(keypoint_losses)
            keypoint_accuracy = self.keypoint_head.get_accuracy(
                output, target, target_weight)
            losses.update(keypoint_accuracy)

        return losses

    def forward_test(self, imgs, img_metas, return_heatmap=False, **kwargs):
        """Defines the computation performed at every call when testing."""
        # imgs (list[Fxtorch.Tensor[NxCximgHximgW]]): multiple input frames
        assert imgs[0].size(0) == len(img_metas)
        num_frames = len(imgs)
        frame_weight = img_metas[0]['frame_weight']

        assert num_frames == len(frame_weight), f'The number of frames ' \
            f'({num_frames}) and the length of weights for each frame ' \
            f'({len(frame_weight)}) must match'

        batch_size, _, img_height, img_width = imgs[0].shape

        if batch_size > 1:
            assert 'bbox_id' in img_metas[0]

        result = {}

        if self.concat_tensors:
            features = [self.backbone(torch.cat(imgs, 0))]
        else:
            features = [self.backbone(img) for img in imgs]

        if self.with_neck:
            features = self.neck(features, frame_weight=frame_weight)

        if self.with_keypoint:
            output_heatmap = self.keypoint_head.inference_model(
                features, flip_pairs=None)

        if self.test_cfg.get('flip_test', True):
            imgs_flipped = [img.flip(3) for img in imgs]

            if self.concat_tensors:
                features_flipped = [self.backbone(torch.cat(imgs_flipped, 0))]
            else:
                features_flipped = [
                    self.backbone(img_flipped) for img_flipped in imgs_flipped
                ]

            if self.with_neck:
                features_flipped = self.neck(
                    features_flipped, frame_weight=frame_weight)

            if self.with_keypoint:
                output_flipped_heatmap = self.keypoint_head.inference_model(
                    features_flipped, img_metas[0]['flip_pairs'])
                output_heatmap = (output_heatmap +
                                  output_flipped_heatmap) * 0.5

        if self.with_keypoint:
            keypoint_result = self.keypoint_head.decode(
                img_metas, output_heatmap, img_size=[img_width, img_height])
            result.update(keypoint_result)

            if not return_heatmap:
                output_heatmap = None

            result['output_heatmap'] = output_heatmap

        return result

    def forward_dummy(self, img):
        """Used for computing network FLOPs.

        See ``tools/get_flops.py``.

        Args:
            img (torch.Tensor[N,C,imgH,imgW], or list|tuple of tensors):
                multiple input frames, N >= 2.

        Returns:
            Tensor: Output heatmaps.
        """
        # concat tensors if they are in a list
        if isinstance(img, (list, tuple)):
            img = torch.cat(img, 0)

        batch_size = img.size(0)
        assert batch_size > 1, 'Input batch size to PoseWarper ' \
            'should be larger than 1.'
        if batch_size == 2:
            warnings.warn('Current batch size: 2, for pytorch2onnx and '
                          'getting flops both.')
        else:
            warnings.warn(
                f'Current batch size: {batch_size}, for getting flops only.')

        frame_weight = np.random.uniform(0, 1, batch_size)
        output = [self.backbone(img)]

        if self.with_neck:
            output = self.neck(output, frame_weight=frame_weight)
        if self.with_keypoint:
            output = self.keypoint_head(output)
        return output
