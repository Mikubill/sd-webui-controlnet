# Copyright (c) OpenMMLab. All rights reserved.
import annotator.mmpkg.mmcv as mmcv
import numpy as np
from annotator.mmpkg.mmcv.utils.misc import deprecated_api_warning

from annotator.mmpkg.mmpose.core import imshow_keypoints, imshow_keypoints_3d
from ..builder import POSENETS
from .top_down import TopDown


@POSENETS.register_module()
class Interhand3D(TopDown):
    """Top-down interhand 3D pose detector of paper ref: Gyeongsik Moon.

    "InterHand2.6M: A Dataset and Baseline for 3D Interacting Hand Pose
    Estimation from a Single RGB Image". A child class of TopDown detector.
    """

    def forward(self,
                img,
                target=None,
                target_weight=None,
                img_metas=None,
                return_loss=True,
                **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=True`, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=False`, img and img_meta
        should be double nested (i.e.  list[Tensor], list[list[dict]]), with
        the outer list indicating test time augmentations.

        Note:
            - batch_size: N
            - num_keypoints: K
            - num_img_channel: C (Default: 3)
            - img height: imgH
            - img width: imgW
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            img (torch.Tensor[NxCximgHximgW]): Input images.
            target (list[torch.Tensor]): Target heatmaps, relative hand
            root depth and hand type.
            target_weight (list[torch.Tensor]): Weights for target
            heatmaps, relative hand root depth and hand type.
            img_metas (list(dict)): Information about data augmentation
                By default this includes:

                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
                - "heatmap3d_depth_bound": depth bound of hand keypoint 3D
                    heatmap
                - "root_depth_bound": depth bound of relative root depth 1D
                    heatmap
            return_loss (bool): Option to `return loss`. `return loss=True`
                for training, `return loss=False` for validation & test.

        Returns:
            dict|tuple: if `return loss` is true, then return losses. \
                Otherwise, return predicted poses, boxes, image paths, \
                heatmaps, relative hand root depth and hand type.
        """
        if return_loss:
            return self.forward_train(img, target, target_weight, img_metas,
                                      **kwargs)
        return self.forward_test(img, img_metas, **kwargs)

    def forward_test(self, img, img_metas, **kwargs):
        """Defines the computation performed at every call when testing."""
        assert img.size(0) == len(img_metas)
        batch_size, _, img_height, img_width = img.shape
        if batch_size > 1:
            assert 'bbox_id' in img_metas[0]

        features = self.backbone(img)
        if self.with_neck:
            features = self.neck(features)
        if self.with_keypoint:
            output = self.keypoint_head.inference_model(
                features, flip_pairs=None)

        if self.test_cfg.get('flip_test', True):
            img_flipped = img.flip(3)
            features_flipped = self.backbone(img_flipped)
            if self.with_neck:
                features_flipped = self.neck(features_flipped)
            if self.with_keypoint:
                output_flipped = self.keypoint_head.inference_model(
                    features_flipped, img_metas[0]['flip_pairs'])
                output = [(out + out_flipped) * 0.5
                          for out, out_flipped in zip(output, output_flipped)]

        if self.with_keypoint:
            result = self.keypoint_head.decode(
                img_metas, output, img_size=[img_width, img_height])
        else:
            result = {}
        return result

    @deprecated_api_warning({'pose_limb_color': 'pose_link_color'},
                            cls_name='Interhand3D')
    def show_result(self,
                    result,
                    img=None,
                    skeleton=None,
                    kpt_score_thr=0.3,
                    radius=8,
                    bbox_color='green',
                    thickness=2,
                    pose_kpt_color=None,
                    pose_link_color=None,
                    vis_height=400,
                    num_instances=-1,
                    axis_azimuth=-115,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Visualize 3D pose estimation results.

        Args:
            result (list[dict]): The pose estimation results containing:

                - "keypoints_3d" ([K,4]): 3D keypoints
                - "keypoints" ([K,3] or [T,K,3]): Optional for visualizing
                    2D inputs. If a sequence is given, only the last frame
                    will be used for visualization
                - "bbox" ([4,] or [T,4]): Optional for visualizing 2D inputs
                - "title" (str): title for the subplot
            img (str or Tensor): Optional. The image to visualize 2D inputs on.
            skeleton (list of [idx_i,idx_j]): Skeleton described by a list of
                links, each is a pair of joint indices.
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            radius (int): Radius of circles.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            thickness (int): Thickness of lines.
            pose_kpt_color (np.array[Nx3]`): Color of N keypoints.
                If None, do not draw keypoints.
            pose_link_color (np.array[Mx3]): Color of M limbs.
                If None, do not draw limbs.
            vis_height (int): The image height of the visualization. The width
                will be N*vis_height depending on the number of visualized
                items.
            num_instances (int): Number of instances to be shown in 3D. If
                smaller than 0, all the instances in the pose_result will be
                shown. Otherwise, pad or truncate the pose_result to a length
                of num_instances.
            axis_azimuth (float): axis azimuth angle for 3D visualizations.
            win_name (str): The window name.
            show (bool): Whether to show the image. Default: False.
            wait_time (int): Value of waitKey param.
                Default: 0.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            Tensor: Visualized img, only if not `show` or `out_file`.
        """
        if num_instances < 0:
            assert len(result) > 0
        result = sorted(result, key=lambda x: x.get('track_id', 0))

        # draw image and 2d poses
        if img is not None:
            img = mmcv.imread(img)

            bbox_result = []
            pose_2d = []
            for res in result:
                if 'bbox' in res:
                    bbox = np.array(res['bbox'])
                    if bbox.ndim != 1:
                        assert bbox.ndim == 2
                        bbox = bbox[-1]  # Get bbox from the last frame
                    bbox_result.append(bbox)
                if 'keypoints' in res:
                    kpts = np.array(res['keypoints'])
                    if kpts.ndim != 2:
                        assert kpts.ndim == 3
                        kpts = kpts[-1]  # Get 2D keypoints from the last frame
                    pose_2d.append(kpts)

            if len(bbox_result) > 0:
                bboxes = np.vstack(bbox_result)
                mmcv.imshow_bboxes(
                    img,
                    bboxes,
                    colors=bbox_color,
                    top_k=-1,
                    thickness=2,
                    show=False)
            if len(pose_2d) > 0:
                imshow_keypoints(
                    img,
                    pose_2d,
                    skeleton,
                    kpt_score_thr=kpt_score_thr,
                    pose_kpt_color=pose_kpt_color,
                    pose_link_color=pose_link_color,
                    radius=radius,
                    thickness=thickness)
            img = mmcv.imrescale(img, scale=vis_height / img.shape[0])

        img_vis = imshow_keypoints_3d(
            result,
            img,
            skeleton,
            pose_kpt_color,
            pose_link_color,
            vis_height,
            axis_limit=300,
            axis_azimuth=axis_azimuth,
            axis_elev=15,
            kpt_score_thr=kpt_score_thr,
            num_instances=num_instances)

        if show:
            mmcv.visualization.imshow(img_vis, win_name, wait_time)

        if out_file is not None:
            mmcv.imwrite(img_vis, out_file)

        return img_vis
