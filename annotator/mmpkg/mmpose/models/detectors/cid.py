# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import annotator.mmpkg.mmcv as mmcv
import torch
import torch.nn.functional as F
from annotator.mmpkg.mmcv.image import imwrite
from annotator.mmpkg.mmcv.utils.misc import deprecated_api_warning
from annotator.mmpkg.mmcv.visualization.image import imshow

from annotator.mmpkg.mmpose.core.evaluation import get_group_preds
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
class CID(BasePose):
    """Contextual Instance Decouple for Multi-Person Pose Estimation.

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
                multi_heatmap=None,
                multi_mask=None,
                instance_coord=None,
                instance_heatmap=None,
                instance_mask=None,
                instance_valid=None,
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
            multi_heatmap (torch.Tensor[N,C,H,W]): Multi-person heatmaps
            multi_mask (torch.Tensor[N,1,H,W]): Multi-person heatmap mask
            instance_coord (torch.Tensor[N,M,2]): Instance center coord
            instance_heatmap (torch.Tensor[N,M,C,H,W]): Single person
                heatmap for each instance
            instance_mask (torch.Tensor[N,M,C,1,1]): Single person heatmap mask
            instance_valid (torch.Tensor[N,M]): Bool mask to indicate the
                existence of each person
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
            return self.forward_train(img, multi_heatmap, multi_mask,
                                      instance_coord, instance_heatmap,
                                      instance_mask, instance_valid, img_metas,
                                      **kwargs)
        return self.forward_test(
            img, img_metas, return_heatmap=return_heatmap, **kwargs)

    def forward_train(self, img, multi_heatmap, multi_mask, instance_coord,
                      instance_heatmap, instance_mask, instance_valid,
                      img_metas, **kwargs):
        """Forward CID model and calculate the loss.

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
            multi_heatmap (torch.Tensor[N,C,H,W]): Multi-person heatmaps
            multi_mask (torch.Tensor[N,1,H,W]): Multi-person heatmap mask
            instance_coord (torch.Tensor[N,M,2]): Instance center coord
            instance_heatmap (torch.Tensor[N,M,C,H,W]): Single person heatmap
                for each instance
            instance_mask (torch.Tensor[N,M,C,1,1]): Single person heatmap mask
            instance_valid (torch.Tensor[N,M]): Bool mask to indicate
                the existence of each person
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

        labels = (multi_heatmap, multi_mask, instance_coord, instance_heatmap,
                  instance_mask, instance_valid)

        losses = dict()
        if self.with_keypoint:
            cid_losses = self.keypoint_head(output, labels)
            losses.update(cid_losses)

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
            output = self.keypoint_head(output, self.test_cfg)
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

        base_size = img_metas['base_size']
        center = img_metas['center']
        scale = img_metas['scale']
        self.test_cfg['flip_index'] = img_metas['flip_index']

        result = {}

        image_resized = aug_data[0].to(img.device)
        if self.test_cfg.get('flip_test', True):
            image_flipped = torch.flip(image_resized, [3])
            image_resized = torch.cat((image_resized, image_flipped), dim=0)
        features = self.backbone(image_resized)
        instance_heatmaps, instance_scores = self.keypoint_head(
            features, self.test_cfg)

        if len(instance_heatmaps) > 0:
            # detect person with pose
            num_people, num_keypoints, h, w = instance_heatmaps.size()
            center_pool_kernel = self.test_cfg.get('center_pool_kernel', 3)
            center_pool = F.avg_pool2d(instance_heatmaps, center_pool_kernel,
                                       1, (center_pool_kernel - 1) // 2)
            instance_heatmaps = (instance_heatmaps + center_pool) / 2.0
            nms_instance_heatmaps = instance_heatmaps.view(
                num_people, num_keypoints, -1)
            vals, inds = torch.max(nms_instance_heatmaps, dim=2)
            x = inds % w
            y = inds // w
            # shift coords by 0.25
            x, y = self.adjust(x, y, instance_heatmaps)

            vals = vals * instance_scores.unsqueeze(1)
            poses = torch.stack((x, y, vals), dim=2)

            poses[:, :, :2] = poses[:, :, :2] * 4 + 2
            scores = torch.mean(poses[:, :, 2], dim=1)
            # add tag dim to match AE eval
            poses = torch.cat((poses,
                               torch.ones((poses.size(0), poses.size(1), 1),
                                          dtype=poses.dtype,
                                          device=poses.device)),
                              dim=2)
            poses = poses.cpu().numpy()
            scores = scores.cpu().numpy()
            poses = get_group_preds([poses], center, scale,
                                    [base_size[0], base_size[1]])
        else:
            poses, scores = [], []

        image_paths = []
        image_paths.append(img_metas['image_file'])

        result['preds'] = poses
        result['scores'] = scores
        result['image_paths'] = image_paths
        result['output_heatmap'] = None

        return result

    def adjust(self, res_x, res_y, heatmaps):
        n, k, h, w = heatmaps.size()

        x_l, x_r = (res_x - 1).clamp(min=0), (res_x + 1).clamp(max=w - 1)
        y_t, y_b = (res_y + 1).clamp(max=h - 1), (res_y - 1).clamp(min=0)
        n_inds = torch.arange(n)[:, None].to(heatmaps.device)
        k_inds = torch.arange(k)[None].to(heatmaps.device)

        px = torch.sign(heatmaps[n_inds, k_inds, res_y, x_r] -
                        heatmaps[n_inds, k_inds, res_y, x_l]) * 0.25
        py = torch.sign(heatmaps[n_inds, k_inds, y_t, res_x] -
                        heatmaps[n_inds, k_inds, y_b, res_x]) * 0.25

        res_x, res_y = res_x.float(), res_y.float()
        x_l, x_r = x_l.float(), x_r.float()
        y_b, y_t = y_b.float(), y_t.float()
        px = px * torch.sign(res_x - x_l) * torch.sign(x_r - res_x)
        py = py * torch.sign(res_y - y_b) * torch.sign(y_t - res_y)

        res_x = res_x.float() + px
        res_y = res_y.float() + py

        return res_x, res_y

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
