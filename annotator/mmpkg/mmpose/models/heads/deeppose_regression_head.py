# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch.nn as nn
from annotator.mmpkg.mmcv.cnn import normal_init

from annotator.mmpkg.mmpose.core.evaluation import (keypoint_pck_accuracy,
                                    keypoints_from_regression)
from annotator.mmpkg.mmpose.core.post_processing import fliplr_regression
from annotator.mmpkg.mmpose.models.builder import HEADS, build_loss


@HEADS.register_module()
class DeepposeRegressionHead(nn.Module):
    """Deeppose regression head with fully connected layers.

    "DeepPose: Human Pose Estimation via Deep Neural Networks".

    Args:
        in_channels (int): Number of input channels
        num_joints (int): Number of joints
        loss_keypoint (dict): Config for keypoint loss. Default: None.
        out_sigma (bool): Predict the sigma (the viriance of the joint
            location) together with the joint location. Default: False
    """

    def __init__(self,
                 in_channels,
                 num_joints,
                 loss_keypoint=None,
                 out_sigma=False,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()

        self.in_channels = in_channels
        self.num_joints = num_joints

        self.loss = build_loss(loss_keypoint)

        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg

        self.out_sigma = out_sigma

        if out_sigma:
            self.fc = nn.Linear(self.in_channels, self.num_joints * 4)
        else:
            self.fc = nn.Linear(self.in_channels, self.num_joints * 2)

    def forward(self, x):
        """Forward function."""
        if isinstance(x, (list, tuple)):
            assert len(x) == 1, ('DeepPoseRegressionHead only supports '
                                 'single-level feature.')
            x = x[0]

        output = self.fc(x)
        N, C = output.shape
        if self.out_sigma:
            return output.reshape([N, C // 4, 4])
        else:
            return output.reshape([N, C // 2, 2])

    def get_loss(self, output, target, target_weight):
        """Calculate top-down keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K

        Args:
            output (torch.Tensor[N, K, 2 or 4]): Output keypoints.
            target (torch.Tensor[N, K, 2]): Target keypoints.
            target_weight (torch.Tensor[N, K, 2]):
                Weights across different joint types.
        """

        losses = dict()
        assert not isinstance(self.loss, nn.Sequential)
        assert target.dim() == 3 and target_weight.dim() == 3

        losses['reg_loss'] = self.loss(output, target, target_weight)

        return losses

    def get_accuracy(self, output, target, target_weight):
        """Calculate accuracy for top-down keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K

        Args:
            output (torch.Tensor[N, K, 2 or 4]): Output keypoints.
            target (torch.Tensor[N, K, 2]): Target keypoints.
            target_weight (torch.Tensor[N, K, 2]):
                Weights across different joint types.
        """

        accuracy = dict()

        N = output.shape[0]
        output = output[..., :2]

        _, avg_acc, cnt = keypoint_pck_accuracy(
            output.detach().cpu().numpy(),
            target.detach().cpu().numpy(),
            target_weight[:, :, 0].detach().cpu().numpy() > 0,
            thr=0.05,
            normalize=np.ones((N, 2), dtype=np.float32))
        accuracy['acc_pose'] = avg_acc

        return accuracy

    def inference_model(self, x, flip_pairs=None):
        """Inference function.

        Returns:
            output_regression (np.ndarray): Output regression.

        Args:
            x (torch.Tensor[N, K, 2]): Input features.
            flip_pairs (None | list[tuple()):
                Pairs of keypoints which are mirrored.
        """
        output = self.forward(x)

        if self.out_sigma:
            output[..., 2:] = output[..., 2:].sigmoid()

        if flip_pairs is not None:
            output_regression = fliplr_regression(
                output.detach().cpu().numpy(), flip_pairs)
        else:
            output_regression = output.detach().cpu().numpy()
        return output_regression

    def decode(self, img_metas, output, **kwargs):
        """Decode the keypoints from output regression.

        Args:
            img_metas (list(dict)): Information about data augmentation
                By default this includes:

                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
            output (np.ndarray[N, K, >=2]): predicted regression vector.
            kwargs: dict contains 'img_size'.
                img_size (tuple(img_width, img_height)): input image size.
        """
        batch_size = len(img_metas)
        sigma = output[..., 2:]
        output = output[..., :2]  # get prediction joint locations

        if 'bbox_id' in img_metas[0]:
            bbox_ids = []
        else:
            bbox_ids = None

        c = np.zeros((batch_size, 2), dtype=np.float32)
        s = np.zeros((batch_size, 2), dtype=np.float32)
        image_paths = []
        score = np.ones(batch_size)
        for i in range(batch_size):
            c[i, :] = img_metas[i]['center']
            s[i, :] = img_metas[i]['scale']
            image_paths.append(img_metas[i]['image_file'])

            if 'bbox_score' in img_metas[i]:
                score[i] = np.array(img_metas[i]['bbox_score']).reshape(-1)
            if bbox_ids is not None:
                bbox_ids.append(img_metas[i]['bbox_id'])

        preds, maxvals = keypoints_from_regression(output, c, s,
                                                   kwargs['img_size'])
        if self.out_sigma:
            maxvals = (1 - sigma).mean(axis=2, keepdims=True)

        all_preds = np.zeros((batch_size, preds.shape[1], 3), dtype=np.float32)
        all_boxes = np.zeros((batch_size, 6), dtype=np.float32)
        all_preds[:, :, 0:2] = preds[:, :, 0:2]
        all_preds[:, :, 2:3] = maxvals
        all_boxes[:, 0:2] = c[:, 0:2]
        all_boxes[:, 2:4] = s[:, 0:2]
        all_boxes[:, 4] = np.prod(s * 200.0, axis=1)
        all_boxes[:, 5] = score

        result = {}

        result['preds'] = all_preds
        result['boxes'] = all_boxes
        result['image_paths'] = image_paths
        result['bbox_ids'] = bbox_ids

        return result

    def init_weights(self):
        normal_init(self.fc, mean=0, std=0.01, bias=0)
