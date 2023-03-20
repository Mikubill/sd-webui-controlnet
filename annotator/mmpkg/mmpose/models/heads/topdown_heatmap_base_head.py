# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

import numpy as np
import torch.nn as nn

from annotator.mmpkg.mmpose.core.evaluation.top_down_eval import keypoints_from_heatmaps


class TopdownHeatmapBaseHead(nn.Module):
    """Base class for top-down heatmap heads.

    All top-down heatmap heads should subclass it.
    All subclass should overwrite:

    Methods:`get_loss`, supporting to calculate loss.
    Methods:`get_accuracy`, supporting to calculate accuracy.
    Methods:`forward`, supporting to forward model.
    Methods:`inference_model`, supporting to inference model.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def get_loss(self, **kwargs):
        """Gets the loss."""

    @abstractmethod
    def get_accuracy(self, **kwargs):
        """Gets the accuracy."""

    @abstractmethod
    def forward(self, **kwargs):
        """Forward function."""

    @abstractmethod
    def inference_model(self, **kwargs):
        """Inference function."""

    def decode(self, img_metas, output, **kwargs):
        """Decode keypoints from heatmaps.

        Args:
            img_metas (list(dict)): Information about data augmentation
                By default this includes:

                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
            output (np.ndarray[N, K, H, W]): model predicted heatmaps.
        """
        batch_size = len(img_metas)

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

        preds, maxvals = keypoints_from_heatmaps(
            output,
            c,
            s,
            unbiased=self.test_cfg.get('unbiased_decoding', False),
            post_process=self.test_cfg.get('post_process', 'default'),
            kernel=self.test_cfg.get('modulate_kernel', 11),
            valid_radius_factor=self.test_cfg.get('valid_radius_factor',
                                                  0.0546875),
            use_udp=self.test_cfg.get('use_udp', False),
            target_type=self.test_cfg.get('target_type', 'GaussianHeatmap'))

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
