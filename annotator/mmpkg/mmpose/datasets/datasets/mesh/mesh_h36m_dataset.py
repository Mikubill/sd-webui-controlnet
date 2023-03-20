# Copyright (c) OpenMMLab. All rights reserved.
import os
from collections import OrderedDict

import json_tricks as json
import numpy as np

from annotator.mmpkg.mmpose.core.evaluation import keypoint_mpjpe
from annotator.mmpkg.mmpose.datasets.builder import DATASETS
from .mesh_base_dataset import MeshBaseDataset


@DATASETS.register_module()
class MeshH36MDataset(MeshBaseDataset):
    """Human3.6M Dataset for 3D human mesh estimation. It inherits all function
    from MeshBaseDataset and has its own evaluate function.

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    def evaluate(self, outputs, res_folder, metric='joint_error', logger=None):
        """Evaluate 3D keypoint results."""
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['joint_error']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        res_file = os.path.join(res_folder, 'result_keypoints.json')
        kpts = []
        for out in outputs:
            for (keypoints, image_path) in zip(out['keypoints_3d'],
                                               out['image_path']):
                kpts.append({
                    'keypoints': keypoints.tolist(),
                    'image': image_path,
                })

        self._write_keypoint_results(kpts, res_file)
        info_str = self._report_metric(res_file)
        name_value = OrderedDict(info_str)
        return name_value

    @staticmethod
    def _write_keypoint_results(keypoints, res_file):
        """Write results into a json file."""

        with open(res_file, 'w') as f:
            json.dump(keypoints, f, sort_keys=True, indent=4)

    def _report_metric(self, res_file):
        """Keypoint evaluation.

        Report mean per joint position error (MPJPE) and mean per joint
        position error after rigid alignment (MPJPE-PA)
        """

        with open(res_file, 'r') as fin:
            preds = json.load(fin)
        assert len(preds) == len(self.db)

        pred_joints_3d = [pred['keypoints'] for pred in preds]
        gt_joints_3d = [item['joints_3d'] for item in self.db]
        gt_joints_visible = [item['joints_3d_visible'] for item in self.db]

        pred_joints_3d = np.array(pred_joints_3d)
        gt_joints_3d = np.array(gt_joints_3d)
        gt_joints_visible = np.array(gt_joints_visible)

        # we only evaluate on 14 lsp joints
        joint_mapper = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18]
        pred_joints_3d = pred_joints_3d[:, joint_mapper, :]
        pred_pelvis = (pred_joints_3d[:, 2] + pred_joints_3d[:, 3]) / 2
        pred_joints_3d = pred_joints_3d - pred_pelvis[:, None, :]

        gt_joints_3d = gt_joints_3d[:, joint_mapper, :]
        gt_pelvis = (gt_joints_3d[:, 2] + gt_joints_3d[:, 3]) / 2
        gt_joints_3d = gt_joints_3d - gt_pelvis[:, None, :]
        gt_joints_visible = gt_joints_visible[:, joint_mapper, 0] > 0

        mpjpe = keypoint_mpjpe(pred_joints_3d, gt_joints_3d, gt_joints_visible)
        mpjpe_pa = keypoint_mpjpe(
            pred_joints_3d,
            gt_joints_3d,
            gt_joints_visible,
            alignment='procrustes')

        info_str = []
        info_str.append(('MPJPE', mpjpe * 1000))
        info_str.append(('MPJPE-PA', mpjpe_pa * 1000))
        return info_str
