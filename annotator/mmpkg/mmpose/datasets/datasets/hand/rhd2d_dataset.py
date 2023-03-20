# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict

import numpy as np
from annotator.mmpkg.mmcv import Config, deprecated_api_warning

from annotator.mmpkg.mmpose.datasets.builder import DATASETS
from ..base import Kpt2dSviewRgbImgTopDownDataset


@DATASETS.register_module()
class Rhd2DDataset(Kpt2dSviewRgbImgTopDownDataset):
    """Rendered Handpose Dataset for top-down hand pose estimation.

    "Learning to Estimate 3D Hand Pose from Single RGB Images",
    ICCV'2017.
    More details can be found in the `paper
    <https://arxiv.org/pdf/1705.01389.pdf>`__ .

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    Rhd keypoint indexes::

        0: 'wrist',
        1: 'thumb4',
        2: 'thumb3',
        3: 'thumb2',
        4: 'thumb1',
        5: 'forefinger4',
        6: 'forefinger3',
        7: 'forefinger2',
        8: 'forefinger1',
        9: 'middle_finger4',
        10: 'middle_finger3',
        11: 'middle_finger2',
        12: 'middle_finger1',
        13: 'ring_finger4',
        14: 'ring_finger3',
        15: 'ring_finger2',
        16: 'ring_finger1',
        17: 'pinky_finger4',
        18: 'pinky_finger3',
        19: 'pinky_finger2',
        20: 'pinky_finger1'

    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        dataset_info (DatasetInfo): A class containing all dataset info.
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 dataset_info=None,
                 test_mode=False):

        if dataset_info is None:
            warnings.warn(
                'dataset_info is missing. '
                'Check https://github.com/open-mmlab/mmpose/pull/663 '
                'for details.', DeprecationWarning)
            cfg = Config.fromfile('configs/_base_/datasets/rhd2d.py')
            dataset_info = cfg._cfg_dict['dataset_info']
        warnings.warn(
            'Please note that the in RHD dataset, its keypoint indices are'
            'different from other hand datasets like COCO-WholeBody-Hand,'
            'FreiHand, CMU Panoptic HandDB, and OneHand10K. You can check'
            '`configs/_base_/datasets/rhd2d.py` for details. If you want to '
            'combine RHD with other hand datasets to train a single model, '
            'please reorder the keypoint indices accordingly.')

        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            test_mode=test_mode)

        self.ann_info['use_different_joint_weights'] = False
        self.db = self._get_db()

        print(f'=> num_images: {self.num_images}')
        print(f'=> load {len(self.db)} samples')

    def _get_db(self):
        """Load dataset."""
        gt_db = []
        bbox_id = 0
        num_joints = self.ann_info['num_joints']
        for img_id in self.img_ids:

            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
            objs = self.coco.loadAnns(ann_ids)

            for obj in objs:
                if max(obj['keypoints']) == 0:
                    continue
                joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
                joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)

                keypoints = np.array(obj['keypoints']).reshape(-1, 3)
                joints_3d[:, :2] = keypoints[:, :2]
                joints_3d_visible[:, :2] = np.minimum(1, keypoints[:, 2:3])

                image_file = osp.join(self.img_prefix, self.id2name[img_id])
                gt_db.append({
                    'image_file': image_file,
                    'rotation': 0,
                    'joints_3d': joints_3d,
                    'joints_3d_visible': joints_3d_visible,
                    'dataset': self.dataset_name,
                    'bbox': obj['bbox'],
                    'bbox_score': 1,
                    'bbox_id': bbox_id
                })
                bbox_id = bbox_id + 1
        gt_db = sorted(gt_db, key=lambda x: x['bbox_id'])

        return gt_db

    @deprecated_api_warning(name_dict=dict(outputs='results'))
    def evaluate(self, results, res_folder=None, metric='PCK', **kwargs):
        """Evaluate rhd keypoint results. The pose prediction results will be
        saved in ``${res_folder}/result_keypoints.json``.

        Note:
            - batch_size: N
            - num_keypoints: K
            - heatmap height: H
            - heatmap width: W

        Args:
            results (list[dict]): Testing results containing the following
                items:

                - preds (np.ndarray[N,K,3]): The first two dimensions are \
                    coordinates, score is the third dimension of the array.
                - boxes (np.ndarray[N,6]): [center[0], center[1], scale[0], \
                    scale[1], area, score]
                - image_paths (list[str]): For example,
                    ['training/rgb/00031426.jpg']
                - output_heatmap (np.ndarray[N, K, H, W]): model outputs.
            res_folder (str, optional): The folder to save the testing
                results. If not specified, a temp folder will be created.
                Default: None.
            metric (str | list[str]): Metric to be performed.
                Options: 'PCK', 'AUC', 'EPE'.

        Returns:
            dict: Evaluation results for evaluation metric.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['PCK', 'AUC', 'EPE']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        if res_folder is not None:
            tmp_folder = None
            res_file = osp.join(res_folder, 'result_keypoints.json')
        else:
            tmp_folder = tempfile.TemporaryDirectory()
            res_file = osp.join(tmp_folder.name, 'result_keypoints.json')

        kpts = []
        for result in results:
            preds = result['preds']
            boxes = result['boxes']
            image_paths = result['image_paths']
            bbox_ids = result['bbox_ids']

            batch_size = len(image_paths)
            for i in range(batch_size):
                image_id = self.name2id[image_paths[i][len(self.img_prefix):]]

                kpts.append({
                    'keypoints': preds[i].tolist(),
                    'center': boxes[i][0:2].tolist(),
                    'scale': boxes[i][2:4].tolist(),
                    'area': float(boxes[i][4]),
                    'score': float(boxes[i][5]),
                    'image_id': image_id,
                    'bbox_id': bbox_ids[i]
                })
        kpts = self._sort_and_unique_bboxes(kpts)

        self._write_keypoint_results(kpts, res_file)
        info_str = self._report_metric(res_file, metrics)
        name_value = OrderedDict(info_str)

        if tmp_folder is not None:
            tmp_folder.cleanup()

        return name_value
