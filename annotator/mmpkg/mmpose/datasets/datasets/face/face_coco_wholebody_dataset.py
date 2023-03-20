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
class FaceCocoWholeBodyDataset(Kpt2dSviewRgbImgTopDownDataset):
    """CocoWholeBodyDataset for face keypoint localization.

    `Whole-Body Human Pose Estimation in the Wild', ECCV'2020.
    More details can be found in the `paper
    <https://arxiv.org/abs/2007.11858>`__ .

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    The face landmark annotations follow the 68 points mark-up.

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
            cfg = Config.fromfile('configs/_base_/datasets/'
                                  'coco_wholebody_face.py')
            dataset_info = cfg._cfg_dict['dataset_info']

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
                if obj['face_valid'] and max(obj['face_kpts']) > 0:
                    joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
                    joints_3d_visible = np.zeros((num_joints, 3),
                                                 dtype=np.float32)

                    keypoints = np.array(obj['face_kpts']).reshape(-1, 3)
                    joints_3d[:, :2] = keypoints[:, :2]
                    joints_3d_visible[:, :2] = np.minimum(1, keypoints[:, 2:3])

                    image_file = osp.join(self.img_prefix,
                                          self.id2name[img_id])
                    gt_db.append({
                        'image_file': image_file,
                        'rotation': 0,
                        'joints_3d': joints_3d,
                        'joints_3d_visible': joints_3d_visible,
                        'dataset': self.dataset_name,
                        'bbox': obj['face_box'],
                        'bbox_score': 1,
                        'bbox_id': bbox_id
                    })
                    bbox_id = bbox_id + 1
        gt_db = sorted(gt_db, key=lambda x: x['bbox_id'])

        return gt_db

    def _get_normalize_factor(self, gts, *args, **kwargs):
        """Get inter-ocular distance as the normalize factor, measured as the
        Euclidean distance between the outer corners of the eyes.

        Args:
            gts (np.ndarray[N, K, 2]): Groundtruth keypoint location.

        Returns:
            np.ndarray[N, 2]: normalized factor
        """

        interocular = np.linalg.norm(
            gts[:, 36, :] - gts[:, 45, :], axis=1, keepdims=True)
        return np.tile(interocular, [1, 2])

    @deprecated_api_warning(name_dict=dict(outputs='results'))
    def evaluate(self, results, res_folder=None, metric='NME', **kwargs):
        """Evaluate COCO-WholeBody Face keypoint results. The pose prediction
        results will be saved in ``${res_folder}/result_keypoints.json``.

        Note:
            - batch_size: N
            - num_keypoints: K
            - heatmap height: H
            - heatmap width: W

        Args:
            results (list[dict]): Testing results containing the following
                items:

                - preds (np.ndarray[1,K,3]): The first two dimensions are \
                    coordinates, score is the third dimension of the array.
                - boxes (np.ndarray[1,6]): [center[0], center[1], scale[0], \
                    scale[1],area, score]
                - image_path (list[str]): For example, ['coco/train2017/\
                    000000000009.jpg']
                - output_heatmap (np.ndarray[N, K, H, W]): model outputs.
            res_folder (str, optional): The folder to save the testing
                results. If not specified, a temp folder will be created.
                Default: None.
            metric (str | list[str]): Metric to be performed.
                Options: 'NME'.

        Returns:
            dict: Evaluation results for evaluation metric.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['NME']
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
