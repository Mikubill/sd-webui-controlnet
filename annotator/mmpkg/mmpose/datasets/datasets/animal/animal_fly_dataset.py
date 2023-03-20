# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict

import numpy as np
from annotator.mmpkg.mmcv import Config, deprecated_api_warning

from ...builder import DATASETS
from ..base import Kpt2dSviewRgbImgTopDownDataset


@DATASETS.register_module()
class AnimalFlyDataset(Kpt2dSviewRgbImgTopDownDataset):
    """AnimalFlyDataset for animal pose estimation.

    "Fast animal pose estimation using deep neural networks"
    Nature methods'2019. More details can be found in the `paper
    <https://www.biorxiv.org/content/biorxiv/\
    early/2018/05/25/331181.full.pdf>`__ .

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    Vinegar Fly keypoint indexes::

        0: "head",
        1: "eyeL",
        2: "eyeR",
        3: "neck",
        4: "thorax",
        5: "abdomen",
        6: "forelegR1",
        7: "forelegR2",
        8: "forelegR3",
        9: "forelegR4",
        10: "midlegR1",
        11: "midlegR2",
        12: "midlegR3",
        13: "midlegR4",
        14: "hindlegR1",
        15: "hindlegR2",
        16: "hindlegR3",
        17: "hindlegR4",
        18: "forelegL1",
        19: "forelegL2",
        20: "forelegL3",
        21: "forelegL4",
        22: "midlegL1",
        23: "midlegL2",
        24: "midlegL3",
        25: "midlegL4",
        26: "hindlegL1",
        27: "hindlegL2",
        28: "hindlegL3",
        29: "hindlegL4",
        30: "wingL",
        31: "wingR"

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
            cfg = Config.fromfile('configs/_base_/datasets/fly.py')
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
                if max(obj['keypoints']) == 0:
                    continue
                joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
                joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)

                keypoints = np.array(obj['keypoints']).reshape(-1, 3)
                joints_3d[:, :2] = keypoints[:, :2]
                joints_3d_visible[:, :2] = np.minimum(1, keypoints[:, 2:3])

                # use the entire image which is 192x192
                bbox = np.array([0, 0, 192, 192], dtype=np.float32)

                image_file = osp.join(self.img_prefix, self.id2name[img_id])

                gt_db.append({
                    'image_file': image_file,
                    'rotation': 0,
                    'joints_3d': joints_3d,
                    'joints_3d_visible': joints_3d_visible,
                    'dataset': self.dataset_name,
                    'bbox': bbox,
                    'bbox_score': 1,
                    'bbox_id': bbox_id
                })
                bbox_id = bbox_id + 1
        gt_db = sorted(gt_db, key=lambda x: x['bbox_id'])

        return gt_db

    @deprecated_api_warning(name_dict=dict(outputs='results'))
    def evaluate(self, results, res_folder=None, metric='PCK', **kwargs):
        """Evaluate Fly keypoint results. The pose prediction results will be
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
                    scale[1],area, score]
                - image_paths (list[str]): For example, ['Test/source/0.jpg']
                - output_heatmap (np.ndarray[N, K, H, W]): model outputs.

            res_folder (str): Path of directory to save the results.
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
