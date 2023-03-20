# Copyright (c) OpenMMLab. All rights reserved.
import copy as cp
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict

import json_tricks as json
import numpy as np
from annotator.mmpkg.mmcv import Config, deprecated_api_warning

from annotator.mmpkg.mmpose.datasets.builder import DATASETS
from ..base import Kpt2dSviewRgbImgTopDownDataset


@DATASETS.register_module()
class TopDownMpiiTrbDataset(Kpt2dSviewRgbImgTopDownDataset):
    """MPII-TRB Dataset dataset for top-down pose estimation.

    "TRB: A Novel Triplet Representation for Understanding 2D Human Body",
    ICCV'2019. More details can be found in the `paper
    <https://arxiv.org/abs/1910.11535>`__ .

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    MPII-TRB keypoint indexes::

        0: 'left_shoulder'
        1: 'right_shoulder'
        2: 'left_elbow'
        3: 'right_elbow'
        4: 'left_wrist'
        5: 'right_wrist'
        6: 'left_hip'
        7: 'right_hip'
        8: 'left_knee'
        9: 'right_knee'
        10: 'left_ankle'
        11: 'right_ankle'
        12: 'head'
        13: 'neck'

        14: 'right_neck'
        15: 'left_neck'
        16: 'medial_right_shoulder'
        17: 'lateral_right_shoulder'
        18: 'medial_right_bow'
        19: 'lateral_right_bow'
        20: 'medial_right_wrist'
        21: 'lateral_right_wrist'
        22: 'medial_left_shoulder'
        23: 'lateral_left_shoulder'
        24: 'medial_left_bow'
        25: 'lateral_left_bow'
        26: 'medial_left_wrist'
        27: 'lateral_left_wrist'
        28: 'medial_right_hip'
        29: 'lateral_right_hip'
        30: 'medial_right_knee'
        31: 'lateral_right_knee'
        32: 'medial_right_ankle'
        33: 'lateral_right_ankle'
        34: 'medial_left_hip'
        35: 'lateral_left_hip'
        36: 'medial_left_knee'
        37: 'lateral_left_knee'
        38: 'medial_left_ankle'
        39: 'lateral_left_ankle'

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
            cfg = Config.fromfile('configs/_base_/datasets/mpii_trb.py')
            dataset_info = cfg._cfg_dict['dataset_info']

        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            test_mode=test_mode)

        self.db = self._get_db(ann_file)
        self.image_set = set(x['image_file'] for x in self.db)
        self.num_images = len(self.image_set)

        print(f'=> num_images: {self.num_images}')
        print(f'=> load {len(self.db)} samples')

    def _get_db(self, ann_file):
        """Load dataset."""
        with open(ann_file, 'r') as f:
            data = json.load(f)
        tmpl = dict(
            image_file=None,
            bbox_id=None,
            center=None,
            scale=None,
            rotation=0,
            joints_3d=None,
            joints_3d_visible=None,
            dataset=self.dataset_name)

        imid2info = {
            int(osp.splitext(x['file_name'])[0]): x
            for x in data['images']
        }

        num_joints = self.ann_info['num_joints']
        gt_db = []

        for anno in data['annotations']:
            newitem = cp.deepcopy(tmpl)
            image_id = anno['image_id']
            newitem['bbox_id'] = anno['id']
            newitem['image_file'] = osp.join(self.img_prefix,
                                             imid2info[image_id]['file_name'])

            if max(anno['keypoints']) == 0:
                continue

            joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
            joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)

            for ipt in range(num_joints):
                joints_3d[ipt, 0] = anno['keypoints'][ipt * 3 + 0]
                joints_3d[ipt, 1] = anno['keypoints'][ipt * 3 + 1]
                joints_3d[ipt, 2] = 0
                t_vis = min(anno['keypoints'][ipt * 3 + 2], 1)
                joints_3d_visible[ipt, :] = (t_vis, t_vis, 0)

            center = np.array(anno['center'], dtype=np.float32)
            scale = self.ann_info['image_size'] / anno['scale'] / 200.0
            newitem['center'] = center
            newitem['scale'] = scale
            newitem['joints_3d'] = joints_3d
            newitem['joints_3d_visible'] = joints_3d_visible
            if 'headbox' in anno:
                newitem['headbox'] = anno['headbox']
            gt_db.append(newitem)
        gt_db = sorted(gt_db, key=lambda x: x['bbox_id'])

        return gt_db

    def _evaluate_kernel(self, pred, joints_3d, joints_3d_visible, headbox):
        """Evaluate one example."""
        num_joints = self.ann_info['num_joints']
        headbox = np.array(headbox)
        threshold = np.linalg.norm(headbox[:2] - headbox[2:]) * 0.3
        hit = np.zeros(num_joints, dtype=np.float32)
        exist = np.zeros(num_joints, dtype=np.float32)

        for i in range(num_joints):
            pred_pt = pred[i]
            gt_pt = joints_3d[i]
            vis = joints_3d_visible[i][0]
            if vis:
                exist[i] = 1
            else:
                continue
            distance = np.linalg.norm(pred_pt[:2] - gt_pt[:2])
            if distance < threshold:
                hit[i] = 1
        return hit, exist

    @deprecated_api_warning(name_dict=dict(outputs='results'))
    def evaluate(self, results, res_folder=None, metric='PCKh', **kwargs):
        """Evaluate PCKh for MPII-TRB dataset.

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
                - image_paths (list[str]): For example, ['/val2017/\
                    000000397133.jpg']
                - heatmap (np.ndarray[N, K, H, W]): model output heatmap.
                - bbox_ids (list[str]): For example, ['27407'].
            res_folder (str, optional): The folder to save the testing
                results. If not specified, a temp folder will be created.
                Default: None.
            metric (str | list[str]): Metrics to be performed.
                Defaults: 'PCKh'.

        Returns:
            dict: PCKh for each joint
        """
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['PCKh']
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
                str_image_path = image_paths[i]
                image_id = int(osp.basename(osp.splitext(str_image_path)[0]))

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
        info_str = self._report_metric(res_file)
        name_value = OrderedDict(info_str)

        if tmp_folder is not None:
            tmp_folder.cleanup()

        return name_value

    @staticmethod
    def _write_keypoint_results(keypoints, res_file):
        """Write results into a json file."""

        with open(res_file, 'w') as f:
            json.dump(keypoints, f, sort_keys=True, indent=4)

    def _report_metric(self, res_file):
        """Keypoint evaluation.

        Report Mean Acc of skeleton, contour and all joints.
        """
        num_joints = self.ann_info['num_joints']
        hit = np.zeros(num_joints, dtype=np.float32)
        exist = np.zeros(num_joints, dtype=np.float32)

        with open(res_file, 'r') as fin:
            preds = json.load(fin)

        assert len(preds) == len(
            self.db), f'len(preds)={len(preds)}, len(self.db)={len(self.db)}'
        for pred, item in zip(preds, self.db):
            h, e = self._evaluate_kernel(pred['keypoints'], item['joints_3d'],
                                         item['joints_3d_visible'],
                                         item['headbox'])
            hit += h
            exist += e
        skeleton = np.sum(hit[:14]) / np.sum(exist[:14])
        contour = np.sum(hit[14:]) / np.sum(exist[14:])
        mean = np.sum(hit) / np.sum(exist)

        info_str = []
        info_str.append(('Skeleton_acc', skeleton.item()))
        info_str.append(('Contour_acc', contour.item()))
        info_str.append(('PCKh', mean.item()))
        return info_str

    def _sort_and_unique_bboxes(self, kpts, key='bbox_id'):
        """sort kpts and remove the repeated ones."""
        kpts = sorted(kpts, key=lambda x: x[key])
        num = len(kpts)
        for i in range(num - 1, 0, -1):
            if kpts[i][key] == kpts[i - 1][key]:
                del kpts[i]

        return kpts
