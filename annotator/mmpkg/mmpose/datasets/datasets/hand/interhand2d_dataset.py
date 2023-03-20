# Copyright (c) OpenMMLab. All rights reserved.
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
class InterHand2DDataset(Kpt2dSviewRgbImgTopDownDataset):
    """InterHand2.6M 2D dataset for top-down hand pose estimation.

    "InterHand2.6M: A Dataset and Baseline for 3D Interacting Hand Pose
    Estimation from a Single RGB Image", ECCV'2020.
    More details can be found in the `paper
    <https://arxiv.org/pdf/2008.09309.pdf>`__ .

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    InterHand2.6M keypoint indexes::

        0: 'thumb4',
        1: 'thumb3',
        2: 'thumb2',
        3: 'thumb1',
        4: 'forefinger4',
        5: 'forefinger3',
        6: 'forefinger2',
        7: 'forefinger1',
        8: 'middle_finger4',
        9: 'middle_finger3',
        10: 'middle_finger2',
        11: 'middle_finger1',
        12: 'ring_finger4',
        13: 'ring_finger3',
        14: 'ring_finger2',
        15: 'ring_finger1',
        16: 'pinky_finger4',
        17: 'pinky_finger3',
        18: 'pinky_finger2',
        19: 'pinky_finger1',
        20: 'wrist'

    Args:
        ann_file (str): Path to the annotation file.
        camera_file (str): Path to the camera file.
        joint_file (str): Path to the joint file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        dataset_info (DatasetInfo): A class containing all dataset info.
        test_mode (str): Store True when building test or
            validation dataset. Default: False.
    """

    def __init__(self,
                 ann_file,
                 camera_file,
                 joint_file,
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
            cfg = Config.fromfile('configs/_base_/datasets/interhand2d.py')
            dataset_info = cfg._cfg_dict['dataset_info']

        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            test_mode=test_mode)

        self.ann_info['use_different_joint_weights'] = False
        self.camera_file = camera_file
        self.joint_file = joint_file
        self.db = self._get_db()

        print(f'=> num_images: {self.num_images}')
        print(f'=> load {len(self.db)} samples')

    @staticmethod
    def _cam2pixel(cam_coord, f, c):
        """Transform the joints from their camera coordinates to their pixel
        coordinates.

        Note:
            - N: number of joints

        Args:
            cam_coord (ndarray[N, 3]): 3D joints coordinates
                in the camera coordinate system
            f (ndarray[2]): focal length of x and y axis
            c (ndarray[2]): principal point of x and y axis

        Returns:
            img_coord (ndarray[N, 3]): the coordinates (x, y, 0)
                in the image plane.
        """
        x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * f[0] + c[0]
        y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * f[1] + c[1]
        z = np.zeros_like(x)
        img_coord = np.concatenate((x[:, None], y[:, None], z[:, None]), 1)
        return img_coord

    @staticmethod
    def _world2cam(world_coord, R, T):
        """Transform the joints from their world coordinates to their camera
        coordinates.

        Note:
            - N: number of joints

        Args:
            world_coord (ndarray[3, N]): 3D joints coordinates
                in the world coordinate system
            R (ndarray[3, 3]): camera rotation matrix
            T (ndarray[3]): camera position (x, y, z)

        Returns:
            cam_coord (ndarray[3, N]): 3D joints coordinates
                in the camera coordinate system
        """
        cam_coord = np.dot(R, world_coord - T)
        return cam_coord

    def _get_db(self):
        """Load dataset.

        Adapted from 'https://github.com/facebookresearch/InterHand2.6M/'
            'blob/master/data/InterHand2.6M/dataset.py'
        Copyright (c) FaceBook Research, under CC-BY-NC 4.0 license.
        """
        with open(self.camera_file, 'r') as f:
            cameras = json.load(f)
        with open(self.joint_file, 'r') as f:
            joints = json.load(f)
        gt_db = []
        bbox_id = 0
        for img_id in self.img_ids:
            num_joints = self.ann_info['num_joints']

            ann_id = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
            ann = self.coco.loadAnns(ann_id)[0]
            img = self.coco.loadImgs(img_id)[0]

            capture_id = str(img['capture'])
            camera_name = img['camera']
            frame_idx = str(img['frame_idx'])
            image_file = osp.join(self.img_prefix, self.id2name[img_id])

            camera_pos, camera_rot = np.array(
                cameras[capture_id]['campos'][camera_name],
                dtype=np.float32), np.array(
                    cameras[capture_id]['camrot'][camera_name],
                    dtype=np.float32)
            focal, principal_pt = np.array(
                cameras[capture_id]['focal'][camera_name],
                dtype=np.float32), np.array(
                    cameras[capture_id]['princpt'][camera_name],
                    dtype=np.float32)
            joint_world = np.array(
                joints[capture_id][frame_idx]['world_coord'], dtype=np.float32)
            joint_cam = self._world2cam(
                joint_world.transpose(1, 0), camera_rot,
                camera_pos.reshape(3, 1)).transpose(1, 0)
            joint_img = self._cam2pixel(joint_cam, focal, principal_pt)[:, :2]
            joint_img = joint_img.reshape(2, -1, 2)

            joint_valid = np.array(
                ann['joint_valid'], dtype=np.float32).reshape(2, -1)
            # if root is not valid -> root-relative 3D pose is also not valid.
            # Therefore, mark all joints as invalid
            for hand in range(2):
                joint_valid[hand, :] *= joint_valid[hand][-1]

                if np.sum(joint_valid[hand, :]) > 11:
                    joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
                    joints_3d_visible = np.zeros((num_joints, 3),
                                                 dtype=np.float32)
                    joints_3d[:, :2] = joint_img[hand, :, :]
                    joints_3d_visible[:, :2] = np.minimum(
                        1, joint_valid[hand, :].reshape(-1, 1))

                    # use the tightest bbox enclosing all keypoints as bbox
                    bbox = [img['width'], img['height'], 0, 0]
                    for i in range(num_joints):
                        if joints_3d_visible[i][0]:
                            bbox[0] = min(bbox[0], joints_3d[i][0])
                            bbox[1] = min(bbox[1], joints_3d[i][1])
                            bbox[2] = max(bbox[2], joints_3d[i][0])
                            bbox[3] = max(bbox[3], joints_3d[i][1])

                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]

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
        """Evaluate interhand2d keypoint results. The pose prediction results
        will be saved in ``${res_folder}/result_keypoints.json``.

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
                - image_paths (list[str]): For example, ['Capture12/\
                    0390_dh_touchROM/cam410209/image62434.jpg']
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
