# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import pickle
from abc import ABCMeta, abstractmethod

import json_tricks as json
import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset

from annotator.mmpkg.mmpose.datasets import DatasetInfo
from annotator.mmpkg.mmpose.datasets.pipelines import Compose


class Kpt3dMviewRgbImgDirectDataset(Dataset, metaclass=ABCMeta):
    """Base class for keypoint 3D top-down pose estimation with multi-view RGB
    images as the input.

    All subclasses should overwrite:
        Methods:`_get_db`, 'evaluate'

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

        self.image_info = {}
        self.ann_info = {}

        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.pipeline = pipeline
        self.test_mode = test_mode

        self.ann_info['image_size'] = np.array(data_cfg['image_size'])
        self.ann_info['heatmap_size'] = np.array(data_cfg['heatmap_size'])
        self.ann_info['num_joints'] = data_cfg['num_joints']

        self.ann_info['space_size'] = data_cfg['space_size']
        self.ann_info['space_center'] = data_cfg['space_center']
        self.ann_info['cube_size'] = data_cfg['cube_size']
        self.ann_info['scale_aware_sigma'] = data_cfg.get(
            'scale_aware_sigma', False)

        if dataset_info is None:
            raise ValueError(
                'Check https://github.com/open-mmlab/mmpose/pull/663 '
                'for details.')

        dataset_info = DatasetInfo(dataset_info)

        self.ann_info['flip_pairs'] = dataset_info.flip_pairs
        self.ann_info['num_scales'] = 1
        self.ann_info['flip_index'] = dataset_info.flip_index
        self.ann_info['upper_body_ids'] = dataset_info.upper_body_ids
        self.ann_info['lower_body_ids'] = dataset_info.lower_body_ids
        self.ann_info['joint_weights'] = dataset_info.joint_weights
        self.ann_info['skeleton'] = dataset_info.skeleton
        self.sigmas = dataset_info.sigmas
        self.dataset_name = dataset_info.dataset_name

        self.load_config(data_cfg)

        self.db = []

        self.pipeline = Compose(self.pipeline)

    @abstractmethod
    def _get_db(self):
        """Load dataset."""
        raise NotImplementedError

    def load_config(self, data_cfg):
        """Initialize dataset attributes according to the config.

        Override this method to set dataset specific attributes.
        """
        self.num_joints = data_cfg['num_joints']
        self.num_cameras = data_cfg['num_cameras']
        self.seq_frame_interval = data_cfg.get('seq_frame_interval', 1)
        self.subset = data_cfg.get('subset', 'train')
        self.need_2d_label = data_cfg.get('need_2d_label', False)
        self.need_camera_param = True

    @staticmethod
    def _get_mapping_id_name(imgs):
        """
        Args:
            imgs (dict): dict of image info.

        Returns:
            tuple: Image name & id mapping dicts.

            - id2name (dict): Mapping image id to name.
            - name2id (dict): Mapping image name to id.
        """
        id2name = {}
        name2id = {}
        for image_id, image in imgs.items():
            file_name = image['file_name']
            id2name[image_id] = file_name
            name2id[file_name] = image_id

        return id2name, name2id

    @abstractmethod
    def evaluate(self, results, *args, **kwargs):
        """Evaluate keypoint results."""

    @staticmethod
    def _write_keypoint_results(keypoints, res_file):
        """Write results into a json file."""

        with open(res_file, 'w') as f:
            json.dump(keypoints, f, sort_keys=True, indent=4)

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.db) // self.num_cameras

    def __getitem__(self, idx):
        """Get the sample given index."""
        results = {}
        # return self.pipeline(results)
        for c in range(self.num_cameras):
            result = copy.deepcopy(self.db[self.num_cameras * idx + c])
            result['ann_info'] = self.ann_info
            results[c] = result

        return self.pipeline(results)

    @staticmethod
    def _sort_and_unique_outputs(outputs, key='sample_id'):
        """sort outputs and remove the repeated ones."""
        outputs = sorted(outputs, key=lambda x: x[key])
        num_outputs = len(outputs)
        for i in range(num_outputs - 1, 0, -1):
            if outputs[i][key] == outputs[i - 1][key]:
                del outputs[i]

        return outputs

    def _get_scale(self, raw_image_size):
        heatmap_size = self.ann_info['heatmap_size']
        image_size = self.ann_info['image_size']
        assert heatmap_size[0][0] / heatmap_size[0][1] \
               == image_size[0] / image_size[1]
        w, h = raw_image_size
        w_resized, h_resized = image_size
        if w / w_resized < h / h_resized:
            w_pad = h / h_resized * w_resized
            h_pad = h
        else:
            w_pad = w
            h_pad = w / w_resized * h_resized

        scale = np.array([w_pad, h_pad], dtype=np.float32)

        return scale

    @staticmethod
    def rotate_points(points, center, rot_rad):
        """Rotate the points around the center.

        Args:
            points: np.ndarray, N*2
            center: np.ndarray, 2
            rot_rad: scalar
        Return:
            np.ndarray (N*2)
        """
        rot_rad = rot_rad * np.pi / 180.0
        rotate_mat = np.array([[np.cos(rot_rad), -np.sin(rot_rad)],
                               [np.sin(rot_rad),
                                np.cos(rot_rad)]])
        center = center.reshape(2, 1)
        points = points.T
        points = rotate_mat.dot(points - center) + center

        return points.T

    @staticmethod
    def calc_bbox(pose, pose_vis):
        """calculate the bbox of a pose."""
        index = pose_vis[:, 0] > 0
        bbox = [
            np.min(pose[index, 0]),
            np.min(pose[index, 1]),
            np.max(pose[index, 0]),
            np.max(pose[index, 1])
        ]

        return np.array(bbox)

    def _get_cam(self, calib):
        """Get camera parameters.

        Returns: Camera parameters.
        """
        cameras = {}
        for id, cam in calib.items():
            sel_cam = {}
            # note the transpose operation different from from VoxelPose
            sel_cam['R'] = np.array(cam['R'], dtype=np.float32).T
            sel_cam['T'] = np.array(cam['T'], dtype=np.float32)

            sel_cam['k'] = np.array(cam['k'], dtype=np.float32)
            sel_cam['p'] = np.array(cam['p'], dtype=np.float32)

            sel_cam['f'] = [[cam['fx']], [cam['fy']]]
            sel_cam['c'] = [[cam['cx']], [cam['cy']]]

            cameras[id] = sel_cam

        return cameras

    def _load_files(self):
        """load related db files."""
        assert osp.exists(self.cam_file), f'camera calibration file ' \
            f"{self.cam_file} doesn't exist, please check again"
        with open(self.cam_file) as cfile:
            calib = json.load(cfile)
        self.cameras = self._get_cam(calib)

        assert osp.exists(self.train_pose_db_file), f'train_pose_db_file ' \
            f"{self.train_pose_db_file} doesn't exist, please check again"
        with open(self.train_pose_db_file, 'rb') as pfile:
            self.train_pose_db = pickle.load(pfile)

        assert osp.exists(self.test_pose_db_file), f'test_pose_db_file ' \
            f"{self.test_pose_db_file} doesn't exist, please check again"
        with open(self.test_pose_db_file, 'rb') as pfile:
            self.test_pose_db = pickle.load(pfile)

        assert osp.exists(self.gt_pose_db_file), f'gt_pose_db_file ' \
            f"{self.gt_pose_db_file} doesn't exist, please check again"
        gt = loadmat(self.gt_pose_db_file)

        self.gt_pose_db = np.array(gt['actor3D'].tolist()).squeeze()

        self.num_persons = len(self.gt_pose_db)
