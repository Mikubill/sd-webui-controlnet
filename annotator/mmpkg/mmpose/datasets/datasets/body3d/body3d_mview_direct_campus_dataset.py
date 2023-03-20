# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------------
# Adapted from https://github.com/microsoft/voxelpose-pytorch
# Original license: Copyright (c) Microsoft Corporation, under the MIT License.
# ------------------------------------------------------------------------------

import copy
import os.path as osp
import random
import tempfile
import warnings
from collections import OrderedDict

import annotator.mmpkg.mmcv as mmcv
import numpy as np
from annotator.mmpkg.mmcv import Config

from annotator.mmpkg.mmpose.core.camera import SimpleCamera
from annotator.mmpkg.mmpose.datasets.builder import DATASETS
from annotator.mmpkg.mmpose.datasets.datasets.base import Kpt3dMviewRgbImgDirectDataset


@DATASETS.register_module()
class Body3DMviewDirectCampusDataset(Kpt3dMviewRgbImgDirectDataset):
    """Campus dataset for direct multi-view human pose estimation.

    `3D Pictorial Structures for Multiple Human Pose Estimation' CVPR'2014
    More details can be found in the paper
    <http://campar.in.tum.de/pub/belagiannis2014cvpr/belagiannis2014cvpr.pdf>`

    The dataset loads both 2D and 3D annotations as well as camera parameters.
    It is worth mentioning that when training multi-view 3D pose models,
    due to the limited and incomplete annotations of this dataset, we may not
    use this dataset to train the model. Instead, we use the 2D pose estimator
    trained on COCO, and use independent 3D human poses from the CMU Panoptic
    dataset to train the 3D model.
    For testing, we first estimate 2D poses and generate 2D heatmaps for this
    dataset as the input to 3D model.

    Campus keypoint indices::

        'Right-Ankle': 0,
        'Right-Knee': 1,
        'Right-Hip': 2,
        'Left-Hip': 3,
        'Left-Knee': 4,
        'Left-Ankle': 5,
        'Right-Wrist': 6,
        'Right-Elbow': 7,
        'Right-Shoulder': 8,
        'Left-Shoulder': 9,
        'Left-Elbow': 10,
        'Left-Wrist': 11,
        'Bottom-Head': 12,
        'Top-Head': 13,

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
    ALLOWED_METRICS = {'pcp', '3dpcp'}
    LIMBS = [[0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8], [9, 10], [10, 11],
             [12, 13]]
    BONE_GROUP = OrderedDict([('Head', [8]), ('Torso', [9]),
                              ('Upper arms', [5, 6]), ('Lower arms', [4, 7]),
                              ('Upper legs', [1, 2]), ('Lower legs', [0, 3])])

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
            cfg = Config.fromfile('configs/_base_/datasets/campus.py')
            dataset_info = cfg._cfg_dict['dataset_info']

        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            test_mode=test_mode)

        self.load_config(data_cfg)
        self.ann_info['use_different_joint_weights'] = data_cfg.get(
            'use_different_joint_weights', False)

        self.db_size = self.num_cameras * len(
            self.frame_range
        ) if self.test_mode else self.num_cameras * self.num_train_samples
        print(f'=> load {self.db_size} samples')

    def load_config(self, data_cfg):
        """Initialize dataset attributes according to the config.

        Override this method to set dataset specific attributes.
        """
        self.num_joints = data_cfg['num_joints']
        self.cam_list = data_cfg['cam_list']
        self.num_cameras = data_cfg['num_cameras']
        assert self.num_cameras == len(self.cam_list)
        self.need_camera_param = True

        self.frame_range = data_cfg['frame_range']
        self.width = data_cfg.get('width', 360)
        self.height = data_cfg.get('height', 288)
        self.center = np.array((self.width / 2, self.height / 2),
                               dtype=np.float32)
        self.scale = self._get_scale((self.width, self.height))

        root_id = data_cfg.get('root_id', [11, 12])
        self.root_id = [root_id] if isinstance(root_id, int) else root_id

        self.max_nposes = data_cfg.get('max_nposes', 10)
        self.min_nposes = data_cfg.get('min_nposes', 1)
        self.num_train_samples = data_cfg.get('num_train_samples', 3000)
        self.maximum_person = data_cfg.get('maximum_person', 10)

        self.cam_file = data_cfg.get(
            'cam_file', osp.join(self.img_prefix, 'calibration_campus.json'))

        self.test_pose_db_file = data_cfg.get(
            'test_pose_db_file',
            osp.join(self.img_prefix, 'pred_campus_maskrcnn_hrnet_coco.pkl'))

        self.train_pose_db_file = data_cfg.get(
            'train_pose_db_file',
            osp.join(self.img_prefix, 'panoptic_training_pose.pkl'))

        self.gt_pose_db_file = data_cfg.get(
            'gt_pose_db_file', osp.join(self.img_prefix, 'actorsGT.mat'))

        self._load_files()

    def _get_db(self):
        """Load dataset."""
        raise NotImplementedError(
            '_get_db method is not overwritten here because of two reasons.'
            'First, the training and test samples are quite different. '
            'Second, the training samples have some randomness which is not'
            'appropriate to collect all samples into a database one time.')

    def __getitem__(self, idx):
        """Get the sample given index."""

        if self.test_mode:
            results = self._prepare_test_sample(idx)
        else:
            results = self._prepare_train_sample(idx)

        return self.pipeline(results)

    def _prepare_test_sample(self, idx):
        results = {}
        fid = self.frame_range[idx]

        for cam_id, cam_param in self.cameras.items():
            image_file = osp.join(
                self.img_prefix, 'Camera' + cam_id,
                'campus4-c{0}-{1:05d}.png'.format(cam_id, fid))

            all_poses_3d = []
            all_poses_3d_vis = []
            all_poses_2d = []
            all_poses_2d_vis = []
            single_view_camera = SimpleCamera(cam_param)

            for person in range(self.num_persons):
                pose3d = self.gt_pose_db[person][fid] * 1000.0
                if len(pose3d[0]) > 0:
                    all_poses_3d.append(pose3d)
                    all_poses_3d_vis.append(np.ones((self.num_joints, 3)))

                    pose2d = single_view_camera.world_to_pixel(pose3d)
                    x_check = np.bitwise_and(pose2d[:, 0] >= 0,
                                             pose2d[:, 0] <= self.width - 1)
                    y_check = np.bitwise_and(pose2d[:, 1] >= 0,
                                             pose2d[:, 1] <= self.height - 1)
                    check = np.bitwise_and(x_check, y_check)

                    joints_vis = np.ones((len(pose2d), 1))
                    joints_vis[np.logical_not(check)] = 0
                    all_poses_2d.append(pose2d)
                    all_poses_2d_vis.append(
                        np.repeat(np.reshape(joints_vis, (-1, 1)), 2, axis=1))

            pred_index = '{}_{}'.format(cam_id, fid)
            pred_poses = self.test_pose_db[pred_index]
            preds = []
            for pose in pred_poses:
                preds.append(np.array(pose['pred']))
            preds = np.array(preds)

            results[int(cam_id)] = {
                'image_file': image_file,
                'joints_3d': all_poses_3d,
                'joints_3d_visible': all_poses_3d_vis,
                'joints_2d': all_poses_2d,
                'joints_2d_visible': all_poses_2d_vis,
                'camera': cam_param,
                'joints': preds,
                'sample_id': idx * self.num_cameras + int(cam_id),
                'center': self.center,
                'scale': self.scale,
                'rotation': 0.0,
                'ann_info': self.ann_info
            }

        return results

    def _prepare_train_sample(self, idx):
        results = {}
        # To prepare a training sample, there are three steps.
        # 1. Randomly sample some 3D poses from motion capture database
        nposes_ori = np.random.choice(range(self.min_nposes, self.max_nposes))
        select_poses = np.random.choice(self.train_pose_db, nposes_ori)

        joints_3d = np.array([p['pose'] for p in select_poses])
        joints_3d_vis = np.array([p['vis'] for p in select_poses])

        bbox_list = []
        center_list = []
        # 2. Place the selected poses at random locations in the space
        for n in range(nposes_ori):
            points = joints_3d[n][:, :2].copy()
            # get the location of a person's root joint
            center = np.mean(points[self.root_id, :2], axis=0)
            rot_rad = np.random.uniform(-180, 180)

            new_center = self.get_new_center(center_list)
            new_xy = self.rotate_points(points, center,
                                        rot_rad) - center + new_center

            loop_count = 0
            # here n will be at least 1
            while not self.isvalid(new_center,
                                   self.calc_bbox(new_xy, joints_3d_vis[n]),
                                   bbox_list):
                loop_count += 1
                if loop_count >= 100:
                    break
                new_center = self.get_new_center(center_list)
                new_xy = self.rotate_points(points, center,
                                            rot_rad) - center + new_center

            if loop_count >= 100:
                nposes = n
                joints_3d = joints_3d[:n]
                joints_3d_vis = joints_3d_vis[:n]
                break
            else:
                nposes = nposes_ori
                center_list.append(new_center)
                bbox_list.append(self.calc_bbox(new_xy, joints_3d_vis[n]))
                joints_3d[n][:, :2] = new_xy

        joints_3d_u = np.zeros((self.maximum_person, len(joints_3d[0]), 3))
        joints_3d_vis_u = np.zeros((self.maximum_person, len(joints_3d[0]), 3))
        for i in range(nposes):
            joints_3d_u[i] = joints_3d[i][:, 0:3]
            joints_3d_vis_u[i] = joints_3d_vis[i][:, 0:3]

        roots_3d = np.mean(joints_3d_u[:, self.root_id], axis=1)

        # 3. Project 3D poses to all views to get the respective 2D locations
        for cam_id, cam_param in self.cameras.items():
            joints = []
            joints_vis = []
            single_view_camera = SimpleCamera(cam_param)
            for n in range(nposes):
                # project the 3D pose to the view to get 2D location
                pose2d = single_view_camera.world_to_pixel(joints_3d[n])
                # check the validity of joint cooridinate
                x_check = np.bitwise_and(pose2d[:, 0] >= 0,
                                         pose2d[:, 0] <= self.width - 1)
                y_check = np.bitwise_and(pose2d[:, 1] >= 0,
                                         pose2d[:, 1] <= self.height - 1)
                check = np.bitwise_and(x_check, y_check)
                vis = joints_3d_vis[n][:, 0] > 0
                vis[np.logical_not(check)] = 0

                joints.append(pose2d)
                joints_vis.append(
                    np.repeat(np.reshape(vis, (-1, 1)), 2, axis=1))

            # make joints and joints_vis having same shape
            joints_u = np.zeros((self.maximum_person, len(joints[0]), 2))
            joints_vis_u = np.zeros((self.maximum_person, len(joints[0]), 2))
            for i in range(nposes):
                joints_u[i] = joints[i]
                joints_vis_u[i] = joints_vis[i]

            results[int(cam_id)] = {
                'joints_3d': joints_3d_u,
                'joints_3d_visible': joints_3d_vis_u,
                'roots_3d': roots_3d,
                'joints': joints_u,
                'joints_visible': joints_vis_u,
                'camera': cam_param,
                'sample_id': idx * self.num_cameras + int(cam_id),
                'center': self.center,
                'scale': self.scale,
                'rotation': 0.0,
                'num_persons': nposes,
                'ann_info': self.ann_info
            }

        return results

    def __len__(self):
        """Get the size of the dataset."""
        if self.test_mode:
            return len(self.frame_range)
        else:
            return self.num_train_samples

    @staticmethod
    def get_new_center(center_list):
        """Generate new center or select from the center list randomly.

        The proability and the parameters related to cooridinates can also be
        tuned, just make sure that the center is within the given 3D space.
        """
        if len(center_list) == 0 or random.random() < 0.7:
            new_center = np.array([
                np.random.uniform(-2500.0, 8500.0),
                np.random.uniform(-1000.0, 10000.0)
            ])
        else:
            xy = center_list[np.random.choice(range(len(center_list)))]
            new_center = xy + np.random.normal(500, 50, 2) * np.random.choice(
                [1, -1], 2)

        return new_center

    def isvalid(self, new_center, bbox, bbox_list):
        """Check if the new person bbox are valid, which need to satisfies:

        1. the center is visible in at least 2 views, and
        2. have a sufficiently small iou with all other person bboxes.
        """
        new_center_us = new_center.reshape(1, -1)
        vis = 0
        for _, cam_param in self.cameras.items():
            single_view_camera = SimpleCamera(cam_param)
            loc_2d = single_view_camera.world_to_pixel(
                np.hstack((new_center_us, [[1000.0]])))
            if 10 < loc_2d[0, 0] < self.width - 10 and 10 < loc_2d[
                    0, 1] < self.height - 10:
                vis += 1

        if len(bbox_list) == 0:
            return vis >= 2

        bbox_list = np.array(bbox_list)
        x0 = np.maximum(bbox[0], bbox_list[:, 0])
        y0 = np.maximum(bbox[1], bbox_list[:, 1])
        x1 = np.minimum(bbox[2], bbox_list[:, 2])
        y1 = np.minimum(bbox[3], bbox_list[:, 3])

        intersection = np.maximum(0, (x1 - x0) * (y1 - y0))
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        area_list = (bbox_list[:, 2] - bbox_list[:, 0]) * (
            bbox_list[:, 3] - bbox_list[:, 1])
        iou_list = intersection / (area + area_list - intersection)

        return vis >= 2 and np.max(iou_list) < 0.01

    def evaluate(self,
                 results,
                 res_folder=None,
                 metric='pcp',
                 recall_threshold=500,
                 alpha_error=0.5,
                 **kwargs):
        """
        Args:
            results (list[dict]): Testing results containing the following
                items:
                - pose_3d (np.ndarray): predicted 3D human pose
                - sample_id (np.ndarray): sample id of a frame.
            res_folder (str, optional): The folder to save the testing
                results. If not specified, a temp folder will be created.
                Default: None.
            metric (str | list[str]): Metric to be performed.
                Defaults: 'pcp'.
            recall_threshold: threshold for calculating recall.
            alpha_error: coefficient when calculating error for correct parts.
            **kwargs:

        Returns:

        """
        pose_3ds = np.concatenate([result['pose_3d'] for result in results],
                                  axis=0)
        sample_ids = []
        for result in results:
            sample_ids.extend(result['sample_id'])
        _results = [
            dict(sample_id=sample_id, pose_3d=pose_3d)
            for (sample_id, pose_3d) in zip(sample_ids, pose_3ds)
        ]
        _results = self._sort_and_unique_outputs(_results, key='sample_id')

        metrics = metric if isinstance(metric, list) else [metric]
        for _metric in metrics:
            if _metric not in self.ALLOWED_METRICS:
                raise ValueError(
                    f'Unsupported metric "{_metric}"'
                    f'Supported metrics are {self.ALLOWED_METRICS}')

        if res_folder is not None:
            tmp_folder = None
            res_file = osp.join(res_folder, 'result_keypoints.json')
        else:
            tmp_folder = tempfile.TemporaryDirectory()
            res_file = osp.join(tmp_folder.name, 'result_keypoints.json')
        mmcv.dump(_results, res_file)

        gt_num = self.db_size // self.num_cameras
        assert len(
            _results) == gt_num, f'number mismatch: {len(_results)}, {gt_num}'

        match_gt = 0
        total_gt = 0

        correct_parts = np.zeros(self.num_persons)
        total_parts = np.zeros(self.num_persons)
        bone_correct_parts = np.zeros((self.num_persons, len(self.LIMBS) + 1))

        for i, fid in enumerate(self.frame_range):
            pred_coco = pose_3ds[i].copy()
            pred_coco = pred_coco[pred_coco[:, 0, 3] >= 0, :, :3]

            if len(pred_coco) == 0:
                continue

            pred = np.stack([
                self.coco2campus3D(p)
                for p in copy.deepcopy(pred_coco[:, :, :3])
            ])

            for person in range(self.num_persons):
                gt = self.gt_pose_db[person][fid] * 1000.0
                if len(gt[0]) == 0:
                    continue

                mpjpes = np.mean(
                    np.sqrt(np.sum((gt[np.newaxis] - pred)**2, axis=-1)),
                    axis=-1)
                min_n = np.argmin(mpjpes)
                min_mpjpe = np.min(mpjpes)
                if min_mpjpe < recall_threshold:
                    match_gt += 1
                total_gt += 1

                for j, k in enumerate(self.LIMBS):
                    total_parts[person] += 1
                    error_s = np.linalg.norm(pred[min_n, k[0], 0:3] - gt[k[0]])
                    error_e = np.linalg.norm(pred[min_n, k[1], 0:3] - gt[k[1]])
                    limb_length = np.linalg.norm(gt[k[0]] - gt[k[1]])

                    if (error_s + error_e) / 2.0 <= alpha_error * limb_length:
                        correct_parts[person] += 1
                        bone_correct_parts[person, j] += 1

                # an extra limb
                total_parts[person] += 1
                # hip position
                rhip_idx, lhip_idx = 2, 3
                pred_hip = (pred[min_n, rhip_idx, 0:3] +
                            pred[min_n, lhip_idx, 0:3]) / 2.0
                gt_hip = (gt[rhip_idx] + gt[lhip_idx]) / 2.0
                error_s = np.linalg.norm(pred_hip - gt_hip)
                # bottom-head position
                bh_idx = 12
                error_e = np.linalg.norm(pred[min_n, bh_idx, 0:3] - gt[bh_idx])
                limb_length = np.linalg.norm(gt_hip - gt[bh_idx])

                if (error_e + error_s) / 2.0 <= alpha_error * limb_length:
                    correct_parts[person] += 1
                    bone_correct_parts[person, -1] += 1

        actor_pcp = correct_parts / (total_parts + 1e-8) * 100.0
        avg_pcp = np.mean(actor_pcp[:3])

        stats_names = [
            f'Actor {person+1} Total PCP' for person in range(self.num_persons)
        ] + ['pcp']
        stats_values = [*actor_pcp, avg_pcp]

        results = OrderedDict()
        for name, value in zip(stats_names, stats_values):
            results[name] = value

        for k, v in self.BONE_GROUP.items():
            cum_pcp = 0
            for person in range(self.num_persons):
                new_k = f'Actor {person+1} ' + k + ' PCP'
                pcp = np.sum(
                    bone_correct_parts[person, v],
                    axis=-1) / (total_parts[person] /
                                (len(self.LIMBS) + 1) * len(v) + 1e-8) * 100
                results[new_k] = pcp
                cum_pcp += pcp

            new_k = 'Average ' + k + ' PCP'
            results[new_k] = cum_pcp / self.num_persons

        return results

    @staticmethod
    def coco2campus3D(coco_pose):
        """transform coco order(our method output) 3d pose to campus dataset
        order with interpolation.

        Args:
            coco_pose: np.array with shape 17x3

        Returns: 3D pose in campus order with shape 14x3
        """
        campus_pose = np.zeros((14, 3))
        coco2campus = np.array([16, 14, 12, 11, 13, 15, 10, 8, 6, 5, 7, 9])
        campus_pose[0:12] += coco_pose[coco2campus]

        # L and R shoulder
        mid_sho = (coco_pose[5] + coco_pose[6]) / 2
        # middle of two ear
        head_center = (coco_pose[3] + coco_pose[4]) / 2

        # nose and head center
        head_bottom = (mid_sho + head_center) / 2
        head_top = head_bottom + (head_center - head_bottom) * 2
        campus_pose[12] += head_bottom
        campus_pose[13] += head_top

        return campus_pose
