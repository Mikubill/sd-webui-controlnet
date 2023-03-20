# Copyright (c) OpenMMLab. All rights reserved.
import copy
import glob
import json
import os.path as osp
import pickle
import tempfile
import warnings
from collections import OrderedDict

import annotator.mmpkg.mmcv as mmcv
import numpy as np
from annotator.mmpkg.mmcv import Config, deprecated_api_warning

from annotator.mmpkg.mmpose.core.camera import SimpleCamera
from annotator.mmpkg.mmpose.datasets.builder import DATASETS
from annotator.mmpkg.mmpose.datasets.datasets.base import Kpt3dMviewRgbImgDirectDataset


@DATASETS.register_module()
class Body3DMviewDirectPanopticDataset(Kpt3dMviewRgbImgDirectDataset):
    """Panoptic dataset for direct multi-view human pose estimation.

    `Panoptic Studio: A Massively Multiview System for Social Motion
    Capture' ICCV'2015
    More details can be found in the `paper
    <https://openaccess.thecvf.com/content_iccv_2015/papers/
    Joo_Panoptic_Studio_A_ICCV_2015_paper.pdf>`__ .

    The dataset loads both 2D and 3D annotations as well as camera parameters.

    Panoptic keypoint indexes::

        'neck': 0,
        'nose': 1,
        'mid-hip': 2,
        'l-shoulder': 3,
        'l-elbow': 4,
        'l-wrist': 5,
        'l-hip': 6,
        'l-knee': 7,
        'l-ankle': 8,
        'r-shoulder': 9,
        'r-elbow': 10,
        'r-wrist': 11,
        'r-hip': 12,
        'r-knee': 13,
        'r-ankle': 14,
        'l-eye': 15,
        'l-ear': 16,
        'r-eye': 17,
        'r-ear': 18,

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
    ALLOWED_METRICS = {'mpjpe', 'mAP'}

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
            cfg = Config.fromfile('configs/_base_/datasets/panoptic_body3d.py')
            dataset_info = cfg._cfg_dict['dataset_info']

        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            test_mode=test_mode)

        self.load_config(data_cfg)
        self.ann_info['use_different_joint_weights'] = False

        if ann_file is None:
            self.db_file = osp.join(
                img_prefix, f'group_{self.subset}_cam{self.num_cameras}.pkl')
        else:
            self.db_file = ann_file

        if osp.exists(self.db_file):
            with open(self.db_file, 'rb') as f:
                info = pickle.load(f)
            assert info['sequence_list'] == self.seq_list
            assert info['interval'] == self.seq_frame_interval
            assert info['cam_list'] == self.cam_list
            self.db = info['db']
        else:
            self.db = self._get_db()
            info = {
                'sequence_list': self.seq_list,
                'interval': self.seq_frame_interval,
                'cam_list': self.cam_list,
                'db': self.db
            }
            with open(self.db_file, 'wb') as f:
                pickle.dump(info, f)

        self.db_size = len(self.db)

        print(f'=> load {len(self.db)} samples')

    def load_config(self, data_cfg):
        """Initialize dataset attributes according to the config.

        Override this method to set dataset specific attributes.
        """
        self.num_joints = data_cfg['num_joints']
        assert self.num_joints <= 19
        self.seq_list = data_cfg['seq_list']
        self.cam_list = data_cfg['cam_list']
        self.num_cameras = data_cfg['num_cameras']
        assert self.num_cameras == len(self.cam_list)
        self.seq_frame_interval = data_cfg.get('seq_frame_interval', 1)
        self.subset = data_cfg.get('subset', 'train')
        self.need_camera_param = True
        self.root_id = data_cfg.get('root_id', 0)
        self.max_persons = data_cfg.get('max_num', 10)

    def _get_cam(self, seq):
        """Get camera parameters.

        Args:
            seq (str): Sequence name.

        Returns: Camera parameters.
        """
        cam_file = osp.join(self.img_prefix, seq,
                            'calibration_{:s}.json'.format(seq))
        with open(cam_file) as cfile:
            calib = json.load(cfile)

        M = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        cameras = {}
        for cam in calib['cameras']:
            if (cam['panel'], cam['node']) in self.cam_list:
                sel_cam = {}
                R_w2c = np.array(cam['R']).dot(M)
                T_w2c = np.array(cam['t']).reshape((3, 1)) * 10.0  # cm to mm
                R_c2w = R_w2c.T
                T_c2w = -R_w2c.T @ T_w2c
                sel_cam['R'] = R_c2w.tolist()
                sel_cam['T'] = T_c2w.tolist()
                sel_cam['K'] = cam['K'][:2]
                distCoef = cam['distCoef']
                sel_cam['k'] = [distCoef[0], distCoef[1], distCoef[4]]
                sel_cam['p'] = [distCoef[2], distCoef[3]]
                cameras[(cam['panel'], cam['node'])] = sel_cam

        return cameras

    def _get_db(self):
        """Get dataset base.

        Returns:
            dict: the dataset base (2D and 3D information)
        """
        width = 1920
        height = 1080
        db = []
        sample_id = 0
        for seq in self.seq_list:
            cameras = self._get_cam(seq)
            curr_anno = osp.join(self.img_prefix, seq,
                                 'hdPose3d_stage1_coco19')
            anno_files = sorted(glob.iglob('{:s}/*.json'.format(curr_anno)))
            print(f'load sequence: {seq}', flush=True)
            for i, file in enumerate(anno_files):
                if i % self.seq_frame_interval == 0:
                    with open(file) as dfile:
                        bodies = json.load(dfile)['bodies']
                    if len(bodies) == 0:
                        continue

                    for k, cam_param in cameras.items():
                        single_view_camera = SimpleCamera(cam_param)
                        postfix = osp.basename(file).replace('body3DScene', '')
                        prefix = '{:02d}_{:02d}'.format(k[0], k[1])
                        image_file = osp.join(seq, 'hdImgs', prefix,
                                              prefix + postfix)
                        image_file = image_file.replace('json', 'jpg')

                        all_poses_3d = np.zeros(
                            (self.max_persons, self.num_joints, 3),
                            dtype=np.float32)
                        all_poses_vis_3d = np.zeros(
                            (self.max_persons, self.num_joints, 3),
                            dtype=np.float32)
                        all_roots_3d = np.zeros((self.max_persons, 3),
                                                dtype=np.float32)
                        all_poses = np.zeros(
                            (self.max_persons, self.num_joints, 3),
                            dtype=np.float32)

                        cnt = 0
                        person_ids = -np.ones(self.max_persons, dtype=int)
                        for body in bodies:
                            if cnt >= self.max_persons:
                                break
                            pose3d = np.array(body['joints19']).reshape(
                                (-1, 4))
                            pose3d = pose3d[:self.num_joints]

                            joints_vis = pose3d[:, -1] > 0.1

                            if not joints_vis[self.root_id]:
                                continue

                            # Coordinate transformation
                            M = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0],
                                          [0.0, 1.0, 0.0]])
                            pose3d[:, 0:3] = pose3d[:, 0:3].dot(M) * 10.0

                            all_poses_3d[cnt] = pose3d[:, :3]
                            all_roots_3d[cnt] = pose3d[self.root_id, :3]
                            all_poses_vis_3d[cnt] = np.repeat(
                                np.reshape(joints_vis, (-1, 1)), 3, axis=1)

                            pose2d = np.zeros((pose3d.shape[0], 3))
                            # get pose_2d from pose_3d
                            pose2d[:, :2] = single_view_camera.world_to_pixel(
                                pose3d[:, :3])
                            x_check = np.bitwise_and(pose2d[:, 0] >= 0,
                                                     pose2d[:, 0] <= width - 1)
                            y_check = np.bitwise_and(
                                pose2d[:, 1] >= 0, pose2d[:, 1] <= height - 1)
                            check = np.bitwise_and(x_check, y_check)
                            joints_vis[np.logical_not(check)] = 0
                            pose2d[:, -1] = joints_vis

                            all_poses[cnt] = pose2d
                            person_ids[cnt] = body['id']
                            cnt += 1

                        if cnt > 0:
                            db.append({
                                'image_file':
                                osp.join(self.img_prefix, image_file),
                                'joints_3d':
                                all_poses_3d,
                                'person_ids':
                                person_ids,
                                'joints_3d_visible':
                                all_poses_vis_3d,
                                'joints': [all_poses],
                                'roots_3d':
                                all_roots_3d,
                                'camera':
                                cam_param,
                                'num_persons':
                                cnt,
                                'sample_id':
                                sample_id,
                                'center':
                                np.array((width / 2, height / 2),
                                         dtype=np.float32),
                                'scale':
                                self._get_scale((width, height))
                            })
                            sample_id += 1
        return db

    @deprecated_api_warning(name_dict=dict(outputs='results'))
    def evaluate(self, results, res_folder=None, metric='mpjpe', **kwargs):
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
                Defaults: 'mpjpe'.
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

        eval_list = []
        gt_num = self.db_size // self.num_cameras
        assert len(
            _results) == gt_num, f'number mismatch: {len(_results)}, {gt_num}'

        total_gt = 0
        for i in range(gt_num):
            index = self.num_cameras * i
            db_rec = copy.deepcopy(self.db[index])
            joints_3d = db_rec['joints_3d']
            joints_3d_vis = db_rec['joints_3d_visible']

            if joints_3d_vis.sum() < 1:
                continue

            pred = _results[i]['pose_3d'].copy()
            pred = pred[pred[:, 0, 3] >= 0]
            for pose in pred:
                mpjpes = []
                for (gt, gt_vis) in zip(joints_3d, joints_3d_vis):
                    vis = gt_vis[:, 0] > 0
                    if vis.sum() < 1:
                        break
                    mpjpe = np.mean(
                        np.sqrt(
                            np.sum((pose[vis, 0:3] - gt[vis])**2, axis=-1)))
                    mpjpes.append(mpjpe)
                min_gt = np.argmin(mpjpes)
                min_mpjpe = np.min(mpjpes)
                score = pose[0, 4]
                eval_list.append({
                    'mpjpe': float(min_mpjpe),
                    'score': float(score),
                    'gt_id': int(total_gt + min_gt)
                })

            total_gt += (joints_3d_vis[:, :, 0].sum(-1) >= 1).sum()

        mpjpe_threshold = np.arange(25, 155, 25)
        aps = []
        ars = []
        for t in mpjpe_threshold:
            ap, ar = self._eval_list_to_ap(eval_list, total_gt, t)
            aps.append(ap)
            ars.append(ar)

        name_value_tuples = []
        for _metric in metrics:
            if _metric == 'mpjpe':
                stats_names = ['RECALL 500mm', 'MPJPE 500mm']
                info_str = list(
                    zip(stats_names, [
                        self._eval_list_to_recall(eval_list, total_gt),
                        self._eval_list_to_mpjpe(eval_list)
                    ]))
            elif _metric == 'mAP':
                stats_names = [
                    'AP 25', 'AP 50', 'AP 75', 'AP 100', 'AP 125', 'AP 150',
                    'mAP', 'AR 25', 'AR 50', 'AR 75', 'AR 100', 'AR 125',
                    'AR 150', 'mAR'
                ]
                mAP = np.array(aps).mean()
                mAR = np.array(ars).mean()
                info_str = list(zip(stats_names, aps + [mAP] + ars + [mAR]))
            else:
                raise NotImplementedError
            name_value_tuples.extend(info_str)

        if tmp_folder is not None:
            tmp_folder.cleanup()

        return OrderedDict(name_value_tuples)

    @staticmethod
    def _eval_list_to_ap(eval_list, total_gt, threshold):
        """Get Average Precision (AP) and Average Recall at a certain
        threshold."""

        eval_list.sort(key=lambda k: k['score'], reverse=True)
        total_num = len(eval_list)

        tp = np.zeros(total_num)
        fp = np.zeros(total_num)
        gt_det = []
        for i, item in enumerate(eval_list):
            if item['mpjpe'] < threshold and item['gt_id'] not in gt_det:
                tp[i] = 1
                gt_det.append(item['gt_id'])
            else:
                fp[i] = 1
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        recall = tp / (total_gt + 1e-5)
        precise = tp / (tp + fp + 1e-5)
        for n in range(total_num - 2, -1, -1):
            precise[n] = max(precise[n], precise[n + 1])

        precise = np.concatenate(([0], precise, [0]))
        recall = np.concatenate(([0], recall, [1]))
        index = np.where(recall[1:] != recall[:-1])[0]
        ap = np.sum((recall[index + 1] - recall[index]) * precise[index + 1])

        return ap, recall[-2]

    @staticmethod
    def _eval_list_to_mpjpe(eval_list, threshold=500):
        """Get MPJPE within a certain threshold."""
        eval_list.sort(key=lambda k: k['score'], reverse=True)
        gt_det = []

        mpjpes = []
        for i, item in enumerate(eval_list):
            if item['mpjpe'] < threshold and item['gt_id'] not in gt_det:
                mpjpes.append(item['mpjpe'])
                gt_det.append(item['gt_id'])

        return np.mean(mpjpes) if len(mpjpes) > 0 else np.inf

    @staticmethod
    def _eval_list_to_recall(eval_list, total_gt, threshold=500):
        """Get Recall at a certain threshold."""
        gt_ids = [e['gt_id'] for e in eval_list if e['mpjpe'] < threshold]

        return len(np.unique(gt_ids)) / total_gt

    def __getitem__(self, idx):
        """Get the sample given index."""
        results = {}
        for c in range(self.num_cameras):
            result = copy.deepcopy(self.db[self.num_cameras * idx + c])
            result['ann_info'] = self.ann_info
            width = 1920
            height = 1080
            result['mask'] = [np.ones((height, width), dtype=np.float32)]
            results[c] = result

        return self.pipeline(results)
