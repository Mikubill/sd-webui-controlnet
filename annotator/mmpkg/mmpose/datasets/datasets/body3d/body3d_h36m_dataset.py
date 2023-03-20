# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict, defaultdict

import annotator.mmpkg.mmcv as mmcv
import numpy as np
from annotator.mmpkg.mmcv import Config, deprecated_api_warning

from annotator.mmpkg.mmpose.core.evaluation import keypoint_mpjpe
from annotator.mmpkg.mmpose.datasets.datasets.base import Kpt3dSviewKpt2dDataset
from ...builder import DATASETS


@DATASETS.register_module()
class Body3DH36MDataset(Kpt3dSviewKpt2dDataset):
    """Human3.6M dataset for 3D human pose estimation.

    "Human3.6M: Large Scale Datasets and Predictive Methods for 3D Human
    Sensing in Natural Environments", TPAMI`2014.
    More details can be found in the `paper
    <http://vision.imar.ro/human3.6m/pami-h36m.pdf>`__.

    Human3.6M keypoint indexes::

        0: 'root (pelvis)',
        1: 'right_hip',
        2: 'right_knee',
        3: 'right_foot',
        4: 'left_hip',
        5: 'left_knee',
        6: 'left_foot',
        7: 'spine',
        8: 'thorax',
        9: 'neck_base',
        10: 'head',
        11: 'left_shoulder',
        12: 'left_elbow',
        13: 'left_wrist',
        14: 'right_shoulder',
        15: 'right_elbow',
        16: 'right_wrist'


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

    JOINT_NAMES = [
        'Root', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot', 'Spine',
        'Thorax', 'NeckBase', 'Head', 'LShoulder', 'LElbow', 'LWrist',
        'RShoulder', 'RElbow', 'RWrist'
    ]

    # 2D joint source options:
    # "gt": from the annotation file
    # "detection": from a detection result file of 2D keypoint
    # "pipeline": will be generate by the pipeline
    SUPPORTED_JOINT_2D_SRC = {'gt', 'detection', 'pipeline'}

    # metric
    ALLOWED_METRICS = {'mpjpe', 'p-mpjpe', 'n-mpjpe'}

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
            cfg = Config.fromfile('configs/_base_/datasets/h36m.py')
            dataset_info = cfg._cfg_dict['dataset_info']

        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            test_mode=test_mode)

    def load_config(self, data_cfg):
        super().load_config(data_cfg)
        # h36m specific attributes
        self.joint_2d_src = data_cfg.get('joint_2d_src', 'gt')
        if self.joint_2d_src not in self.SUPPORTED_JOINT_2D_SRC:
            raise ValueError(
                f'Unsupported joint_2d_src "{self.joint_2d_src}". '
                f'Supported options are {self.SUPPORTED_JOINT_2D_SRC}')

        self.joint_2d_det_file = data_cfg.get('joint_2d_det_file', None)

        self.need_camera_param = data_cfg.get('need_camera_param', False)
        if self.need_camera_param:
            assert 'camera_param_file' in data_cfg
            self.camera_param = self._load_camera_param(
                data_cfg['camera_param_file'])

        # h36m specific annotation info
        ann_info = {}
        ann_info['use_different_joint_weights'] = False
        # action filter
        actions = data_cfg.get('actions', '_all_')
        self.actions = set(
            actions if isinstance(actions, (list, tuple)) else [actions])

        # subject filter
        subjects = data_cfg.get('subjects', '_all_')
        self.subjects = set(
            subjects if isinstance(subjects, (list, tuple)) else [subjects])

        self.ann_info.update(ann_info)

    def load_annotations(self):
        data_info = super().load_annotations()

        # get 2D joints
        if self.joint_2d_src == 'gt':
            data_info['joints_2d'] = data_info['joints_2d']
        elif self.joint_2d_src == 'detection':
            data_info['joints_2d'] = self._load_joint_2d_detection(
                self.joint_2d_det_file)
            assert data_info['joints_2d'].shape[0] == data_info[
                'joints_3d'].shape[0]
            assert data_info['joints_2d'].shape[2] == 3
        elif self.joint_2d_src == 'pipeline':
            # joint_2d will be generated in the pipeline
            pass
        else:
            raise NotImplementedError(
                f'Unhandled joint_2d_src option {self.joint_2d_src}')

        return data_info

    @staticmethod
    def _parse_h36m_imgname(imgname):
        """Parse imgname to get information of subject, action and camera.

        A typical h36m image filename is like:
        S1_Directions_1.54138969_000001.jpg
        """
        subj, rest = osp.basename(imgname).split('_', 1)
        action, rest = rest.split('.', 1)
        camera, rest = rest.split('_', 1)

        return subj, action, camera

    def build_sample_indices(self):
        """Split original videos into sequences and build frame indices.

        This method overrides the default one in the base class.
        """

        # Group frames into videos. Assume that self.data_info is
        # chronological.
        video_frames = defaultdict(list)
        for idx, imgname in enumerate(self.data_info['imgnames']):
            subj, action, camera = self._parse_h36m_imgname(imgname)

            if '_all_' not in self.actions and action not in self.actions:
                continue

            if '_all_' not in self.subjects and subj not in self.subjects:
                continue

            video_frames[(subj, action, camera)].append(idx)

        # build sample indices
        sample_indices = []
        _len = (self.seq_len - 1) * self.seq_frame_interval + 1
        _step = self.seq_frame_interval
        for _, _indices in sorted(video_frames.items()):
            n_frame = len(_indices)

            if self.temporal_padding:
                # Pad the sequence so that every frame in the sequence will be
                # predicted.
                if self.causal:
                    frames_left = self.seq_len - 1
                    frames_right = 0
                else:
                    frames_left = (self.seq_len - 1) // 2
                    frames_right = frames_left
                for i in range(n_frame):
                    pad_left = max(0, frames_left - i // _step)
                    pad_right = max(0,
                                    frames_right - (n_frame - 1 - i) // _step)
                    start = max(i % _step, i - frames_left * _step)
                    end = min(n_frame - (n_frame - 1 - i) % _step,
                              i + frames_right * _step + 1)
                    sample_indices.append([_indices[0]] * pad_left +
                                          _indices[start:end:_step] +
                                          [_indices[-1]] * pad_right)
            else:
                seqs_from_video = [
                    _indices[i:(i + _len):_step]
                    for i in range(0, n_frame - _len + 1)
                ]
                sample_indices.extend(seqs_from_video)

        # reduce dataset size if self.subset < 1
        assert 0 < self.subset <= 1
        subset_size = int(len(sample_indices) * self.subset)
        start = np.random.randint(0, len(sample_indices) - subset_size + 1)
        end = start + subset_size

        return sample_indices[start:end]

    def _load_joint_2d_detection(self, det_file):
        """"Load 2D joint detection results from file."""
        joints_2d = np.load(det_file).astype(np.float32)

        return joints_2d

    @deprecated_api_warning(name_dict=dict(outputs='results'))
    def evaluate(self, results, res_folder=None, metric='mpjpe', **kwargs):
        metrics = metric if isinstance(metric, list) else [metric]
        for _metric in metrics:
            if _metric not in self.ALLOWED_METRICS:
                raise ValueError(
                    f'Unsupported metric "{_metric}" for human3.6 dataset.'
                    f'Supported metrics are {self.ALLOWED_METRICS}')

        if res_folder is not None:
            tmp_folder = None
            res_file = osp.join(res_folder, 'result_keypoints.json')
        else:
            tmp_folder = tempfile.TemporaryDirectory()
            res_file = osp.join(tmp_folder.name, 'result_keypoints.json')

        kpts = []
        for result in results:
            preds = result['preds']
            image_paths = result['target_image_paths']
            batch_size = len(image_paths)
            for i in range(batch_size):
                target_id = self.name2id[image_paths[i]]
                kpts.append({
                    'keypoints': preds[i],
                    'target_id': target_id,
                })

        mmcv.dump(kpts, res_file)

        name_value_tuples = []
        for _metric in metrics:
            if _metric == 'mpjpe':
                _nv_tuples = self._report_mpjpe(kpts)
            elif _metric == 'p-mpjpe':
                _nv_tuples = self._report_mpjpe(kpts, mode='p-mpjpe')
            elif _metric == 'n-mpjpe':
                _nv_tuples = self._report_mpjpe(kpts, mode='n-mpjpe')
            else:
                raise NotImplementedError
            name_value_tuples.extend(_nv_tuples)

        if tmp_folder is not None:
            tmp_folder.cleanup()

        return OrderedDict(name_value_tuples)

    def _report_mpjpe(self, keypoint_results, mode='mpjpe'):
        """Cauculate mean per joint position error (MPJPE) or its variants like
        P-MPJPE or N-MPJPE.

        Args:
            keypoint_results (list): Keypoint predictions. See
                'Body3DH36MDataset.evaluate' for details.
            mode (str): Specify mpjpe variants. Supported options are:

                - ``'mpjpe'``: Standard MPJPE.
                - ``'p-mpjpe'``: MPJPE after aligning prediction to groundtruth
                    via a rigid transformation (scale, rotation and
                    translation).
                - ``'n-mpjpe'``: MPJPE after aligning prediction to groundtruth
                    in scale only.
        """

        preds = []
        gts = []
        masks = []
        action_category_indices = defaultdict(list)
        for idx, result in enumerate(keypoint_results):
            pred = result['keypoints']
            target_id = result['target_id']
            gt, gt_visible = np.split(
                self.data_info['joints_3d'][target_id], [3], axis=-1)
            preds.append(pred)
            gts.append(gt)
            masks.append(gt_visible)

            action = self._parse_h36m_imgname(
                self.data_info['imgnames'][target_id])[1]
            action_category = action.split('_')[0]
            action_category_indices[action_category].append(idx)

        preds = np.stack(preds)
        gts = np.stack(gts)
        masks = np.stack(masks).squeeze(-1) > 0

        err_name = mode.upper()
        if mode == 'mpjpe':
            alignment = 'none'
        elif mode == 'p-mpjpe':
            alignment = 'procrustes'
        elif mode == 'n-mpjpe':
            alignment = 'scale'
        else:
            raise ValueError(f'Invalid mode: {mode}')

        error = keypoint_mpjpe(preds, gts, masks, alignment)
        name_value_tuples = [(err_name, error)]

        for action_category, indices in action_category_indices.items():
            _error = keypoint_mpjpe(preds[indices], gts[indices],
                                    masks[indices])
            name_value_tuples.append((f'{err_name}_{action_category}', _error))

        return name_value_tuples

    def _load_camera_param(self, camera_param_file):
        """Load camera parameters from file."""
        return mmcv.load(camera_param_file)

    def get_camera_param(self, imgname):
        """Get camera parameters of a frame by its image name."""
        assert hasattr(self, 'camera_param')
        subj, _, camera = self._parse_h36m_imgname(imgname)
        return self.camera_param[(subj, camera)]
