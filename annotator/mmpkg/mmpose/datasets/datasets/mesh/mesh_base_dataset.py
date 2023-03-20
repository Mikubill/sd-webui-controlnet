# Copyright (c) OpenMMLab. All rights reserved.
import copy as cp
import os
from abc import ABCMeta

import numpy as np
from torch.utils.data import Dataset

from annotator.mmpkg.mmpose.datasets.pipelines import Compose


class MeshBaseDataset(Dataset, metaclass=ABCMeta):
    """Base dataset for 3D human mesh estimation task. In 3D humamesh
    estimation task, all datasets share this BaseDataset for training and have
    their own evaluate function.

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    This dataset can only be used for training.
    For evaluation, subclass should write an extra evaluate function.

    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
    """

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 test_mode=False):

        self.image_info = {}
        self.ann_info = {}

        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.pipeline = pipeline
        self.test_mode = test_mode

        self.ann_info['image_size'] = np.array(data_cfg['image_size'])
        self.ann_info['iuv_size'] = np.array(data_cfg['iuv_size'])
        self.ann_info['num_joints'] = data_cfg['num_joints']
        self.ann_info['flip_pairs'] = None
        self.db = []
        self.pipeline = Compose(self.pipeline)

        # flip_pairs
        # For all mesh dataset, we use 24 joints as CMR and SPIN.
        self.ann_info['flip_pairs'] = [[0, 5], [1, 4], [2, 3], [6, 11],
                                       [7, 10], [8, 9], [20, 21], [22, 23]]
        self.ann_info['use_different_joint_weights'] = False
        assert self.ann_info['num_joints'] == 24
        self.ann_info['joint_weights'] = np.ones([24, 1], dtype=np.float32)

        self.ann_info['uv_type'] = data_cfg['uv_type']
        self.ann_info['use_IUV'] = data_cfg['use_IUV']
        uv_type = self.ann_info['uv_type']
        self.iuv_prefix = os.path.join(self.img_prefix, f'{uv_type}_IUV_gt')
        self.db = self._get_db(ann_file)

    def _get_db(self, ann_file):
        """Load dataset."""
        data = np.load(ann_file)
        tmpl = dict(
            image_file=None,
            center=None,
            scale=None,
            rotation=0,
            joints_2d=None,
            joints_2d_visible=None,
            joints_3d=None,
            joints_3d_visible=None,
            gender=None,
            pose=None,
            beta=None,
            has_smpl=0,
            iuv_file=None,
            has_iuv=0)
        gt_db = []

        _imgnames = data['imgname']
        _scales = data['scale'].astype(np.float32)
        _centers = data['center'].astype(np.float32)
        dataset_len = len(_imgnames)

        # Get 2D keypoints
        if 'part' in data.keys():
            _keypoints = data['part'].astype(np.float32)
        else:
            _keypoints = np.zeros((dataset_len, 24, 3), dtype=np.float32)

        # Get gt 3D joints, if available
        if 'S' in data.keys():
            _joints_3d = data['S'].astype(np.float32)
        else:
            _joints_3d = np.zeros((dataset_len, 24, 4), dtype=np.float32)

        # Get gt SMPL parameters, if available
        if 'pose' in data.keys() and 'shape' in data.keys():
            _poses = data['pose'].astype(np.float32)
            _betas = data['shape'].astype(np.float32)
            has_smpl = 1
        else:
            _poses = np.zeros((dataset_len, 72), dtype=np.float32)
            _betas = np.zeros((dataset_len, 10), dtype=np.float32)
            has_smpl = 0

        # Get gender data, if available
        if 'gender' in data.keys():
            _genders = data['gender']
            _genders = np.array([str(g) != 'm' for g in _genders]).astype(int)
        else:
            _genders = -1 * np.ones(dataset_len).astype(int)

        # Get IUV image, if available
        if 'iuv_names' in data.keys():
            _iuv_names = data['iuv_names']
            has_iuv = has_smpl
        else:
            _iuv_names = [''] * dataset_len
            has_iuv = 0

        for i in range(len(_imgnames)):
            newitem = cp.deepcopy(tmpl)
            newitem['image_file'] = os.path.join(self.img_prefix, _imgnames[i])
            newitem['scale'] = np.array([_scales[i], _scales[i]])
            newitem['center'] = _centers[i]
            newitem['joints_2d'] = _keypoints[i, :, :2]
            newitem['joints_2d_visible'] = _keypoints[i, :, -1][:, None]
            newitem['joints_3d'] = _joints_3d[i, :, :3]
            newitem['joints_3d_visible'] = _joints_3d[i, :, -1][:, None]
            newitem['pose'] = _poses[i]
            newitem['beta'] = _betas[i]
            newitem['has_smpl'] = has_smpl
            newitem['gender'] = _genders[i]
            newitem['iuv_file'] = os.path.join(self.iuv_prefix, _iuv_names[i])
            newitem['has_iuv'] = has_iuv
            gt_db.append(newitem)
        return gt_db

    def __len__(self, ):
        """Get the size of the dataset."""
        return len(self.db)

    def __getitem__(self, idx):
        """Get the sample given index."""
        results = cp.deepcopy(self.db[idx])
        results['ann_info'] = self.ann_info
        return self.pipeline(results)
