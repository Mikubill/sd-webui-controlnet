# Copyright (c) OpenMMLab. All rights reserved.
import copy as cp
from abc import ABCMeta

import numpy as np
from torch.utils.data import Dataset

from annotator.mmpkg.mmpose.datasets.builder import DATASETS
from annotator.mmpkg.mmpose.datasets.pipelines import Compose


@DATASETS.register_module()
class MoshDataset(Dataset, metaclass=ABCMeta):
    """Mosh Dataset for the adversarial training in 3D human mesh estimation
    task.

    The dataset return a dict containing real-world SMPL parameters.

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    def __init__(self, ann_file, pipeline, test_mode=False):

        self.ann_file = ann_file
        self.pipeline = pipeline
        self.test_mode = test_mode

        self.db = self._get_db(ann_file)
        self.pipeline = Compose(self.pipeline)

    @staticmethod
    def _get_db(ann_file):
        """Load dataset."""
        data = np.load(ann_file)
        _betas = data['shape'].astype(np.float32)
        _poses = data['pose'].astype(np.float32)
        tmpl = dict(
            pose=None,
            beta=None,
        )
        gt_db = []
        dataset_len = len(_betas)

        for i in range(dataset_len):
            newitem = cp.deepcopy(tmpl)
            newitem['pose'] = _poses[i]
            newitem['beta'] = _betas[i]
            gt_db.append(newitem)
        return gt_db

    def __len__(self, ):
        """Get the size of the dataset."""
        return len(self.db)

    def __getitem__(self, idx):
        """Get the sample given index."""
        item = cp.deepcopy(self.db[idx])
        trivial, pose, beta = \
            np.zeros(3, dtype=np.float32), item['pose'], item['beta']
        results = {
            'mosh_theta':
            np.concatenate((trivial, pose, beta), axis=0).astype(np.float32)
        }
        return self.pipeline(results)
