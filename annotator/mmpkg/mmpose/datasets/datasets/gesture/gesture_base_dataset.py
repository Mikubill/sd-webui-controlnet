# Copyright (c) OpenMMLab. All rights reserved.
import copy
from abc import ABCMeta, abstractmethod

import numpy as np
from torch.utils.data import Dataset

from annotator.mmpkg.mmpose.datasets.pipelines import Compose


class GestureBaseDataset(Dataset, metaclass=ABCMeta):
    """Base class for gesture recognition datasets with Multi-Modal video as
    the input.

    All gesture datasets should subclass it.
    All subclasses should overwrite:
        Methods:`_get_single`, 'evaluate'

    Args:
        ann_file (str): Path to the annotation file.
        vid_prefix (str): Path to a directory where videos are held.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        dataset_info (DatasetInfo): A class containing all dataset info.
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    def __init__(self,
                 ann_file,
                 vid_prefix,
                 data_cfg,
                 pipeline,
                 dataset_info=None,
                 test_mode=False):

        self.video_info = {}
        self.ann_info = {}

        self.ann_file = ann_file
        self.vid_prefix = vid_prefix
        self.pipeline = pipeline
        self.test_mode = test_mode

        self.ann_info['video_size'] = np.array(data_cfg['video_size'])
        self.ann_info['flip_pairs'] = dataset_info.flip_pairs
        self.modality = data_cfg['modality']
        if isinstance(self.modality, (list, tuple)):
            self.modality = self.modality
        else:
            self.modality = (self.modality, )
        self.bbox_file = data_cfg.get('bbox_file', None)
        self.dataset_name = dataset_info.dataset_name
        self.pipeline = Compose(self.pipeline)

    @abstractmethod
    def _get_single(self, idx):
        """Get anno for a single video."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, results, *args, **kwargs):
        """Evaluate recognition results."""

    def prepare_train_vid(self, idx):
        """Prepare video for training given the index."""
        results = copy.deepcopy(self._get_single(idx))
        results['ann_info'] = self.ann_info
        return self.pipeline(results)

    def prepare_test_vid(self, idx):
        """Prepare video for testing given the index."""
        results = copy.deepcopy(self._get_single(idx))
        results['ann_info'] = self.ann_info
        return self.pipeline(results)

    def __len__(self):
        """Get dataset length."""
        return len(self.vid_ids)

    def __getitem__(self, idx):
        """Get the sample for either training or testing given index."""
        if self.test_mode:
            return self.prepare_test_vid(idx)

        return self.prepare_train_vid(idx)
