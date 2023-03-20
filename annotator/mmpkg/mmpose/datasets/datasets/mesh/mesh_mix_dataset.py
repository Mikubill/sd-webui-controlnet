# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta

import numpy as np
from torch.utils.data import ConcatDataset, Dataset, WeightedRandomSampler

from annotator.mmpkg.mmpose.datasets.builder import DATASETS
from .mesh_base_dataset import MeshBaseDataset


@DATASETS.register_module()
class MeshMixDataset(Dataset, metaclass=ABCMeta):
    """Mix Dataset for 3D human mesh estimation.

    The dataset combines data from multiple datasets (MeshBaseDataset) and
    sample the data from different datasets with the provided proportions.
    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    Args:
        configs (list): List of configs for multiple datasets.
        partition (list): Sample proportion of multiple datasets. The length
            of partition should be same with that of configs. The elements
            of it should be non-negative and is not necessary summing up to
            one.

    Example:
        >>> from annotator.mmpkg.mmpose.datasets import MeshMixDataset
        >>> data_cfg = dict(
        >>>     image_size=[256, 256],
        >>>     iuv_size=[64, 64],
        >>>     num_joints=24,
        >>>     use_IUV=True,
        >>>     uv_type='BF')
        >>>
        >>> mix_dataset = MeshMixDataset(
        >>>     configs=[
        >>>         dict(
        >>>             ann_file='tests/data/h36m/test_h36m.npz',
        >>>             img_prefix='tests/data/h36m',
        >>>             data_cfg=data_cfg,
        >>>             pipeline=[]),
        >>>         dict(
        >>>             ann_file='tests/data/h36m/test_h36m.npz',
        >>>             img_prefix='tests/data/h36m',
        >>>             data_cfg=data_cfg,
        >>>             pipeline=[]),
        >>>     ],
        >>>     partition=[0.6, 0.4])
    """

    def __init__(self, configs, partition):
        """Load data from multiple datasets."""
        assert min(partition) >= 0
        datasets = [MeshBaseDataset(**cfg) for cfg in configs]
        self.dataset = ConcatDataset(datasets)
        self.length = max(len(ds) for ds in datasets)
        weights = [
            np.ones(len(ds)) * p / len(ds)
            for (p, ds) in zip(partition, datasets)
        ]
        weights = np.concatenate(weights, axis=0)
        self.sampler = WeightedRandomSampler(weights, 1)

    def __len__(self):
        """Get the size of the dataset."""
        return self.length

    def __getitem__(self, idx):
        """Given index, sample the data from multiple datasets with the given
        proportion."""
        idx_new = list(self.sampler)[0]
        return self.dataset[idx_new]
