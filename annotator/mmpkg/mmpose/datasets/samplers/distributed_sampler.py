# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.utils.data import DistributedSampler as _DistributedSampler

from annotator.mmpkg.mmpose.core import sync_random_seed


class DistributedSampler(_DistributedSampler):
    """DistributedSampler inheriting from
    `torch.utils.data.DistributedSampler`.

    In pytorch of lower versions, there is no `shuffle` argument. This child
    class will port one to DistributedSampler.
    """

    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 seed=0):
        super().__init__(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        # for the compatibility from PyTorch 1.3+
        # In distributed sampling, different ranks should sample non-overlapped
        # data in the dataset. Therefore, this function is used to make sure
        # that each rank shuffles the data indices in the same order based
        # on the same seed. Then different ranks could use different indices
        # to select non-overlapped data from the same data list.
        self.seed = sync_random_seed(seed) if seed is not None else 0

    def __iter__(self):
        """Deterministically shuffle based on epoch."""
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch + self.seed)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)
