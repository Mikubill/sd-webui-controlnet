# Copyright (c) OpenMMLab. All rights reserved.

from annotator.mmpkg.mmcv.parallel import MMDistributedDataParallel
from .scatter_gather import scatter_kwargs


class MLUDistributedDataParallel(MMDistributedDataParallel):
    """The DDP module supports DataContainer.

    MLUDDP has one difference from MMDDP which moves data to MLU with coping
    instead of scattering.
    """

    def to_kwargs(self, inputs, kwargs, device_id):
        # Use `self.to_kwargs` instead of `self.scatter` in pytorch1.8
        # to move all tensors to device_id
        return scatter_kwargs(inputs, kwargs, [device_id], dim=self.dim)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)
