# Copyright Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.

from annotator.mmpkg.mmcv.device.scatter_gather import scatter_kwargs
from annotator.mmpkg.mmcv.parallel import MMDistributedDataParallel


class NPUDistributedDataParallel(MMDistributedDataParallel):
    """The DDP module supports DataContainer.

    NPUDDP has one difference from MMDDP which moves data to NPU with coping
    instead of scattering.
    """

    def to_kwargs(self, inputs, kwargs, device_id):
        # Use `self.to_kwargs` instead of `self.scatter` in pytorch1.8
        # to move all tensors to device_id
        return scatter_kwargs(inputs, kwargs, [device_id], dim=self.dim)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def forward(self, *inputs, **kwargs):
        # Since the scatter method is not supported on the NPU
        # and the DDP class is rewritten, when the forward of DDP
        # is used, the NPU will mask the scatter branch,
        # resulting in the input not being placed on the device side.
        # So, forward has been rewritten here primarily to circumvent
        # this situation that would cause the device misalignment.
        if self.device_ids:
            inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
            return super().forward(*inputs[0], **kwargs[0])
        return super().forward(*inputs, **kwargs)
