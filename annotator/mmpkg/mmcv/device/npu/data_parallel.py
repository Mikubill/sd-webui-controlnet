# Copyright Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.

import sys

import torch

from annotator.mmpkg.mmcv.device.scatter_gather import scatter_kwargs
from annotator.mmpkg.mmcv.parallel import MMDataParallel


def _check_balance(*args, **kwargs):
    return


# Since we do not have a similar hardware unit multi_processor
# on the NPU, the corresponding# devices_properties does not
# have this property and cannot be checked. So we masked the
# _check_balance function in DataParallel to make initialization pass.
for m in sys.modules:
    if m.startswith('torch') or 'mmcv' in m:
        if hasattr(sys.modules[m], '_check_balance'):
            setattr(sys.modules[m], '_check_balance', _check_balance)


class NPUDataParallel(MMDataParallel):
    """The NPUDataParallel module that supports DataContainer.

    NPUDataParallel is a class inherited from MMDataParall, which supports
    NPU training and inference only.

    The main differences with MMDataParallel:

    - It only supports single-card of NPU, and only use first card to
      run training and inference.

    - It uses direct host-to-device copy instead of stream-background
      scatter.

    .. warning::
        NPUDataParallel only supports single NPU training, if you need to
        train with multiple NPUs, please use NPUDistributedDataParallel
        instead. If you have multiple NPUs, you can toggle device_ids
        parameters passed in for this function to specify the running device.

    Args:
        module (:class:`nn.Module`): Module to be encapsulated.
        dim (int): Dimension used to scatter the data. Defaults to 0.
    """

    def __init__(self, *args, dim=0, **kwargs):
        super().__init__(*args, dim=dim, **kwargs)
        device_id = kwargs.get('device_ids', [0])[0]
        self.device_ids = [device_id]
        self.src_device_obj = torch.device(f'npu:{device_id}')
        torch.npu.set_device(self.src_device_obj)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)
