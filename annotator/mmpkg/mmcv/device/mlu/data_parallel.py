# Copyright (c) OpenMMLab. All rights reserved.

import torch

from annotator.mmpkg.mmcv.parallel import MMDataParallel
from .scatter_gather import scatter_kwargs


class MLUDataParallel(MMDataParallel):
    """The MLUDataParallel module that supports DataContainer.

    MLUDataParallel is a class inherited from MMDataParall, which supports
    MLU training and inference only.

    The main differences with MMDataParallel:

    - It only supports single-card of MLU, and only use first card to
      run training and inference.

    - It uses direct host-to-device copy instead of stream-background
      scatter.

    .. warning::
        MLUDataParallel only supports single MLU training, if you need to
        train with multiple MLUs, please use MLUDistributedDataParallel
        instead. If you have multiple MLUs, you can set the environment
        variable ``MLU_VISIBLE_DEVICES=0`` (or any other card number(s))
        to specify the running device.

    Args:
        module (:class:`nn.Module`): Module to be encapsulated.
        dim (int): Dimension used to scatter the data. Defaults to 0.
    """

    def __init__(self, *args, dim=0, **kwargs):
        super().__init__(*args, dim=dim, **kwargs)
        self.device_ids = [0]
        self.src_device_obj = torch.device('mlu:0')

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)
