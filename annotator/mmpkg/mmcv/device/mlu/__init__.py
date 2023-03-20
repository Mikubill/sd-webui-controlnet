# Copyright (c) OpenMMLab. All rights reserved.
from .data_parallel import MLUDataParallel
from .distributed import MLUDistributedDataParallel

__all__ = ['MLUDataParallel', 'MLUDistributedDataParallel']
