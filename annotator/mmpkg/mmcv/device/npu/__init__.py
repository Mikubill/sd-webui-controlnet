# Copyright Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.
from .data_parallel import NPUDataParallel
from .distributed import NPUDistributedDataParallel

__all__ = ['NPUDataParallel', 'NPUDistributedDataParallel']
