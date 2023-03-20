# Copyright (c) OpenMMLab. All rights reserved.
from annotator.mmpkg.mmcv.utils import (IS_CUDA_AVAILABLE, IS_MLU_AVAILABLE, IS_MPS_AVAILABLE,
                        IS_NPU_AVAILABLE)


def get_device() -> str:
    """Returns the currently existing device type.

    .. note::
        Since npu provides tools to automatically convert cuda functions,
        we need to make judgments on npu first to avoid entering
        the cuda branch when using npu.

    Returns:
        str: cuda | mlu | mps | cpu.
    """
    if IS_NPU_AVAILABLE:
        return 'npu'
    elif IS_CUDA_AVAILABLE:
        return 'cuda'
    elif IS_MLU_AVAILABLE:
        return 'mlu'
    elif IS_MPS_AVAILABLE:
        return 'mps'
    else:
        return 'cpu'
