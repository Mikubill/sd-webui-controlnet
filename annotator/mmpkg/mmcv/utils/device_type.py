# Copyright (c) OpenMMLab. All rights reserved.


def is_ipu_available() -> bool:
    try:
        import poptorch
        return poptorch.ipuHardwareIsAvailable()
    except ImportError:
        return False


IS_IPU_AVAILABLE = is_ipu_available()


def is_mlu_available() -> bool:
    try:
        import torch
        return (hasattr(torch, 'is_mlu_available')
                and torch.is_mlu_available())
    except Exception:
        return False


IS_MLU_AVAILABLE = is_mlu_available()


def is_mps_available() -> bool:
    """Return True if mps devices exist.

    It's specialized for mac m1 chips and require torch version 1.12 or higher.
    """
    try:
        import torch
        return hasattr(torch.backends,
                       'mps') and torch.backends.mps.is_available()
    except Exception:
        return False


IS_MPS_AVAILABLE = is_mps_available()


def is_npu_available() -> bool:
    """Return True if npu devices exist."""
    try:
        import torch
        import torch_npu
        return (hasattr(torch, 'npu') and torch_npu.npu.is_available())
    except Exception:
        return False


IS_NPU_AVAILABLE = is_npu_available()
