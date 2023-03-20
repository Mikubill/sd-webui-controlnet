# Copyright (c) OpenMMLab. All rights reserved.
import torch

from .parrots_wrapper import TORCH_VERSION
from .version_utils import digit_version

_torch_version_meshgrid_indexing = (
    'parrots' not in TORCH_VERSION
    and digit_version(TORCH_VERSION) >= digit_version('1.10.0a0'))


def torch_meshgrid(*tensors):
    """A wrapper of torch.meshgrid to compat different PyTorch versions.

    Since PyTorch 1.10.0a0, torch.meshgrid supports the arguments ``indexing``.
    So we implement a wrapper here to avoid warning when using high-version
    PyTorch and avoid compatibility issues when using previous versions of
    PyTorch.

    Args:
        tensors (List[Tensor]): List of scalars or 1 dimensional tensors.

    Returns:
        Sequence[Tensor]: Sequence of meshgrid tensors.
    """
    if _torch_version_meshgrid_indexing:
        return torch.meshgrid(*tensors, indexing='ij')
    else:
        return torch.meshgrid(*tensors)  # Uses indexing='ij' by default
