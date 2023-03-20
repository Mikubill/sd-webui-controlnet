# Copyright (c) OpenMMLab. All rights reserved.
import torch
from packaging import version

_torch_version_meshgrid_indexing = version.parse(
    torch.__version__) >= version.parse('1.10.0a0')


def torch_meshgrid_ij(*tensors):
    if _torch_version_meshgrid_indexing:
        return torch.meshgrid(*tensors, indexing='ij')
    else:
        return torch.meshgrid(*tensors)  # Uses indexing='ij' by default
