# Copyright (c) OpenMMLab. All rights reserved.
import torch

from annotator.mmpkg.mmcv.parallel.data_container import DataContainer
from annotator.mmpkg.mmcv.utils import deprecated_api_warning
from ._functions import Scatter
from .utils import get_device


@deprecated_api_warning({'target_mlus': 'target_devices'})
def scatter(inputs, target_devices, dim=0):
    """Scatter inputs to target devices.

    The only difference from original :func:`scatter` is to add support for
    :type:`~mmcv.parallel.DataContainer`.
    """
    current_device = get_device()

    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            if target_devices != [-1]:
                obj = obj.to(current_device)
                return [obj]
            else:
                # for CPU inference we use self-implemented scatter
                return Scatter.forward(target_devices, obj)
        if isinstance(obj, DataContainer):
            if obj.cpu_only:
                return obj.data
            else:
                return Scatter.forward(target_devices, obj.data)
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            out = list(map(list, zip(*map(scatter_map, obj))))
            return out
        if isinstance(obj, dict) and len(obj) > 0:
            out = list(map(type(obj), zip(*map(scatter_map, obj.items()))))
            return out
        return [obj for _ in target_devices]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None


@deprecated_api_warning({'target_mlus': 'target_devices'})
def scatter_kwargs(inputs, kwargs, target_devices, dim=0):
    """Scatter with support for kwargs dictionary."""
    inputs = scatter(inputs, target_devices, dim) if inputs else []
    kwargs = scatter(kwargs, target_devices, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs
