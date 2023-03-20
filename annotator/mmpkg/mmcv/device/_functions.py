# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union

import torch

from annotator.mmpkg.mmcv.utils import deprecated_api_warning
from .utils import get_device


def scatter(input: Union[List, torch.Tensor], devices: List) -> List:
    """scatter copies tensor to devices directly."""
    current_device = get_device()
    if isinstance(input, list):
        outputs = [scatter(_input, devices) for _input in input]
        return outputs
    elif isinstance(input, torch.Tensor):
        output = input.contiguous()
        return output.to(current_device) if devices != [-1] else output
    else:
        raise Exception(f'Unknown type {type(input)}.')


class Scatter:

    @staticmethod
    @deprecated_api_warning({'target_mlus': 'target_devices'},
                            cls_name='Scatter')
    def forward(target_devices, input):
        outputs = scatter(input, target_devices)
        return tuple(outputs) if isinstance(outputs, list) else (outputs, )
