# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union

import torch


def scatter(input: Union[List, torch.Tensor], devices: List) -> List:
    """scatter copies tensor to MLU directly."""
    if isinstance(input, list):
        outputs = [scatter(_input, devices) for _input in input]
        return outputs
    elif isinstance(input, torch.Tensor):
        output = input.contiguous()
        return output.to('mlu') if devices != [-1] else output
    else:
        raise Exception(f'Unknown type {type(input)}.')


class Scatter:

    @staticmethod
    def forward(target_mlus, input):
        outputs = scatter(input, target_mlus)
        return tuple(outputs) if isinstance(outputs, list) else (outputs, )
