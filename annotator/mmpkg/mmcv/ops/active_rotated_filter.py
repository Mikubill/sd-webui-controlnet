# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from ..utils import ext_loader

ext_module = ext_loader.load_ext(
    '_ext',
    ['active_rotated_filter_forward', 'active_rotated_filter_backward'])


class ActiveRotatedFilterFunction(Function):
    """Encoding the orientation information and generating orientation-
    sensitive features.

    The details are described in the paper `Align Deep Features for Oriented
    Object Detection  <https://arxiv.org/abs/2008.09397>_`.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor,
                indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): Input features with shape
                [num_output_planes, num_input_planes, num_orientations, H, W].
            indices (torch.Tensor): Indices with shape
                [num_orientations, H, W, num_rotations].

        Returns:
            torch.Tensor: Refined features with shape [num_output_planes *
            num_rotations, num_input_planes * num_orientations, H, W].
        """
        ctx.save_for_backward(input, indices)
        op, ip, o, h, w = input.size()
        o, h, w, r = indices.size()
        output = input.new_zeros((op * r, ip * o, h, w))
        ext_module.active_rotated_filter_forward(input, indices, output)

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Args:
            grad_output (torch.Tensor): The gradient of output features
                with shape [num_output_planes * num_rotations,
                num_input_planes * num_orientations, H, W].

        Returns:
            torch.Tensor: The gradient of input features with shape
            [num_output_planes, num_input_planes, num_orientations, H, W].
        """
        input, indices = ctx.saved_tensors
        grad_in = torch.zeros_like(input)
        ext_module.active_rotated_filter_backward(grad_out, indices, grad_in)
        return grad_in, None


active_rotated_filter = ActiveRotatedFilterFunction.apply
