# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', [
    'sigmoid_focal_loss_forward', 'sigmoid_focal_loss_backward',
    'softmax_focal_loss_forward', 'softmax_focal_loss_backward'
])


class SigmoidFocalLossFunction(Function):

    @staticmethod
    def symbolic(g, input: torch.Tensor, target: torch.LongTensor,
                 gamma: float, alpha: float, weight: torch.Tensor,
                 reduction: str):
        return g.op(
            'mmcv::MMCVSigmoidFocalLoss',
            input,
            target,
            gamma_f=gamma,
            alpha_f=alpha,
            weight_f=weight,
            reduction_s=reduction)

    @staticmethod
    def forward(ctx,
                input: torch.Tensor,
                target: Union[torch.LongTensor, torch.cuda.LongTensor],
                gamma: float = 2.0,
                alpha: float = 0.25,
                weight: Optional[torch.Tensor] = None,
                reduction: str = 'mean') -> torch.Tensor:

        assert target.dtype == torch.long
        assert input.dim() == 2
        assert target.dim() == 1
        assert input.size(0) == target.size(0)
        if weight is None:
            weight = input.new_empty(0)
        else:
            assert weight.dim() == 1
            assert input.size(1) == weight.size(0)
        ctx.reduction_dict = {'none': 0, 'mean': 1, 'sum': 2}
        assert reduction in ctx.reduction_dict.keys()

        ctx.gamma = float(gamma)
        ctx.alpha = float(alpha)
        ctx.reduction = ctx.reduction_dict[reduction]

        output = input.new_zeros(input.size())

        ext_module.sigmoid_focal_loss_forward(
            input, target, weight, output, gamma=ctx.gamma, alpha=ctx.alpha)
        if ctx.reduction == ctx.reduction_dict['mean']:
            output = output.sum() / input.size(0)
        elif ctx.reduction == ctx.reduction_dict['sum']:
            output = output.sum()
        ctx.save_for_backward(input, target, weight)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        input, target, weight = ctx.saved_tensors

        grad_input = input.new_zeros(input.size())

        ext_module.sigmoid_focal_loss_backward(
            input,
            target,
            weight,
            grad_input,
            gamma=ctx.gamma,
            alpha=ctx.alpha)

        grad_input *= grad_output
        if ctx.reduction == ctx.reduction_dict['mean']:
            grad_input /= input.size(0)
        return grad_input, None, None, None, None, None


sigmoid_focal_loss = SigmoidFocalLossFunction.apply


class SigmoidFocalLoss(nn.Module):

    def __init__(self,
                 gamma: float,
                 alpha: float,
                 weight: Optional[torch.Tensor] = None,
                 reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.register_buffer('weight', weight)
        self.reduction = reduction

    def forward(
        self,
        input: torch.Tensor,
        target: Union[torch.LongTensor, torch.cuda.LongTensor],
    ) -> torch.Tensor:
        return sigmoid_focal_loss(input, target, self.gamma, self.alpha,
                                  self.weight, self.reduction)

    def __repr__(self):
        s = self.__class__.__name__
        s += f'(gamma={self.gamma}, '
        s += f'alpha={self.alpha}, '
        s += f'reduction={self.reduction})'
        return s


class SoftmaxFocalLossFunction(Function):

    @staticmethod
    def symbolic(g, input: torch.Tensor, target: torch.LongTensor,
                 gamma: float, alpha: float, weight: torch.Tensor,
                 reduction: str):
        return g.op(
            'mmcv::MMCVSoftmaxFocalLoss',
            input,
            target,
            gamma_f=gamma,
            alpha_f=alpha,
            weight_f=weight,
            reduction_s=reduction)

    @staticmethod
    def forward(ctx,
                input: torch.Tensor,
                target: Union[torch.LongTensor, torch.cuda.LongTensor],
                gamma: float = 2.0,
                alpha: float = 0.25,
                weight: Optional[torch.Tensor] = None,
                reduction='mean') -> torch.Tensor:

        assert target.dtype == torch.long
        assert input.dim() == 2
        assert target.dim() == 1
        assert input.size(0) == target.size(0)
        if weight is None:
            weight = input.new_empty(0)
        else:
            assert weight.dim() == 1
            assert input.size(1) == weight.size(0)
        ctx.reduction_dict = {'none': 0, 'mean': 1, 'sum': 2}
        assert reduction in ctx.reduction_dict.keys()

        ctx.gamma = float(gamma)
        ctx.alpha = float(alpha)
        ctx.reduction = ctx.reduction_dict[reduction]

        channel_stats, _ = torch.max(input, dim=1)
        input_softmax = input - channel_stats.unsqueeze(1).expand_as(input)
        input_softmax.exp_()

        channel_stats = input_softmax.sum(dim=1)
        input_softmax /= channel_stats.unsqueeze(1).expand_as(input)

        output = input.new_zeros(input.size(0))
        ext_module.softmax_focal_loss_forward(
            input_softmax,
            target,
            weight,
            output,
            gamma=ctx.gamma,
            alpha=ctx.alpha)

        if ctx.reduction == ctx.reduction_dict['mean']:
            output = output.sum() / input.size(0)
        elif ctx.reduction == ctx.reduction_dict['sum']:
            output = output.sum()
        ctx.save_for_backward(input_softmax, target, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        input_softmax, target, weight = ctx.saved_tensors
        buff = input_softmax.new_zeros(input_softmax.size(0))
        grad_input = input_softmax.new_zeros(input_softmax.size())

        ext_module.softmax_focal_loss_backward(
            input_softmax,
            target,
            weight,
            buff,
            grad_input,
            gamma=ctx.gamma,
            alpha=ctx.alpha)

        grad_input *= grad_output
        if ctx.reduction == ctx.reduction_dict['mean']:
            grad_input /= input_softmax.size(0)
        return grad_input, None, None, None, None, None


softmax_focal_loss = SoftmaxFocalLossFunction.apply


class SoftmaxFocalLoss(nn.Module):

    def __init__(self,
                 gamma: float,
                 alpha: float,
                 weight: Optional[torch.Tensor] = None,
                 reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.register_buffer('weight', weight)
        self.reduction = reduction

    def forward(
        self,
        input: torch.Tensor,
        target: Union[torch.LongTensor, torch.cuda.LongTensor],
    ) -> torch.Tensor:
        return softmax_focal_loss(input, target, self.gamma, self.alpha,
                                  self.weight, self.reduction)

    def __repr__(self):
        s = self.__class__.__name__
        s += f'(gamma={self.gamma}, '
        s += f'alpha={self.alpha}, '
        s += f'reduction={self.reduction})'
        return s
