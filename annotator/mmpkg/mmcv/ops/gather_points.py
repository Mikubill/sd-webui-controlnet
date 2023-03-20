from typing import Tuple

import torch
from torch.autograd import Function

from ..utils import ext_loader

ext_module = ext_loader.load_ext(
    '_ext', ['gather_points_forward', 'gather_points_backward'])


class GatherPoints(Function):
    """Gather points with given index."""

    @staticmethod
    def forward(ctx, features: torch.Tensor,
                indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): (B, C, N) features to gather.
            indices (torch.Tensor): (B, M) where M is the number of points.

        Returns:
            torch.Tensor: (B, C, M) where M is the number of points.
        """
        assert features.is_contiguous()
        assert indices.is_contiguous()

        B, npoint = indices.size()
        _, C, N = features.size()
        output = features.new_zeros((B, C, npoint))

        ext_module.gather_points_forward(
            features, indices, output, b=B, c=C, n=N, npoints=npoint)

        ctx.for_backwards = (indices, C, N)
        if torch.__version__ != 'parrots':
            ctx.mark_non_differentiable(indices)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, None]:
        idx, C, N = ctx.for_backwards
        B, npoint = idx.size()

        grad_features = grad_out.new_zeros((B, C, N))
        grad_out_data = grad_out.data.contiguous()
        ext_module.gather_points_backward(
            grad_out_data,
            idx,
            grad_features.data,
            b=B,
            c=C,
            n=N,
            npoints=npoint)
        return grad_features, None


gather_points = GatherPoints.apply
