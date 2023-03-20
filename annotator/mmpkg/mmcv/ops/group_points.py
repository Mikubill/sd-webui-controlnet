# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union

import torch
from torch import nn as nn
from torch.autograd import Function

from ..utils import ext_loader
from .ball_query import ball_query
from .knn import knn

ext_module = ext_loader.load_ext('_ext', [
    'group_points_forward', 'group_points_backward',
    'stack_group_points_forward', 'stack_group_points_backward'
])


class QueryAndGroup(nn.Module):
    """Groups points with a ball query of radius.

    Args:
        max_radius (float): The maximum radius of the balls.
            If None is given, we will use kNN sampling instead of ball query.
        sample_num (int): Maximum number of features to gather in the ball.
        min_radius (float, optional): The minimum radius of the balls.
            Default: 0.
        use_xyz (bool, optional): Whether to use xyz.
            Default: True.
        return_grouped_xyz (bool, optional): Whether to return grouped xyz.
            Default: False.
        normalize_xyz (bool, optional): Whether to normalize xyz.
            Default: False.
        uniform_sample (bool, optional): Whether to sample uniformly.
            Default: False
        return_unique_cnt (bool, optional): Whether to return the count of
            unique samples. Default: False.
        return_grouped_idx (bool, optional): Whether to return grouped idx.
            Default: False.
    """

    def __init__(self,
                 max_radius: float,
                 sample_num: int,
                 min_radius: float = 0.,
                 use_xyz: bool = True,
                 return_grouped_xyz: bool = False,
                 normalize_xyz: bool = False,
                 uniform_sample: bool = False,
                 return_unique_cnt: bool = False,
                 return_grouped_idx: bool = False):
        super().__init__()
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.sample_num = sample_num
        self.use_xyz = use_xyz
        self.return_grouped_xyz = return_grouped_xyz
        self.normalize_xyz = normalize_xyz
        self.uniform_sample = uniform_sample
        self.return_unique_cnt = return_unique_cnt
        self.return_grouped_idx = return_grouped_idx
        if self.return_unique_cnt:
            assert self.uniform_sample, \
                'uniform_sample should be True when ' \
                'returning the count of unique samples'
        if self.max_radius is None:
            assert not self.normalize_xyz, \
                'can not normalize grouped xyz when max_radius is None'

    def forward(
        self,
        points_xyz: torch.Tensor,
        center_xyz: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple]:
        """
        Args:
            points_xyz (torch.Tensor): (B, N, 3) xyz coordinates of the
                points.
            center_xyz (torch.Tensor): (B, npoint, 3) coordinates of the
                centriods.
            features (torch.Tensor): (B, C, N) The features of grouped
                points.

        Returns:
            Tuple | torch.Tensor: (B, 3 + C, npoint, sample_num) Grouped
            concatenated coordinates and features of points.
        """
        # if self.max_radius is None, we will perform kNN instead of ball query
        # idx is of shape [B, npoint, sample_num]
        if self.max_radius is None:
            idx = knn(self.sample_num, points_xyz, center_xyz, False)
            idx = idx.transpose(1, 2).contiguous()
        else:
            idx = ball_query(self.min_radius, self.max_radius, self.sample_num,
                             points_xyz, center_xyz)

        if self.uniform_sample:
            unique_cnt = torch.zeros((idx.shape[0], idx.shape[1]))
            for i_batch in range(idx.shape[0]):
                for i_region in range(idx.shape[1]):
                    unique_ind = torch.unique(idx[i_batch, i_region, :])
                    num_unique = unique_ind.shape[0]
                    unique_cnt[i_batch, i_region] = num_unique
                    sample_ind = torch.randint(
                        0,
                        num_unique, (self.sample_num - num_unique, ),
                        dtype=torch.long)
                    all_ind = torch.cat((unique_ind, unique_ind[sample_ind]))
                    idx[i_batch, i_region, :] = all_ind

        xyz_trans = points_xyz.transpose(1, 2).contiguous()
        # (B, 3, npoint, sample_num)
        grouped_xyz = grouping_operation(xyz_trans, idx)
        grouped_xyz_diff = grouped_xyz - \
            center_xyz.transpose(1, 2).unsqueeze(-1)  # relative offsets
        if self.normalize_xyz:
            grouped_xyz_diff /= self.max_radius

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                # (B, C + 3, npoint, sample_num)
                new_features = torch.cat([grouped_xyz_diff, grouped_features],
                                         dim=1)
            else:
                new_features = grouped_features
        else:
            assert (self.use_xyz
                    ), 'Cannot have not features and not use xyz as a feature!'
            new_features = grouped_xyz_diff

        ret = [new_features]
        if self.return_grouped_xyz:
            ret.append(grouped_xyz)
        if self.return_unique_cnt:
            ret.append(unique_cnt)
        if self.return_grouped_idx:
            ret.append(idx)
        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)


class GroupAll(nn.Module):
    """Group xyz with feature.

    Args:
        use_xyz (bool): Whether to use xyz.
    """

    def __init__(self, use_xyz: bool = True):
        super().__init__()
        self.use_xyz = use_xyz

    def forward(self,
                xyz: torch.Tensor,
                new_xyz: torch.Tensor,
                features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            xyz (Tensor): (B, N, 3) xyz coordinates of the features.
            new_xyz (Tensor): new xyz coordinates of the features.
            features (Tensor): (B, C, N) features to group.

        Returns:
            Tensor: (B, C + 3, 1, N) Grouped feature.
        """
        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                # (B, 3 + C, 1, N)
                new_features = torch.cat([grouped_xyz, grouped_features],
                                         dim=1)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        return new_features


class GroupingOperation(Function):
    """Group feature with given index."""

    @staticmethod
    def forward(
            ctx,
            features: torch.Tensor,
            indices: torch.Tensor,
            features_batch_cnt: Optional[torch.Tensor] = None,
            indices_batch_cnt: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            features (Tensor): Tensor of features to group, input shape is
                (B, C, N) or stacked inputs (N1 + N2 ..., C).
            indices (Tensor):  The indices of features to group with, input
                shape is (B, npoint, nsample) or stacked inputs
                (M1 + M2 ..., nsample).
            features_batch_cnt (Tensor, optional): Input features nums in
                each batch, just like (N1, N2, ...). Defaults to None.
                New in version 1.7.0.
            indices_batch_cnt (Tensor, optional): Input indices nums in
                each batch, just like (M1, M2, ...). Defaults to None.
                New in version 1.7.0.

        Returns:
            Tensor: Grouped features, the shape is (B, C, npoint, nsample)
            or (M1 + M2 ..., C, nsample).
        """
        features = features.contiguous()
        indices = indices.contiguous()
        if features_batch_cnt is not None and indices_batch_cnt is not None:
            assert features_batch_cnt.dtype == torch.int
            assert indices_batch_cnt.dtype == torch.int
            M, nsample = indices.size()
            N, C = features.size()
            B = indices_batch_cnt.shape[0]
            output = features.new_zeros((M, C, nsample))
            ext_module.stack_group_points_forward(
                features,
                features_batch_cnt,
                indices,
                indices_batch_cnt,
                output,
                b=B,
                m=M,
                c=C,
                nsample=nsample)
            ctx.for_backwards = (B, N, indices, features_batch_cnt,
                                 indices_batch_cnt)
        else:
            B, nfeatures, nsample = indices.size()
            _, C, N = features.size()
            output = features.new_zeros(B, C, nfeatures, nsample)

            ext_module.group_points_forward(
                features,
                indices,
                output,
                b=B,
                c=C,
                n=N,
                npoints=nfeatures,
                nsample=nsample)

            ctx.for_backwards = (indices, N)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple:
        """
        Args:
            grad_out (Tensor): (B, C, npoint, nsample) tensor of the gradients
                of the output from forward.

        Returns:
            Tensor: (B, C, N) gradient of the features.
        """
        if len(ctx.for_backwards) != 5:
            idx, N = ctx.for_backwards

            B, C, npoint, nsample = grad_out.size()
            grad_features = grad_out.new_zeros(B, C, N)

            grad_out_data = grad_out.data.contiguous()
            ext_module.group_points_backward(
                grad_out_data,
                idx,
                grad_features.data,
                b=B,
                c=C,
                n=N,
                npoints=npoint,
                nsample=nsample)
            return grad_features, None
        else:
            B, N, idx, features_batch_cnt, idx_batch_cnt = ctx.for_backwards

            M, C, nsample = grad_out.size()
            grad_features = grad_out.new_zeros(N, C)

            grad_out_data = grad_out.data.contiguous()
            ext_module.stack_group_points_backward(
                grad_out_data,
                idx,
                idx_batch_cnt,
                features_batch_cnt,
                grad_features.data,
                b=B,
                c=C,
                m=M,
                n=N,
                nsample=nsample)
            return grad_features, None, None, None


grouping_operation = GroupingOperation.apply
