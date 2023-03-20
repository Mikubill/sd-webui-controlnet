# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, List, Tuple, Union

import torch
from torch import nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair

from ..utils import ext_loader

ext_module = ext_loader.load_ext(
    '_ext', ['dynamic_voxelize_forward', 'hard_voxelize_forward'])


class _Voxelization(Function):

    @staticmethod
    def forward(
            ctx: Any,
            points: torch.Tensor,
            voxel_size: Union[tuple, float],
            coors_range: Union[tuple, float],
            max_points: int = 35,
            max_voxels: int = 20000,
            deterministic: bool = True) -> Union[Tuple[torch.Tensor], Tuple]:
        """Convert kitti points(N, >=3) to voxels.

        Args:
            points (torch.Tensor): [N, ndim]. Points[:, :3] contain xyz points
                and points[:, 3:] contain other information like reflectivity.
            voxel_size (tuple or float): The size of voxel with the shape of
                [3].
            coors_range (tuple or float): The coordinate range of voxel with
                the shape of [6].
            max_points (int, optional): maximum points contained in a voxel. if
                max_points=-1, it means using dynamic_voxelize. Default: 35.
            max_voxels (int, optional): maximum voxels this function create.
                for second, 20000 is a good choice. Users should shuffle points
                before call this function because max_voxels may drop points.
                Default: 20000.
            deterministic: bool. whether to invoke the non-deterministic
                version of hard-voxelization implementations. non-deterministic
                version is considerablly fast but is not deterministic. only
                affects hard voxelization. default True. for more information
                of this argument and the implementation insights, please refer
                to the following links:
                https://github.com/open-mmlab/mmdetection3d/issues/894
                https://github.com/open-mmlab/mmdetection3d/pull/904
                it is an experimental feature and we will appreciate it if
                you could share with us the failing cases.

        Returns:
            tuple[torch.Tensor]: tuple[torch.Tensor]: A tuple contains three
            elements. The first one is the output voxels with the shape of
            [M, max_points, n_dim], which only contain points and returned
            when max_points != -1. The second is the voxel coordinates with
            shape of [M, 3]. The last is number of point per voxel with the
            shape of [M], which only returned when max_points != -1.
        """
        if max_points == -1 or max_voxels == -1:
            coors = points.new_zeros(size=(points.size(0), 3), dtype=torch.int)
            ext_module.dynamic_voxelize_forward(
                points,
                torch.tensor(voxel_size, dtype=torch.float),
                torch.tensor(coors_range, dtype=torch.float),
                coors,
                NDim=3)
            return coors
        else:
            voxels = points.new_zeros(
                size=(max_voxels, max_points, points.size(1)))
            coors = points.new_zeros(size=(max_voxels, 3), dtype=torch.int)
            num_points_per_voxel = points.new_zeros(
                size=(max_voxels, ), dtype=torch.int)
            voxel_num = torch.zeros(size=(), dtype=torch.long)
            ext_module.hard_voxelize_forward(
                points,
                torch.tensor(voxel_size, dtype=torch.float),
                torch.tensor(coors_range, dtype=torch.float),
                voxels,
                coors,
                num_points_per_voxel,
                voxel_num,
                max_points=max_points,
                max_voxels=max_voxels,
                NDim=3,
                deterministic=deterministic)
            # select the valid voxels
            voxels_out = voxels[:voxel_num]
            coors_out = coors[:voxel_num]
            num_points_per_voxel_out = num_points_per_voxel[:voxel_num]
            return voxels_out, coors_out, num_points_per_voxel_out


voxelization = _Voxelization.apply


class Voxelization(nn.Module):
    """Convert kitti points(N, >=3) to voxels.

    Please refer to `Point-Voxel CNN for Efficient 3D Deep Learning
    <https://arxiv.org/abs/1907.03739>`_ for more details.

    Args:
        voxel_size (tuple or float): The size of voxel with the shape of [3].
        point_cloud_range (tuple or float): The coordinate range of voxel with
            the shape of [6].
        max_num_points (int): maximum points contained in a voxel. if
            max_points=-1, it means using dynamic_voxelize.
        max_voxels (int, optional): maximum voxels this function create.
            for second, 20000 is a good choice. Users should shuffle points
            before call this function because max_voxels may drop points.
            Default: 20000.
    """

    def __init__(self,
                 voxel_size: List,
                 point_cloud_range: List,
                 max_num_points: int,
                 max_voxels: Union[tuple, int] = 20000,
                 deterministic: bool = True):
        """
        Args:
            voxel_size (list): list [x, y, z] size of three dimension
            point_cloud_range (list):
                [x_min, y_min, z_min, x_max, y_max, z_max]
            max_num_points (int): max number of points per voxel
            max_voxels (tuple or int): max number of voxels in
                (training, testing) time
            deterministic: bool. whether to invoke the non-deterministic
                version of hard-voxelization implementations. non-deterministic
                version is considerablly fast but is not deterministic. only
                affects hard voxelization. default True. for more information
                of this argument and the implementation insights, please refer
                to the following links:
                https://github.com/open-mmlab/mmdetection3d/issues/894
                https://github.com/open-mmlab/mmdetection3d/pull/904
                it is an experimental feature and we will appreciate it if
                you could share with us the failing cases.
        """
        super().__init__()

        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_num_points = max_num_points
        if isinstance(max_voxels, tuple):
            self.max_voxels = max_voxels
        else:
            self.max_voxels = _pair(max_voxels)
        self.deterministic = deterministic

        point_cloud_range = torch.tensor(
            point_cloud_range, dtype=torch.float32)
        voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        grid_size = (
            point_cloud_range[3:] -  # type: ignore
            point_cloud_range[:3]) / voxel_size  # type: ignore
        grid_size = torch.round(grid_size).long()
        input_feat_shape = grid_size[:2]
        self.grid_size = grid_size
        # the origin shape is as [x-len, y-len, z-len]
        # [w, h, d] -> [d, h, w]
        self.pcd_shape = [*input_feat_shape, 1][::-1]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            max_voxels = self.max_voxels[0]
        else:
            max_voxels = self.max_voxels[1]

        return voxelization(input, self.voxel_size, self.point_cloud_range,
                            self.max_num_points, max_voxels,
                            self.deterministic)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'voxel_size=' + str(self.voxel_size)
        s += ', point_cloud_range=' + str(self.point_cloud_range)
        s += ', max_num_points=' + str(self.max_num_points)
        s += ', max_voxels=' + str(self.max_voxels)
        s += ', deterministic=' + str(self.deterministic)
        s += ')'
        return s
