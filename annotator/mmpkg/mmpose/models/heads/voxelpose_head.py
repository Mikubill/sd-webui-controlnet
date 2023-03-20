# ------------------------------------------------------------------------------
# Copyright and License Information
# https://github.com/microsoft/voxelpose-pytorch/blob/main/lib/models
# Original Licence: MIT License
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import HEADS


@HEADS.register_module()
class CuboidCenterHead(nn.Module):
    """Get results from the 3D human center heatmap. In this module, human 3D
    centers are local maximums obtained from the 3D heatmap via NMS (max-
    pooling).

    Args:
        space_size (list[3]): The size of the 3D space.
        cube_size (list[3]): The size of the heatmap volume.
        space_center (list[3]): The coordinate of space center.
        max_num (int): Maximum of human center detections.
        max_pool_kernel (int): Kernel size of the max-pool kernel in nms.
    """

    def __init__(self,
                 space_size,
                 space_center,
                 cube_size,
                 max_num=10,
                 max_pool_kernel=3):
        super(CuboidCenterHead, self).__init__()
        # use register_buffer
        self.register_buffer('grid_size', torch.tensor(space_size))
        self.register_buffer('cube_size', torch.tensor(cube_size))
        self.register_buffer('grid_center', torch.tensor(space_center))

        self.num_candidates = max_num
        self.max_pool_kernel = max_pool_kernel
        self.loss = nn.MSELoss()

    def _get_real_locations(self, indices):
        """
        Args:
            indices (torch.Tensor(NXP)): Indices of points in the 3D tensor

        Returns:
            real_locations (torch.Tensor(NXPx3)): Locations of points
                in the world coordinate system
        """
        real_locations = indices.float() / (
                self.cube_size - 1) * self.grid_size + \
            self.grid_center - self.grid_size / 2.0
        return real_locations

    def _nms_by_max_pool(self, heatmap_volumes):
        max_num = self.num_candidates
        batch_size = heatmap_volumes.shape[0]
        root_cubes_nms = self._max_pool(heatmap_volumes)
        root_cubes_nms_reshape = root_cubes_nms.reshape(batch_size, -1)
        topk_values, topk_index = root_cubes_nms_reshape.topk(max_num)
        topk_unravel_index = self._get_3d_indices(topk_index,
                                                  heatmap_volumes[0].shape)

        return topk_values, topk_unravel_index

    def _max_pool(self, inputs):
        kernel = self.max_pool_kernel
        padding = (kernel - 1) // 2
        max = F.max_pool3d(
            inputs, kernel_size=kernel, stride=1, padding=padding)
        keep = (inputs == max).float()
        return keep * inputs

    @staticmethod
    def _get_3d_indices(indices, shape):
        """Get indices in the 3-D tensor.

        Args:
            indices (torch.Tensor(NXp)): Indices of points in the 1D tensor
            shape (torch.Size(3)): The shape of the original 3D tensor

        Returns:
            indices: Indices of points in the original 3D tensor
        """
        batch_size = indices.shape[0]
        num_people = indices.shape[1]
        indices_x = (indices //
                     (shape[1] * shape[2])).reshape(batch_size, num_people, -1)
        indices_y = ((indices % (shape[1] * shape[2])) //
                     shape[2]).reshape(batch_size, num_people, -1)
        indices_z = (indices % shape[2]).reshape(batch_size, num_people, -1)
        indices = torch.cat([indices_x, indices_y, indices_z], dim=2)
        return indices

    def forward(self, heatmap_volumes):
        """

        Args:
            heatmap_volumes (torch.Tensor(NXLXWXH)):
                3D human center heatmaps predicted by the network.
        Returns:
            human_centers (torch.Tensor(NXPX5)):
                Coordinates of human centers.
        """
        batch_size = heatmap_volumes.shape[0]

        topk_values, topk_unravel_index = self._nms_by_max_pool(
            heatmap_volumes.detach())

        topk_unravel_index = self._get_real_locations(topk_unravel_index)

        human_centers = torch.zeros(
            batch_size, self.num_candidates, 5, device=heatmap_volumes.device)
        human_centers[:, :, 0:3] = topk_unravel_index
        human_centers[:, :, 4] = topk_values

        return human_centers

    def get_loss(self, pred_cubes, gt):

        return dict(loss_center=self.loss(pred_cubes, gt))


@HEADS.register_module()
class CuboidPoseHead(nn.Module):

    def __init__(self, beta):
        """Get results from the 3D human pose heatmap. Instead of obtaining
        maximums on the heatmap, this module regresses the coordinates of
        keypoints via integral pose regression. Refer to `paper.

        <https://arxiv.org/abs/2004.06239>` for more details.

        Args:
            beta: Constant to adjust the magnification of soft-maxed heatmap.
        """
        super(CuboidPoseHead, self).__init__()
        self.beta = beta
        self.loss = nn.L1Loss()

    def forward(self, heatmap_volumes, grid_coordinates):
        """

        Args:
            heatmap_volumes (torch.Tensor(NxKxLxWxH)):
                3D human pose heatmaps predicted by the network.
            grid_coordinates (torch.Tensor(Nx(LxWxH)x3)):
                Coordinates of the grids in the heatmap volumes.
        Returns:
            human_poses (torch.Tensor(NxKx3)): Coordinates of human poses.
        """
        batch_size = heatmap_volumes.size(0)
        channel = heatmap_volumes.size(1)
        x = heatmap_volumes.reshape(batch_size, channel, -1, 1)
        x = F.softmax(self.beta * x, dim=2)
        grid_coordinates = grid_coordinates.unsqueeze(1)
        x = torch.mul(x, grid_coordinates)
        human_poses = torch.sum(x, dim=2)

        return human_poses

    def get_loss(self, preds, targets, weights):

        return dict(loss_pose=self.loss(preds * weights, targets * weights))
