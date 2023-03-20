# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', ['min_area_polygons'])


def min_area_polygons(pointsets: torch.Tensor) -> torch.Tensor:
    """Find the smallest polygons that surrounds all points in the point sets.

    Args:
        pointsets (Tensor): point sets with shape  (N, 18).

    Returns:
        torch.Tensor: Return the smallest polygons with shape (N, 8).
    """
    polygons = pointsets.new_zeros((pointsets.size(0), 8))
    ext_module.min_area_polygons(pointsets, polygons)
    return polygons
