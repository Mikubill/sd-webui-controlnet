import torch
from torch import Tensor

from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', ['points_in_polygons_forward'])


def points_in_polygons(points: Tensor, polygons: Tensor) -> Tensor:
    """Judging whether points are inside polygons, which is used in the ATSS
    assignment for the rotated boxes.

    It should be noted that when the point is just at the polygon boundary, the
    judgment will be inaccurate, but the effect on assignment is limited.

    Args:
        points (torch.Tensor): It has shape (B, 2), indicating (x, y).
            M means the number of predicted points.
        polygons (torch.Tensor): It has shape (M, 8), indicating
            (x1, y1, x2, y2, x3, y3, x4, y4). M means the number of
            ground truth polygons.

    Returns:
        torch.Tensor: Return the result with the shape of (B, M),
        1 indicates that the point is inside the polygon,
        0 indicates that the point is outside the polygon.
    """
    assert points.shape[1] == 2, \
        'points dimension should be 2, ' \
        f'but got unexpected shape {points.shape[1]}'
    assert polygons.shape[1] == 8, \
        'polygons dimension should be 8, ' \
        f'but got unexpected shape {polygons.shape[1]}'
    output = torch.full([points.shape[0], polygons.shape[0]],
                        0.).cuda().float()
    ext_module.points_in_polygons_forward(points.contiguous(),
                                          polygons.contiguous(), output)
    return output
