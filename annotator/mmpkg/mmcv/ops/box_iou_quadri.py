# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', ['box_iou_quadri'])


def box_iou_quadri(bboxes1: torch.Tensor,
                   bboxes2: torch.Tensor,
                   mode: str = 'iou',
                   aligned: bool = False) -> torch.Tensor:
    """Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in
    (x1, y1, ..., x4, y4) format.

    If ``aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    Args:
        bboxes1 (torch.Tensor): quadrilateral bboxes 1. It has shape (N, 8),
            indicating (x1, y1, ..., x4, y4) for each row.
        bboxes2 (torch.Tensor): quadrilateral bboxes 2. It has shape (M, 8),
            indicating (x1, y1, ..., x4, y4) for each row.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).

    Returns:
        torch.Tensor: Return the ious betweens boxes. If ``aligned`` is
        ``False``, the shape of ious is (N, M) else (N,).
    """
    assert mode in ['iou', 'iof']
    mode_dict = {'iou': 0, 'iof': 1}
    mode_flag = mode_dict[mode]
    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if aligned:
        ious = bboxes1.new_zeros(rows)
    else:
        ious = bboxes1.new_zeros(rows * cols)
    bboxes1 = bboxes1.contiguous()
    bboxes2 = bboxes2.contiguous()
    ext_module.box_iou_quadri(
        bboxes1, bboxes2, ious, mode_flag=mode_flag, aligned=aligned)
    if not aligned:
        ious = ious.view(rows, cols)
    return ious
