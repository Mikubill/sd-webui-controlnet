# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np


def bbox_xyxy2xywh(bbox_xyxy):
    """Transform the bbox format from x1y1x2y2 to xywh.

    Args:
        bbox_xyxy (np.ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5). (left, top, right, bottom, [score])

    Returns:
        np.ndarray: Bounding boxes (with scores),
          shaped (n, 4) or (n, 5). (left, top, width, height, [score])
    """
    bbox_xywh = bbox_xyxy.copy()
    bbox_xywh[:, 2] = bbox_xywh[:, 2] - bbox_xywh[:, 0]
    bbox_xywh[:, 3] = bbox_xywh[:, 3] - bbox_xywh[:, 1]

    return bbox_xywh


def bbox_xywh2xyxy(bbox_xywh):
    """Transform the bbox format from xywh to x1y1x2y2.

    Args:
        bbox_xywh (ndarray): Bounding boxes (with scores),
            shaped (n, 4) or (n, 5). (left, top, width, height, [score])
    Returns:
        np.ndarray: Bounding boxes (with scores), shaped (n, 4) or
          (n, 5). (left, top, right, bottom, [score])
    """
    bbox_xyxy = bbox_xywh.copy()
    bbox_xyxy[:, 2] = bbox_xyxy[:, 2] + bbox_xyxy[:, 0]
    bbox_xyxy[:, 3] = bbox_xyxy[:, 3] + bbox_xyxy[:, 1]

    return bbox_xyxy


def bbox_xywh2cs(bbox, aspect_ratio, padding=1., pixel_std=200.):
    """Transform the bbox format from (x,y,w,h) into (center, scale)

    Args:
        bbox (ndarray): Single bbox in (x, y, w, h)
        aspect_ratio (float): The expected bbox aspect ratio (w over h)
        padding (float): Bbox padding factor that will be multilied to scale.
            Default: 1.0
        pixel_std (float): The scale normalization factor. Default: 200.0

    Returns:
        tuple: A tuple containing center and scale.
        - np.ndarray[float32](2,): Center of the bbox (x, y).
        - np.ndarray[float32](2,): Scale of the bbox w & h.
    """

    x, y, w, h = bbox[:4]
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio

    scale = np.array([w, h], dtype=np.float32) / pixel_std
    scale = scale * padding

    return center, scale


def bbox_cs2xywh(center, scale, padding=1., pixel_std=200.):
    """Transform the bbox format from (center, scale) to (x,y,w,h). Note that
    this is not an exact inverse operation of ``bbox_xywh2cs`` because the
    normalization of aspect ratio in ``bbox_xywh2cs`` is irreversible.

    Args:
        center (ndarray): Single bbox center in (x, y)
        scale (ndarray): Single bbox scale in (scale_x, scale_y)
        padding (float): Bbox padding factor that will be multilied to scale.
            Default: 1.0
        pixel_std (float): The scale normalization factor. Default: 200.0

    Returns:
        ndarray: Single bbox in (x, y, w, h)
    """

    wh = scale / padding * pixel_std
    xy = center - 0.5 * wh
    return np.r_[xy, wh]
