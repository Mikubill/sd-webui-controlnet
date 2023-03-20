# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import os
import os.path as osp
import sys
import time
from contextlib import contextmanager
from typing import List, Optional, Tuple
from urllib.parse import urlparse
from urllib.request import urlopen

import cv2
import numpy as np
from torch.hub import HASH_REGEX, download_url_to_file


@contextmanager
def limit_max_fps(fps: float):
    """A context manager to limit maximum frequence of entering the context.

    Args:
        fps (float): The maximum frequence of entering the context

    Example::
        >>> from annotator.mmpkg.mmpose.apis.webcam.utils import limit_max_fps
        >>> import cv2

        >>> while True:
        ...     with limit_max_fps(20):
        ...         cv2.imshow(img)  # display image at most 20 fps
    """
    t_start = time.time()
    try:
        yield
    finally:
        t_end = time.time()
        if fps is not None:
            t_sleep = 1.0 / fps - t_end + t_start
            if t_sleep > 0:
                time.sleep(t_sleep)


def _is_url(filename: str) -> bool:
    """Check if the file is a url link.

    Args:
        filename (str): the file name or url link

    Returns:
        bool: is url or not.
    """
    prefixes = ['http://', 'https://']
    for p in prefixes:
        if filename.startswith(p):
            return True
    return False


def load_image_from_disk_or_url(filename: str,
                                readFlag: int = cv2.IMREAD_COLOR
                                ) -> np.ndarray:
    """Load an image file, from disk or url.

    Args:
        filename (str): file name on the disk or url link
        readFlag (int): readFlag for imdecode. Default: cv2.IMREAD_COLOR

    Returns:
        np.ndarray: A loaded image
    """
    if _is_url(filename):
        # download the image, convert it to a NumPy array, and then read
        # it into OpenCV format
        resp = urlopen(filename)
        image = np.asarray(bytearray(resp.read()), dtype='uint8')
        image = cv2.imdecode(image, readFlag)
        return image
    else:
        image = cv2.imread(filename, readFlag)
        return image


def mkdir_or_exist(dir_name: str, mode: int = 0o777):
    """Create a directory if it doesn't exist."""
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


def get_cached_file_path(url: str,
                         save_dir: Optional[str] = None,
                         progress: bool = True,
                         check_hash: bool = False,
                         file_name: Optional[str] = None) -> str:
    r"""Loads the Torch serialized object at the given URL.

    If downloaded file is a zip file, it will be automatically decompressed

    If the object is already present in `model_dir`, it's deserialized and
    returned.
    The default value of ``model_dir`` is ``<hub_dir>/checkpoints`` where
    ``hub_dir`` is the directory returned by :func:`~torch.hub.get_dir`.

    Args:
        url (str): URL of the object to download
        save_dir (str, optional): directory in which to save the object
        progress (bool): whether or not to display a progress bar
            to stderr. Default: ``True``
        check_hash(bool): If True, the filename part of the URL
            should follow the naming convention ``filename-<sha256>.ext``
            where ``<sha256>`` is the first eight or more digits of the
            SHA256 hash of the contents of the file. The hash is used to
            ensure unique names and to verify the contents of the file.
            Default: ``False``
        file_name (str, optional): name for the downloaded file. Filename
            from ``url`` will be used if not set. Default: ``None``.

    Returns:
        str: The path to the cached file.
    """
    if save_dir is None:
        save_dir = os.path.join('webcam_resources')

    mkdir_or_exist(save_dir)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.join(save_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        if check_hash:
            r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
            hash_prefix = r.group(1) if r else None
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)
    return cached_file


def screen_matting(img: np.ndarray,
                   color_low: Optional[Tuple] = None,
                   color_high: Optional[Tuple] = None,
                   color: Optional[str] = None) -> np.ndarray:
    """Get screen matting mask.

    Args:
        img (np.ndarray): Image data.
        color_low (tuple): Lower limit (b, g, r).
        color_high (tuple): Higher limit (b, g, r).
        color (str): Support colors include:

            - 'green' or 'g'
            - 'blue' or 'b'
            - 'black' or 'k'
            - 'white' or 'w'

    Returns:
        np.ndarray: A mask with the same shape of the input image. The value
        is 0 at the pixels in the matting color range, and 1 everywhere else.
    """

    if color_high is None or color_low is None:
        if color is not None:
            if color.lower() == 'g' or color.lower() == 'green':
                color_low = (0, 200, 0)
                color_high = (60, 255, 60)
            elif color.lower() == 'b' or color.lower() == 'blue':
                color_low = (230, 0, 0)
                color_high = (255, 40, 40)
            elif color.lower() == 'k' or color.lower() == 'black':
                color_low = (0, 0, 0)
                color_high = (40, 40, 40)
            elif color.lower() == 'w' or color.lower() == 'white':
                color_low = (230, 230, 230)
                color_high = (255, 255, 255)
            else:
                NotImplementedError(f'Not supported color: {color}.')
        else:
            ValueError('color or color_high | color_low should be given.')

    mask = cv2.inRange(img, np.array(color_low), np.array(color_high)) == 0

    return mask.astype(np.uint8)


def expand_and_clamp(box: List, im_shape: Tuple, scale: float = 1.25) -> List:
    """Expand the bbox and clip it to fit the image shape.

    Args:
        box (list): x1, y1, x2, y2
        im_shape (tuple): image shape (h, w, c)
        scale (float): expand ratio

    Returns:
        list: x1, y1, x2, y2
    """

    x1, y1, x2, y2 = box[:4]
    w = x2 - x1
    h = y2 - y1
    deta_w = w * (scale - 1) / 2
    deta_h = h * (scale - 1) / 2

    x1, y1, x2, y2 = x1 - deta_w, y1 - deta_h, x2 + deta_w, y2 + deta_h

    img_h, img_w = im_shape[:2]

    x1 = min(max(0, int(x1)), img_w - 1)
    y1 = min(max(0, int(y1)), img_h - 1)
    x2 = min(max(0, int(x2)), img_w - 1)
    y2 = min(max(0, int(y2)), img_h - 1)

    return [x1, y1, x2, y2]


def _find_bbox(mask):
    """Find the bounding box for the mask.

    Args:
        mask (ndarray): Mask.

    Returns:
        list(4, ): Returned box (x1, y1, x2, y2).
    """
    mask_shape = mask.shape
    if len(mask_shape) == 3:
        assert mask_shape[-1] == 1, 'the channel of the mask should be 1.'
    elif len(mask_shape) == 2:
        pass
    else:
        NotImplementedError()

    h, w = mask_shape[:2]
    mask_w = mask.sum(0)
    mask_h = mask.sum(1)

    left = 0
    right = w - 1
    up = 0
    down = h - 1

    for i in range(w):
        if mask_w[i] > 0:
            break
        left += 1

    for i in range(w - 1, left, -1):
        if mask_w[i] > 0:
            break
        right -= 1

    for i in range(h):
        if mask_h[i] > 0:
            break
        up += 1

    for i in range(h - 1, up, -1):
        if mask_h[i] > 0:
            break
        down -= 1

    return [left, up, right, down]


def copy_and_paste(
    img: np.ndarray,
    background_img: np.ndarray,
    mask: np.ndarray,
    bbox: Optional[List] = None,
    effect_region: Tuple = (0.2, 0.2, 0.8, 0.8),
    min_size: Tuple = (20, 20)
) -> np.ndarray:
    """Copy the image region and paste to the background.

    Args:
        img (np.ndarray): Image data.
        background_img (np.ndarray): Background image data.
        mask (ndarray): instance segmentation result.
        bbox (list, optional): instance bbox in (x1, y1, x2, y2). If not
            given, the bbox will be obtained by ``_find_bbox()``. Default:
            ``None``
        effect_region (tuple): The region to apply mask, the coordinates
            are normalized (x1, y1, x2, y2). Default: (0.2, 0.2, 0.8, 0.8)
        min_size (tuple): The minimum region size (w, h) in pixels.
            Default: (20, 20)

    Returns:
        np.ndarray: The background with pasted image region.
    """
    background_img = background_img.copy()
    background_h, background_w = background_img.shape[:2]
    region_h = (effect_region[3] - effect_region[1]) * background_h
    region_w = (effect_region[2] - effect_region[0]) * background_w
    region_aspect_ratio = region_w / region_h

    if bbox is None:
        bbox = _find_bbox(mask)
    instance_w = bbox[2] - bbox[0]
    instance_h = bbox[3] - bbox[1]

    if instance_w > min_size[0] and instance_h > min_size[1]:
        aspect_ratio = instance_w / instance_h
        if region_aspect_ratio > aspect_ratio:
            resize_rate = region_h / instance_h
        else:
            resize_rate = region_w / instance_w

        mask_inst = mask[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        img_inst = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        img_inst = cv2.resize(img_inst, (int(
            resize_rate * instance_w), int(resize_rate * instance_h)))
        mask_inst = cv2.resize(
            mask_inst,
            (int(resize_rate * instance_w), int(resize_rate * instance_h)),
            interpolation=cv2.INTER_NEAREST)

        mask_ids = list(np.where(mask_inst == 1))
        mask_ids[1] += int(effect_region[0] * background_w)
        mask_ids[0] += int(effect_region[1] * background_h)

        background_img[tuple(mask_ids)] = img_inst[np.where(mask_inst == 1)]

    return background_img


def is_image_file(path: str) -> bool:
    """Check if a path is an image file by its extension.

    Args:
        path (str): The image path.

    Returns:
        bool: Weather the path is an image file.
    """
    if isinstance(path, str):
        if path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            return True
    return False


def get_config_path(path: str, module_name: str):
    """Get config path from an OpenMMLab codebase.

    If the path is an existing file, it will be directly returned. If the file
    doesn't exist, it will be searched in the 'configs' folder of the
    specified module.

    Args:
        path (str): the path of the config file
        module_name (str): The module name of an OpenMMLab codebase

    Returns:
        str: The config file path.

    Example::
        >>> path = 'configs/_base_/filters/one_euro.py'
        >>> get_config_path(path, 'mmpose')
        '/home/mmpose/configs/_base_/filters/one_euro.py'
    """

    if osp.isfile(path):
        return path

    module = importlib.import_module(module_name)
    module_dir = osp.dirname(module.__file__)
    path_in_module = osp.join(module_dir, '.mim', path)

    if not osp.isfile(path_in_module):
        raise FileNotFoundError(f'Can not find the config file "{path}"')

    return path_in_module
