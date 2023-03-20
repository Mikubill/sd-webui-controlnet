# Copyright (c) OpenMMLab. All rights reserved.
import io
import os.path as osp
import warnings
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from cv2 import (IMREAD_COLOR, IMREAD_GRAYSCALE, IMREAD_IGNORE_ORIENTATION,
                 IMREAD_UNCHANGED)

from annotator.mmpkg.mmcv.fileio import FileClient
from annotator.mmpkg.mmcv.utils import is_filepath, is_str

try:
    from turbojpeg import TJCS_RGB, TJPF_BGR, TJPF_GRAY, TurboJPEG
except ImportError:
    TJCS_RGB = TJPF_GRAY = TJPF_BGR = TurboJPEG = None

try:
    from PIL import Image, ImageOps
except ImportError:
    Image = None

try:
    import tifffile
except ImportError:
    tifffile = None

jpeg = None
supported_backends = ['cv2', 'turbojpeg', 'pillow', 'tifffile']

imread_flags = {
    'color': IMREAD_COLOR,
    'grayscale': IMREAD_GRAYSCALE,
    'unchanged': IMREAD_UNCHANGED,
    'color_ignore_orientation': IMREAD_IGNORE_ORIENTATION | IMREAD_COLOR,
    'grayscale_ignore_orientation':
    IMREAD_IGNORE_ORIENTATION | IMREAD_GRAYSCALE
}

imread_backend = 'cv2'


def use_backend(backend: str) -> None:
    """Select a backend for image decoding.

    Args:
        backend (str): The image decoding backend type. Options are `cv2`,
        `pillow`, `turbojpeg` (see https://github.com/lilohuang/PyTurboJPEG)
        and `tifffile`. `turbojpeg` is faster but it only supports `.jpeg`
        file format.
    """
    assert backend in supported_backends
    global imread_backend
    imread_backend = backend
    if imread_backend == 'turbojpeg':
        if TurboJPEG is None:
            raise ImportError('`PyTurboJPEG` is not installed')
        global jpeg
        if jpeg is None:
            jpeg = TurboJPEG()
    elif imread_backend == 'pillow':
        if Image is None:
            raise ImportError('`Pillow` is not installed')
    elif imread_backend == 'tifffile':
        if tifffile is None:
            raise ImportError('`tifffile` is not installed')


def _jpegflag(flag: str = 'color', channel_order: str = 'bgr'):
    channel_order = channel_order.lower()
    if channel_order not in ['rgb', 'bgr']:
        raise ValueError('channel order must be either "rgb" or "bgr"')

    if flag == 'color':
        if channel_order == 'bgr':
            return TJPF_BGR
        elif channel_order == 'rgb':
            return TJCS_RGB
    elif flag == 'grayscale':
        return TJPF_GRAY
    else:
        raise ValueError('flag must be "color" or "grayscale"')


def _pillow2array(img,
                  flag: str = 'color',
                  channel_order: str = 'bgr') -> np.ndarray:
    """Convert a pillow image to numpy array.

    Args:
        img (:obj:`PIL.Image.Image`): The image loaded using PIL
        flag (str): Flags specifying the color type of a loaded image,
            candidates are 'color', 'grayscale' and 'unchanged'.
            Default to 'color'.
        channel_order (str): The channel order of the output image array,
            candidates are 'bgr' and 'rgb'. Default to 'bgr'.

    Returns:
        np.ndarray: The converted numpy array
    """
    channel_order = channel_order.lower()
    if channel_order not in ['rgb', 'bgr']:
        raise ValueError('channel order must be either "rgb" or "bgr"')

    if flag == 'unchanged':
        array = np.array(img)
        if array.ndim >= 3 and array.shape[2] >= 3:  # color image
            array[:, :, :3] = array[:, :, (2, 1, 0)]  # RGB to BGR
    else:
        # Handle exif orientation tag
        if flag in ['color', 'grayscale']:
            img = ImageOps.exif_transpose(img)
        # If the image mode is not 'RGB', convert it to 'RGB' first.
        if img.mode != 'RGB':
            if img.mode != 'LA':
                # Most formats except 'LA' can be directly converted to RGB
                img = img.convert('RGB')
            else:
                # When the mode is 'LA', the default conversion will fill in
                #  the canvas with black, which sometimes shadows black objects
                #  in the foreground.
                #
                # Therefore, a random color (124, 117, 104) is used for canvas
                img_rgba = img.convert('RGBA')
                img = Image.new('RGB', img_rgba.size, (124, 117, 104))
                img.paste(img_rgba, mask=img_rgba.split()[3])  # 3 is alpha
        if flag in ['color', 'color_ignore_orientation']:
            array = np.array(img)
            if channel_order != 'rgb':
                array = array[:, :, ::-1]  # RGB to BGR
        elif flag in ['grayscale', 'grayscale_ignore_orientation']:
            img = img.convert('L')
            array = np.array(img)
        else:
            raise ValueError(
                'flag must be "color", "grayscale", "unchanged", '
                f'"color_ignore_orientation" or "grayscale_ignore_orientation"'
                f' but got {flag}')
    return array


def imread(img_or_path: Union[np.ndarray, str, Path],
           flag: str = 'color',
           channel_order: str = 'bgr',
           backend: Optional[str] = None,
           file_client_args: Optional[dict] = None) -> np.ndarray:
    """Read an image.

    Note:
        In v1.4.1 and later, add `file_client_args` parameters.

    Args:
        img_or_path (ndarray or str or Path): Either a numpy array or str or
            pathlib.Path. If it is a numpy array (loaded image), then
            it will be returned as is.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale`, `unchanged`,
            `color_ignore_orientation` and `grayscale_ignore_orientation`.
            By default, `cv2` and `pillow` backend would rotate the image
            according to its EXIF info unless called with `unchanged` or
            `*_ignore_orientation` flags. `turbojpeg` and `tifffile` backend
            always ignore image's EXIF info regardless of the flag.
            The `turbojpeg` backend only supports `color` and `grayscale`.
        channel_order (str): Order of channel, candidates are `bgr` and `rgb`.
        backend (str | None): The image decoding backend type. Options are
            `cv2`, `pillow`, `turbojpeg`, `tifffile`, `None`.
            If backend is None, the global imread_backend specified by
            ``mmcv.use_backend()`` will be used. Default: None.
        file_client_args (dict | None): Arguments to instantiate a
            FileClient. See :class:`mmcv.fileio.FileClient` for details.
            Default: None.

    Returns:
        ndarray: Loaded image array.

    Examples:
        >>> import annotator.mmpkg.mmcv as mmcv
        >>> img_path = '/path/to/img.jpg'
        >>> img = mmcv.imread(img_path)
        >>> img = mmcv.imread(img_path, flag='color', channel_order='rgb',
        ...     backend='cv2')
        >>> img = mmcv.imread(img_path, flag='color', channel_order='bgr',
        ...     backend='pillow')
        >>> s3_img_path = 's3://bucket/img.jpg'
        >>> # infer the file backend by the prefix s3
        >>> img = mmcv.imread(s3_img_path)
        >>> # manually set the file backend petrel
        >>> img = mmcv.imread(s3_img_path, file_client_args={
        ...     'backend': 'petrel'})
        >>> http_img_path = 'http://path/to/img.jpg'
        >>> img = mmcv.imread(http_img_path)
        >>> img = mmcv.imread(http_img_path, file_client_args={
        ...     'backend': 'http'})
    """

    if isinstance(img_or_path, Path):
        img_or_path = str(img_or_path)

    if isinstance(img_or_path, np.ndarray):
        return img_or_path
    elif is_str(img_or_path):
        file_client = FileClient.infer_client(file_client_args, img_or_path)
        img_bytes = file_client.get(img_or_path)
        return imfrombytes(img_bytes, flag, channel_order, backend)
    else:
        raise TypeError('"img" must be a numpy array or a str or '
                        'a pathlib.Path object')


def imfrombytes(content: bytes,
                flag: str = 'color',
                channel_order: str = 'bgr',
                backend: Optional[str] = None) -> np.ndarray:
    """Read an image from bytes.

    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Same as :func:`imread`.
        channel_order (str): The channel order of the output, candidates
            are 'bgr' and 'rgb'. Default to 'bgr'.
        backend (str | None): The image decoding backend type. Options are
            `cv2`, `pillow`, `turbojpeg`, `tifffile`, `None`. If backend is
            None, the global imread_backend specified by ``mmcv.use_backend()``
            will be used. Default: None.

    Returns:
        ndarray: Loaded image array.

    Examples:
        >>> img_path = '/path/to/img.jpg'
        >>> with open(img_path, 'rb') as f:
        >>>     img_buff = f.read()
        >>> img = mmcv.imfrombytes(img_buff)
        >>> img = mmcv.imfrombytes(img_buff, flag='color', channel_order='rgb')
        >>> img = mmcv.imfrombytes(img_buff, backend='pillow')
        >>> img = mmcv.imfrombytes(img_buff, backend='cv2')
    """

    if backend is None:
        backend = imread_backend
    if backend not in supported_backends:
        raise ValueError(
            f'backend: {backend} is not supported. Supported '
            "backends are 'cv2', 'turbojpeg', 'pillow', 'tifffile'")
    if backend == 'turbojpeg':
        img = jpeg.decode(  # type: ignore
            content, _jpegflag(flag, channel_order))
        if img.shape[-1] == 1:
            img = img[:, :, 0]
        return img
    elif backend == 'pillow':
        with io.BytesIO(content) as buff:
            img = Image.open(buff)
            img = _pillow2array(img, flag, channel_order)
        return img
    elif backend == 'tifffile':
        with io.BytesIO(content) as buff:
            img = tifffile.imread(buff)
        return img
    else:
        img_np = np.frombuffer(content, np.uint8)
        flag = imread_flags[flag] if is_str(flag) else flag
        img = cv2.imdecode(img_np, flag)
        if flag == IMREAD_COLOR and channel_order == 'rgb':
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        return img


def imwrite(img: np.ndarray,
            file_path: str,
            params: Optional[list] = None,
            auto_mkdir: Optional[bool] = None,
            file_client_args: Optional[dict] = None) -> bool:
    """Write image to file.

    Note:
        In v1.4.1 and later, add `file_client_args` parameters.

    Warning:
        The parameter `auto_mkdir` will be deprecated in the future and every
        file clients will make directory automatically.

    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv :func:`imwrite` interface.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically. It will be deprecated.
        file_client_args (dict | None): Arguments to instantiate a
            FileClient. See :class:`mmcv.fileio.FileClient` for details.
            Default: None.

    Returns:
        bool: Successful or not.

    Examples:
        >>> # write to hard disk client
        >>> ret = mmcv.imwrite(img, '/path/to/img.jpg')
        >>> # infer the file backend by the prefix s3
        >>> ret = mmcv.imwrite(img, 's3://bucket/img.jpg')
        >>> # manually set the file backend petrel
        >>> ret = mmcv.imwrite(img, 's3://bucket/img.jpg', file_client_args={
        ...     'backend': 'petrel'})
    """
    assert is_filepath(file_path)
    file_path = str(file_path)
    if auto_mkdir is not None:
        warnings.warn(
            'The parameter `auto_mkdir` will be deprecated in the future and '
            'every file clients will make directory automatically.')
    file_client = FileClient.infer_client(file_client_args, file_path)
    img_ext = osp.splitext(file_path)[-1]
    # Encode image according to image suffix.
    # For example, if image path is '/path/your/img.jpg', the encode
    # format is '.jpg'.
    flag, img_buff = cv2.imencode(img_ext, img, params)
    file_client.put(img_buff.tobytes(), file_path)
    return flag
