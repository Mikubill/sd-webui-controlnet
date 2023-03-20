# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

import cv2
import numpy as np

from .misc import load_image_from_disk_or_url


class ImageCapture:
    """A mock-up of cv2.VideoCapture that always return a const image.

    Args:
        image (str | ndarray): The image path or image data
    """

    def __init__(self, image: Union[str, np.ndarray]):
        if isinstance(image, str):
            self.image = load_image_from_disk_or_url(image)
        else:
            self.image = image

    def isOpened(self):
        return (self.image is not None)

    def read(self):
        return True, self.image.copy()

    def release(self):
        pass

    def get(self, propId):
        if propId == cv2.CAP_PROP_FRAME_WIDTH:
            return self.image.shape[1]
        elif propId == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.image.shape[0]
        elif propId == cv2.CAP_PROP_FPS:
            return np.nan
        else:
            raise NotImplementedError()
