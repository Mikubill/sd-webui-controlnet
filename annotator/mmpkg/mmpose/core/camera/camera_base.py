# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

from annotator.mmpkg.mmcv.utils import Registry

CAMERAS = Registry('camera')


class SingleCameraBase(metaclass=ABCMeta):
    """Base class for single camera model.

    Args:
        param (dict): Camera parameters

    Methods:
        world_to_camera: Project points from world coordinates to camera
            coordinates
        camera_to_world: Project points from camera coordinates to world
            coordinates
        camera_to_pixel: Project points from camera coordinates to pixel
            coordinates
        world_to_pixel: Project points from world coordinates to pixel
            coordinates
    """

    @abstractmethod
    def __init__(self, param):
        """Load camera parameters and check validity."""

    def world_to_camera(self, X):
        """Project points from world coordinates to camera coordinates."""
        raise NotImplementedError

    def camera_to_world(self, X):
        """Project points from camera coordinates to world coordinates."""
        raise NotImplementedError

    def camera_to_pixel(self, X):
        """Project points from camera coordinates to pixel coordinates."""
        raise NotImplementedError

    def world_to_pixel(self, X):
        """Project points from world coordinates to pixel coordinates."""
        _X = self.world_to_camera(X)
        return self.camera_to_pixel(_X)
