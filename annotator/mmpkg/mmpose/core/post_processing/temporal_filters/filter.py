# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod


class TemporalFilter(metaclass=ABCMeta):
    """Base class of temporal filter.

    A subclass should implement the method __call__().

    Parameters:
        window_size (int): the size of the sliding window.
    """

    # If the filter can be shared by multiple humans or targets
    _shareable: bool = True

    def __init__(self, window_size=1):
        self._window_size = window_size

    @property
    def window_size(self):
        return self._window_size

    @property
    def shareable(self):
        return self._shareable

    @abstractmethod
    def __call__(self, x):
        """Apply filter to a pose sequence.

        Note:
            T: The temporal length of the pose sequence
            K: The keypoint number of each target
            C: The keypoint coordinate dimension

        Args:
            x (np.ndarray): input pose sequence in shape [T, K, C]

        Returns:
            np.ndarray: Smoothed pose sequence in shape [T, K, C]
        """
