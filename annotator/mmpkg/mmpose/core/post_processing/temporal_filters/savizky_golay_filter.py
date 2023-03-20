# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from scipy.signal import savgol_filter

from .builder import FILTERS
from .filter import TemporalFilter


@FILTERS.register_module(name=['SavizkyGolayFilter', 'savgol'])
class SavizkyGolayFilter(TemporalFilter):
    """Savizky-Golay filter.

    Adapted from:
    https://docs.scipy.org/doc/scipy/reference/generated/
    scipy.signal.savgol_filter.html.

    Args:
        window_size (int): The size of the filter window (i.e., the number
            of coefficients). window_length must be a positive odd integer.
            Default: 11
        polyorder (int): The order of the polynomial used to fit the samples.
            polyorder must be less than window_size.
    """

    def __init__(self, window_size: int = 11, polyorder: int = 2):
        super().__init__(window_size)

        # 1-D Savitzky-Golay filter
        assert polyorder > 0, (
            f'Got invalid parameter polyorder={polyorder}. Polyorder '
            'should be positive.')
        assert polyorder < window_size, (
            f'Got invalid parameters polyorder={polyorder} and '
            f'window_size={window_size}. Polyorder should be less than '
            'window_size.')
        self.polyorder = polyorder

    def __call__(self, x: np.ndarray):

        assert x.ndim == 3, ('Input should be an array with shape [T, K, C]'
                             f', but got invalid shape {x.shape}')

        T = x.shape[0]
        if T < self.window_size:
            pad_width = [(self.window_size - T, 0), (0, 0), (0, 0)]
            x = np.pad(x, pad_width, mode='edge')

        smoothed = savgol_filter(x, self.window_size, self.polyorder, axis=0)

        return smoothed[-T:]
