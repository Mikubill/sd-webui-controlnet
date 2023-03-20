# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta

from torch.utils.data import Dataset


class FashionBaseDataset(Dataset, metaclass=ABCMeta):
    """This class has been deprecated and replaced by
    Kpt2dSviewRgbImgTopDownDataset."""

    def __init__(self, *args, **kwargs):
        raise (ImportError(
            'FashionBaseDataset has been replaced by '
            'Kpt2dSviewRgbImgTopDownDataset,'
            'check https://github.com/open-mmlab/mmpose/pull/663 for details.')
               )
