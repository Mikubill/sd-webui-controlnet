# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta

from torch.utils.data import Dataset


class Body3DBaseDataset(Dataset, metaclass=ABCMeta):
    """This class has been deprecated and replaced by
    Kpt3dSviewKpt2dDataset."""

    def __init__(self, *args, **kwargs):
        raise (ImportError(
            'Body3DBaseDataset has been replaced by '
            'Kpt3dSviewKpt2dDataset'
            'check https://github.com/open-mmlab/mmpose/pull/663 for details.')
               )
