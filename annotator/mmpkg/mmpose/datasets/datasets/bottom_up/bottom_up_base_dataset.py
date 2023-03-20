# Copyright (c) OpenMMLab. All rights reserved.
from torch.utils.data import Dataset


class BottomUpBaseDataset(Dataset):
    """This class has been deprecated and replaced by
    Kpt2dSviewRgbImgBottomUpDataset."""

    def __init__(self, *args, **kwargs):
        raise (ImportError(
            'BottomUpBaseDataset has been replaced by '
            'Kpt2dSviewRgbImgBottomUpDataset,'
            'check https://github.com/open-mmlab/mmpose/pull/663 for details.')
               )
