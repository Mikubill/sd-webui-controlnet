# Copyright (c) OpenMMLab. All rights reserved.
# flake8: noqa
import warnings

from .arraymisc import *
from .fileio import *
from .image import *
from .utils import *
from .version import *
from .video import *
from .visualization import *

# The following modules are not imported to this level, so mmcv may be used
# without PyTorch.
# - runner
# - parallel
# - op
# - device

warnings.warn(
    'On January 1, 2023, MMCV will release v2.0.0, in which it will remove '
    'components related to the training process and add a data transformation '
    'module. In addition, it will rename the package names mmcv to mmcv-lite '
    'and mmcv-full to mmcv. '
    'See https://github.com/open-mmlab/mmcv/blob/master/docs/en/compatibility.md '
    'for more details.')
