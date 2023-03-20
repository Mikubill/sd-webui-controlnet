# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from .builder import DATASETS, PIPELINES

__all__ = ['DATASETS', 'PIPELINES']

warnings.simplefilter('once', DeprecationWarning)
warnings.warn(
    'Registries (DATASETS, PIPELINES) have been moved to '
    'mmpose.datasets.builder. Importing from '
    'mmpose.models.registry will be deprecated in the future.',
    DeprecationWarning)
