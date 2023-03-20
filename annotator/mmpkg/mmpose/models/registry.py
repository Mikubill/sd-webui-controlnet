# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from .builder import BACKBONES, HEADS, LOSSES, NECKS, POSENETS

__all__ = ['BACKBONES', 'HEADS', 'LOSSES', 'NECKS', 'POSENETS']

warnings.simplefilter('once', DeprecationWarning)
warnings.warn(
    'Registries (BACKBONES, NECKS, HEADS, LOSSES, POSENETS) have '
    'been moved to mmpose.models.builder. Importing from '
    'mmpose.models.registry will be deprecated in the future.',
    DeprecationWarning)
