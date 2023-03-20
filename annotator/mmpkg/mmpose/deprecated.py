# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from .datasets.builder import DATASETS
from .datasets.datasets.base import Kpt2dSviewRgbImgTopDownDataset
from .models.builder import HEADS, POSENETS
from .models.detectors import AssociativeEmbedding
from .models.heads import (AEHigherResolutionHead, AESimpleHead,
                           DeepposeRegressionHead, HMRMeshHead,
                           TopdownHeatmapMSMUHead,
                           TopdownHeatmapMultiStageHead,
                           TopdownHeatmapSimpleHead)


@DATASETS.register_module()
class TopDownFreiHandDataset(Kpt2dSviewRgbImgTopDownDataset):
    """Deprecated TopDownFreiHandDataset."""

    def __init__(self, *args, **kwargs):
        raise (ImportError(
            'TopDownFreiHandDataset has been renamed into FreiHandDataset,'
            'check https://github.com/open-mmlab/mmpose/pull/202 for details.')
               )

    def _get_db(self):
        return []

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        return None


@DATASETS.register_module()
class TopDownOneHand10KDataset(Kpt2dSviewRgbImgTopDownDataset):
    """Deprecated TopDownOneHand10KDataset."""

    def __init__(self, *args, **kwargs):
        raise (ImportError(
            'TopDownOneHand10KDataset has been renamed into OneHand10KDataset,'
            'check https://github.com/open-mmlab/mmpose/pull/202 for details.')
               )

    def _get_db(self):
        return []

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        return None


@DATASETS.register_module()
class TopDownPanopticDataset(Kpt2dSviewRgbImgTopDownDataset):
    """Deprecated TopDownPanopticDataset."""

    def __init__(self, *args, **kwargs):
        raise (ImportError(
            'TopDownPanopticDataset has been renamed into PanopticDataset,'
            'check https://github.com/open-mmlab/mmpose/pull/202 for details.')
               )

    def _get_db(self):
        return []

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        return None


@HEADS.register_module()
class BottomUpHigherResolutionHead(AEHigherResolutionHead):
    """Bottom-up head for Higher Resolution.

    BottomUpHigherResolutionHead has been renamed into AEHigherResolutionHead,
    check https://github.com/open- mmlab/mmpose/pull/656 for details.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(
            'BottomUpHigherResolutionHead has been renamed into '
            'AEHigherResolutionHead, check '
            'https://github.com/open-mmlab/mmpose/pull/656 for details.',
            DeprecationWarning)


@HEADS.register_module()
class BottomUpSimpleHead(AESimpleHead):
    """Bottom-up simple head.

    BottomUpSimpleHead has been renamed into AESimpleHead, check
    https://github.com/open-mmlab/mmpose/pull/656 for details.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(
            'BottomUpHigherResolutionHead has been renamed into '
            'AEHigherResolutionHead, check '
            'https://github.com/open-mmlab/mmpose/pull/656 for details',
            DeprecationWarning)


@HEADS.register_module()
class TopDownSimpleHead(TopdownHeatmapSimpleHead):
    """Top-down heatmap simple head.

    TopDownSimpleHead has been renamed into TopdownHeatmapSimpleHead, check
    https://github.com/open-mmlab/mmpose/pull/656 for details.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(
            'TopDownSimpleHead has been renamed into '
            'TopdownHeatmapSimpleHead, check '
            'https://github.com/open-mmlab/mmpose/pull/656 for details.',
            DeprecationWarning)


@HEADS.register_module()
class TopDownMultiStageHead(TopdownHeatmapMultiStageHead):
    """Top-down heatmap multi-stage head.

    TopDownMultiStageHead has been renamed into TopdownHeatmapMultiStageHead,
    check https://github.com/open-mmlab/mmpose/pull/656 for details.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(
            'TopDownMultiStageHead has been renamed into '
            'TopdownHeatmapMultiStageHead, check '
            'https://github.com/open-mmlab/mmpose/pull/656 for details.',
            DeprecationWarning)


@HEADS.register_module()
class TopDownMSMUHead(TopdownHeatmapMSMUHead):
    """Heads for multi-stage multi-unit heads.

    TopDownMSMUHead has been renamed into TopdownHeatmapMSMUHead, check
    https://github.com/open-mmlab/mmpose/pull/656 for details.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(
            'TopDownMSMUHead has been renamed into '
            'TopdownHeatmapMSMUHead, check '
            'https://github.com/open-mmlab/mmpose/pull/656 for details.',
            DeprecationWarning)


@HEADS.register_module()
class MeshHMRHead(HMRMeshHead):
    """SMPL parameters regressor head.

    MeshHMRHead has been renamed into HMRMeshHead, check
    https://github.com/open-mmlab/mmpose/pull/656 for details.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(
            'MeshHMRHead has been renamed into '
            'HMRMeshHead, check '
            'https://github.com/open-mmlab/mmpose/pull/656 for details.',
            DeprecationWarning)


@HEADS.register_module()
class FcHead(DeepposeRegressionHead):
    """FcHead (deprecated).

    FcHead has been renamed into DeepposeRegressionHead, check
    https://github.com/open-mmlab/mmpose/pull/656 for details.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(
            'FcHead has been renamed into '
            'DeepposeRegressionHead, check '
            'https://github.com/open-mmlab/mmpose/pull/656 for details.',
            DeprecationWarning)


@POSENETS.register_module()
class BottomUp(AssociativeEmbedding):
    """Associative Embedding.

    BottomUp has been renamed into AssociativeEmbedding, check
    https://github.com/open-mmlab/mmpose/pull/656 for details.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(
            'BottomUp has been renamed into '
            'AssociativeEmbedding, check '
            'https://github.com/open-mmlab/mmpose/pull/656 for details.',
            DeprecationWarning)
