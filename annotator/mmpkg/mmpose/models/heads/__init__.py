# Copyright (c) OpenMMLab. All rights reserved.
from .ae_higher_resolution_head import AEHigherResolutionHead
from .ae_multi_stage_head import AEMultiStageHead
from .ae_simple_head import AESimpleHead
from .cid_head import CIDHead
from .deconv_head import DeconvHead
from .deeppose_regression_head import DeepposeRegressionHead
from .dekr_head import DEKRHead
from .hmr_head import HMRMeshHead
from .interhand_3d_head import Interhand3DHead
from .mtut_head import MultiModalSSAHead
from .temporal_regression_head import TemporalRegressionHead
from .topdown_heatmap_base_head import TopdownHeatmapBaseHead
from .topdown_heatmap_multi_stage_head import (TopdownHeatmapMSMUHead,
                                               TopdownHeatmapMultiStageHead)
from .topdown_heatmap_simple_head import TopdownHeatmapSimpleHead
from .vipnas_heatmap_simple_head import ViPNASHeatmapSimpleHead
from .voxelpose_head import CuboidCenterHead, CuboidPoseHead

__all__ = [
    'TopdownHeatmapSimpleHead', 'TopdownHeatmapMultiStageHead',
    'TopdownHeatmapMSMUHead', 'TopdownHeatmapBaseHead',
    'AEHigherResolutionHead', 'AESimpleHead', 'AEMultiStageHead', 'CIDHead',
    'DeepposeRegressionHead', 'TemporalRegressionHead', 'Interhand3DHead',
    'HMRMeshHead', 'DeconvHead', 'ViPNASHeatmapSimpleHead', 'CuboidCenterHead',
    'CuboidPoseHead', 'MultiModalSSAHead', 'DEKRHead'
]
