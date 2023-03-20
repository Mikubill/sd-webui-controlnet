# Copyright (c) OpenMMLab. All rights reserved.
from annotator.mmpkg.mmcv.utils import IS_MLU_AVAILABLE
from .active_rotated_filter import active_rotated_filter
from .assign_score_withk import assign_score_withk
from .ball_query import ball_query
from .bbox import bbox_overlaps
from .border_align import BorderAlign, border_align
from .box_iou_quadri import box_iou_quadri
from .box_iou_rotated import box_iou_rotated
from .carafe import CARAFE, CARAFENaive, CARAFEPack, carafe, carafe_naive
from .cc_attention import CrissCrossAttention
from .chamfer_distance import chamfer_distance
from .contour_expand import contour_expand
from .convex_iou import convex_giou, convex_iou
from .corner_pool import CornerPool
from .correlation import Correlation
from .deform_conv import DeformConv2d, DeformConv2dPack, deform_conv2d
from .deform_roi_pool import (DeformRoIPool, DeformRoIPoolPack,
                              ModulatedDeformRoIPoolPack, deform_roi_pool)
from .deprecated_wrappers import Conv2d_deprecated as Conv2d
from .deprecated_wrappers import ConvTranspose2d_deprecated as ConvTranspose2d
from .deprecated_wrappers import Linear_deprecated as Linear
from .deprecated_wrappers import MaxPool2d_deprecated as MaxPool2d
from .diff_iou_rotated import diff_iou_rotated_2d, diff_iou_rotated_3d
from .focal_loss import (SigmoidFocalLoss, SoftmaxFocalLoss,
                         sigmoid_focal_loss, softmax_focal_loss)
from .furthest_point_sample import (furthest_point_sample,
                                    furthest_point_sample_with_dist)
from .fused_bias_leakyrelu import FusedBiasLeakyReLU, fused_bias_leakyrelu
from .gather_points import gather_points
from .group_points import GroupAll, QueryAndGroup, grouping_operation
from .info import (get_compiler_version, get_compiling_cuda_version,
                   get_onnxruntime_op_path)
from .iou3d import (boxes_iou3d, boxes_iou_bev, boxes_overlap_bev, nms3d,
                    nms3d_normal, nms_bev, nms_normal_bev)
from .knn import knn
from .masked_conv import MaskedConv2d, masked_conv2d
from .min_area_polygons import min_area_polygons
from .modulated_deform_conv import (ModulatedDeformConv2d,
                                    ModulatedDeformConv2dPack,
                                    modulated_deform_conv2d)
from .multi_scale_deform_attn import MultiScaleDeformableAttention
from .nms import batched_nms, nms, nms_match, nms_quadri, nms_rotated, soft_nms
from .pixel_group import pixel_group
from .point_sample import (SimpleRoIAlign, point_sample,
                           rel_roi_point_to_rel_img_point)
from .points_in_boxes import (points_in_boxes_all, points_in_boxes_cpu,
                              points_in_boxes_part)
from .points_in_polygons import points_in_polygons
from .points_sampler import PointsSampler
from .prroi_pool import PrRoIPool, prroi_pool
from .psa_mask import PSAMask
from .riroi_align_rotated import RiRoIAlignRotated, riroi_align_rotated
from .roi_align import RoIAlign, roi_align
from .roi_align_rotated import RoIAlignRotated, roi_align_rotated
from .roi_pool import RoIPool, roi_pool
from .roiaware_pool3d import RoIAwarePool3d
from .roipoint_pool3d import RoIPointPool3d
from .rotated_feature_align import rotated_feature_align
from .saconv import SAConv2d
from .scatter_points import DynamicScatter, dynamic_scatter
from .sparse_conv import (SparseConv2d, SparseConv3d, SparseConvTranspose2d,
                          SparseConvTranspose3d, SparseInverseConv2d,
                          SparseInverseConv3d, SubMConv2d, SubMConv3d)
from .sparse_modules import SparseModule, SparseSequential
from .sparse_pool import SparseMaxPool2d, SparseMaxPool3d
from .sparse_structure import SparseConvTensor, scatter_nd
from .sync_bn import SyncBatchNorm
from .three_interpolate import three_interpolate
from .three_nn import three_nn
from .tin_shift import TINShift, tin_shift
from .upfirdn2d import upfirdn2d
from .voxelize import Voxelization, voxelization

__all__ = [
    'bbox_overlaps', 'CARAFE', 'CARAFENaive', 'CARAFEPack', 'carafe',
    'carafe_naive', 'CornerPool', 'DeformConv2d', 'DeformConv2dPack',
    'deform_conv2d', 'DeformRoIPool', 'DeformRoIPoolPack',
    'ModulatedDeformRoIPoolPack', 'deform_roi_pool', 'SigmoidFocalLoss',
    'SoftmaxFocalLoss', 'sigmoid_focal_loss', 'softmax_focal_loss',
    'get_compiler_version', 'get_compiling_cuda_version',
    'get_onnxruntime_op_path', 'MaskedConv2d', 'masked_conv2d',
    'ModulatedDeformConv2d', 'ModulatedDeformConv2dPack',
    'modulated_deform_conv2d', 'batched_nms', 'nms', 'soft_nms', 'nms_match',
    'RoIAlign', 'roi_align', 'RoIPool', 'roi_pool', 'SyncBatchNorm', 'Conv2d',
    'ConvTranspose2d', 'Linear', 'MaxPool2d', 'CrissCrossAttention', 'PSAMask',
    'point_sample', 'rel_roi_point_to_rel_img_point', 'SimpleRoIAlign',
    'SAConv2d', 'TINShift', 'tin_shift', 'assign_score_withk',
    'box_iou_rotated', 'box_iou_quadri', 'RoIPointPool3d', 'nms_rotated',
    'knn', 'ball_query', 'upfirdn2d', 'FusedBiasLeakyReLU',
    'fused_bias_leakyrelu', 'rotated_feature_align', 'RiRoIAlignRotated',
    'riroi_align_rotated', 'RoIAlignRotated', 'roi_align_rotated',
    'pixel_group', 'QueryAndGroup', 'GroupAll', 'grouping_operation',
    'contour_expand', 'three_nn', 'three_interpolate',
    'MultiScaleDeformableAttention', 'BorderAlign', 'border_align',
    'gather_points', 'furthest_point_sample', 'nms_quadri',
    'furthest_point_sample_with_dist', 'PointsSampler', 'Correlation',
    'boxes_iou3d', 'boxes_iou_bev', 'boxes_overlap_bev', 'nms_bev',
    'nms_normal_bev', 'nms3d', 'nms3d_normal', 'Voxelization', 'voxelization',
    'dynamic_scatter', 'DynamicScatter', 'RoIAwarePool3d', 'SparseConv2d',
    'SparseConv3d', 'SparseConvTranspose2d', 'SparseConvTranspose3d',
    'SparseInverseConv2d', 'SparseInverseConv3d', 'SubMConv2d', 'SubMConv3d',
    'SparseModule', 'SparseSequential', 'SparseMaxPool2d', 'SparseMaxPool3d',
    'SparseConvTensor', 'scatter_nd', 'points_in_boxes_part',
    'points_in_boxes_cpu', 'points_in_boxes_all', 'points_in_polygons',
    'min_area_polygons', 'active_rotated_filter', 'convex_iou', 'convex_giou',
    'diff_iou_rotated_2d', 'diff_iou_rotated_3d', 'chamfer_distance',
    'PrRoIPool', 'prroi_pool'
]

if IS_MLU_AVAILABLE:
    from .deform_conv import DeformConv2dPack_MLU  # noqa:F401
    from .modulated_deform_conv import \
        ModulatedDeformConv2dPack_MLU  # noqa:F401
    __all__.extend(['ModulatedDeformConv2dPack_MLU', 'DeformConv2dPack_MLU'])
