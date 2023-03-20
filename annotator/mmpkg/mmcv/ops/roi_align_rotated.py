# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair

from ..utils import deprecated_api_warning, ext_loader

ext_module = ext_loader.load_ext(
    '_ext', ['roi_align_rotated_forward', 'roi_align_rotated_backward'])


class RoIAlignRotatedFunction(Function):

    @staticmethod
    def symbolic(g, input, rois, output_size, spatial_scale, sampling_ratio,
                 aligned, clockwise):
        if isinstance(output_size, int):
            out_h = output_size
            out_w = output_size
        elif isinstance(output_size, tuple):
            assert len(output_size) == 2
            assert isinstance(output_size[0], int)
            assert isinstance(output_size[1], int)
            out_h, out_w = output_size
        else:
            raise TypeError(
                '"output_size" must be an integer or tuple of integers')
        return g.op(
            'mmcv::MMCVRoIAlignRotated',
            input,
            rois,
            output_height_i=out_h,
            output_width_i=out_h,
            spatial_scale_f=spatial_scale,
            sampling_ratio_i=sampling_ratio,
            aligned_i=aligned,
            clockwise_i=clockwise)

    @staticmethod
    def forward(ctx: Any,
                input: torch.Tensor,
                rois: torch.Tensor,
                output_size: Union[int, tuple],
                spatial_scale: float,
                sampling_ratio: int = 0,
                aligned: bool = True,
                clockwise: bool = False) -> torch.Tensor:
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        ctx.aligned = aligned
        ctx.clockwise = clockwise
        ctx.save_for_backward(rois)
        ctx.feature_size = input.size()

        batch_size, num_channels, data_height, data_width = input.size()
        num_rois = rois.size(0)

        output = input.new_zeros(num_rois, num_channels, ctx.output_size[0],
                                 ctx.output_size[1])
        ext_module.roi_align_rotated_forward(
            input,
            rois,
            output,
            pooled_height=ctx.output_size[0],
            pooled_width=ctx.output_size[1],
            spatial_scale=ctx.spatial_scale,
            sampling_ratio=ctx.sampling_ratio,
            aligned=ctx.aligned,
            clockwise=ctx.clockwise)
        return output

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], None, None,
               None, None, None]:
        feature_size = ctx.feature_size
        rois = ctx.saved_tensors[0]
        assert feature_size is not None
        batch_size, num_channels, data_height, data_width = feature_size

        out_w = grad_output.size(3)
        out_h = grad_output.size(2)

        grad_input = grad_rois = None

        if ctx.needs_input_grad[0]:
            grad_input = rois.new_zeros(batch_size, num_channels, data_height,
                                        data_width)
            ext_module.roi_align_rotated_backward(
                grad_output.contiguous(),
                rois,
                grad_input,
                pooled_height=out_h,
                pooled_width=out_w,
                spatial_scale=ctx.spatial_scale,
                sampling_ratio=ctx.sampling_ratio,
                aligned=ctx.aligned,
                clockwise=ctx.clockwise)
        return grad_input, grad_rois, None, None, None, None, None


roi_align_rotated = RoIAlignRotatedFunction.apply


class RoIAlignRotated(nn.Module):
    """RoI align pooling layer for rotated proposals.

    It accepts a feature map of shape (N, C, H, W) and rois with shape
    (n, 6) with each roi decoded as (batch_index, center_x, center_y,
    w, h, angle). The angle is in radian.

    Args:
        output_size (tuple): h, w
        spatial_scale (float): scale the input boxes by this number
        sampling_ratio(int): number of inputs samples to take for each
            output sample. 0 to take samples densely for current models.
        aligned (bool): if False, use the legacy implementation in
            MMDetection. If True, align the results more perfectly.
            Default: True.
        clockwise (bool): If True, the angle in each proposal follows a
            clockwise fashion in image space, otherwise, the angle is
            counterclockwise. Default: False.

    Note:
        The implementation of RoIAlign when aligned=True is modified from
        https://github.com/facebookresearch/detectron2/

        The meaning of aligned=True:

        Given a continuous coordinate c, its two neighboring pixel
        indices (in our pixel model) are computed by floor(c - 0.5) and
        ceil(c - 0.5). For example, c=1.3 has pixel neighbors with discrete
        indices [0] and [1] (which are sampled from the underlying signal
        at continuous coordinates 0.5 and 1.5). But the original roi_align
        (aligned=False) does not subtract the 0.5 when computing
        neighboring pixel indices and therefore it uses pixels with a
        slightly incorrect alignment (relative to our pixel model) when
        performing bilinear interpolation.

        With `aligned=True`,
        we first appropriately scale the ROI and then shift it by -0.5
        prior to calling roi_align. This produces the correct neighbors;

        The difference does not make a difference to the model's
        performance if ROIAlign is used together with conv layers.
    """

    @deprecated_api_warning(
        {
            'out_size': 'output_size',
            'sample_num': 'sampling_ratio'
        },
        cls_name='RoIAlignRotated')
    def __init__(self,
                 output_size: Union[int, tuple],
                 spatial_scale: float,
                 sampling_ratio: int = 0,
                 aligned: bool = True,
                 clockwise: bool = False):
        super().__init__()

        self.output_size = _pair(output_size)
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)
        self.aligned = aligned
        self.clockwise = clockwise

    def forward(self, input: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        return RoIAlignRotatedFunction.apply(input, rois, self.output_size,
                                             self.spatial_scale,
                                             self.sampling_ratio, self.aligned,
                                             self.clockwise)

    def __repr__(self):
        s = self.__class__.__name__
        s += f'(output_size={self.output_size}, '
        s += f'spatial_scale={self.spatial_scale}, '
        s += f'sampling_ratio={self.sampling_ratio}, '
        s += f'aligned={self.aligned}, '
        s += f'clockwise={self.clockwise})'
        return s
