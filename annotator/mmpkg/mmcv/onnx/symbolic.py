# Copyright (c) OpenMMLab. All rights reserved.
"""Modified from https://github.com/pytorch/pytorch."""
import os
import warnings

import numpy as np
import torch
from torch.nn.modules.utils import _pair, _single, _triple
from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import parse_args

from .onnx_utils import symbolic_helper as sym_help


def _interpolate(name, dim, interpolate_mode):

    def symbolic_fn(g, input, output_size, *args):
        scales, align_corners = sym_help._get_interpolate_attributes(
            g, interpolate_mode, args)
        align_corners = sym_help._maybe_get_scalar(align_corners)
        transformation_mode = 'asymmetric' \
            if interpolate_mode == 'nearest' \
            else 'align_corners' if align_corners else 'pytorch_half_pixel'
        empty_tensor = g.op(
            'Constant', value_t=torch.tensor([], dtype=torch.float32))

        if scales is None:
            if 'ONNX_BACKEND' in os.environ and os.environ[
                    'ONNX_BACKEND'] == 'TensorRT':
                input_size = input.type().sizes()
                # slice the first two dim
                input_size = input_size[:2]
                # convert output_size to int type
                output_size = sym_help._maybe_get_const(output_size, 'is')
                input_size.extend(output_size)
                output_size = g.op(
                    'Constant',
                    value_t=torch.tensor(input_size, dtype=torch.int64))
            else:
                input_size = g.op('Shape', input)
                input_size_beg = sym_help._slice_helper(
                    g, input_size, axes=[0], ends=[2], starts=[0])
                output_size = g.op(
                    'Cast',
                    output_size,
                    to_i=sym_help.cast_pytorch_to_onnx['Long'])
                output_size = g.op(
                    'Concat', input_size_beg, output_size, axis_i=0)
            scales = g.op(
                'Constant', value_t=torch.tensor([], dtype=torch.float32))
            return g.op(
                'Resize',
                input,
                empty_tensor,
                # roi only takes effect with
                # coordinate_transformation_mode="tf_crop_and_resize"
                scales,  # scales is not needed since we are sending out_size
                output_size,
                coordinate_transformation_mode_s=transformation_mode,
                cubic_coeff_a_f=-0.75,  # only valid when mode="cubic"
                mode_s=interpolate_mode,  # nearest, linear, or cubic
                nearest_mode_s='floor')  # only valid when mode="nearest"
        else:
            return g.op(
                'Resize',
                input,
                empty_tensor,
                # roi only takes effect with
                # coordinate_transformation_mode="tf_crop_and_resize"
                scales,  # scales is not needed since we are sending out_size
                coordinate_transformation_mode_s=transformation_mode,
                cubic_coeff_a_f=-0.75,  # only valid when mode="cubic"
                mode_s=interpolate_mode,  # nearest, linear, or cubic
                nearest_mode_s='floor')  # only valid when mode="nearest"

    return symbolic_fn


upsample_nearest1d = _interpolate('upsample_nearest1d', 3, 'nearest')
upsample_nearest2d = _interpolate('upsample_nearest2d', 4, 'nearest')
upsample_nearest3d = _interpolate('upsample_nearest3d', 5, 'nearest')
upsample_linear1d = _interpolate('upsample_linear1d', 3, 'linear')
upsample_bilinear2d = _interpolate('upsample_bilinear2d', 4, 'linear')
upsample_trilinear3d = _interpolate('upsample_trilinear3d', 5, 'linear')
upsample_bicubic2d = _interpolate('upsample_bicubic2d', 4, 'cubic')


@parse_args('v', 'v', 'i', 'i', 'i', 'none')
def topk(g, self, k, dim, largest, sorted, out=None):
    return sym_help._topk_helper(
        g, self, k, dim, largest=largest, sorted=sorted, out=out)


def masked_select(g, self, mask):
    from torch.onnx.symbolic_opset9 import expand_as, nonzero
    index = nonzero(g, expand_as(g, mask, self))
    return g.op('GatherND', self, index)


def _prepare_onnx_paddings(g, dim, pad):
    pad_len = torch.onnx.symbolic_opset9.size(
        g, pad, g.op('Constant', value_t=torch.tensor([0])))
    # Set extension = [0] * (dim * 2 - len(pad))
    extension = g.op(
        'Sub',
        g.op('Mul',
             g.op('Constant', value_t=torch.tensor(dim, dtype=torch.int64)),
             g.op('Constant', value_t=torch.tensor(2, dtype=torch.int64))),
        pad_len)
    pad = g.op('Cast', pad, to_i=sym_help.cast_pytorch_to_onnx['Long'])
    paddings = g.op(
        'Concat',
        pad,
        g.op(
            'ConstantOfShape',
            extension,
            value_t=torch.tensor([0], dtype=torch.int64)),
        axis_i=0)
    paddings = g.op('Reshape', paddings,
                    g.op('Constant', value_t=torch.tensor([-1, 2])))
    paddings = g.op(
        'Transpose',
        torch.onnx.symbolic_opset10.flip(g, paddings, [0]),
        perm_i=[1, 0])
    paddings = g.op('Reshape', paddings,
                    g.op('Constant', value_t=torch.tensor([-1])))
    padding_c = g.op(
        'Cast', paddings, to_i=sym_help.cast_pytorch_to_onnx['Long'])
    return padding_c


def constant_pad_nd(g, input, padding, value=None):
    mode = 'constant'
    value = sym_help._maybe_get_scalar(value)
    value = sym_help._if_scalar_type_as(g, value, input)
    pad = _prepare_onnx_paddings(g, input.type().dim(), padding)
    return g.op('Pad', input, pad, value, mode_s=mode)


def reflection_pad(g, input, padding):
    mode = 'reflect'
    paddings = _prepare_onnx_paddings(g, input.type().dim(), padding)
    return g.op('Pad', input, paddings, mode_s=mode)


reflection_pad1d = reflection_pad
reflection_pad2d = reflection_pad
reflection_pad3d = reflection_pad


def _avg_pool(name, tuple_fn):

    @parse_args('v', 'is', 'is', 'is', 'i', 'i', 'none')
    def symbolic_fn(g,
                    input,
                    kernel_size,
                    stride,
                    padding,
                    ceil_mode,
                    count_include_pad,
                    divisor_override=None):
        padding = sym_help._avgpool_helper(tuple_fn, padding, kernel_size,
                                           stride, divisor_override, name)
        if not stride:
            stride = kernel_size
        if count_include_pad:
            input = g.op(
                'Pad',
                input,
                g.op(
                    'Constant',
                    value_t=torch.tensor(((0, ) * 2 + padding) * 2)),
                mode_s='constant')
            padding = (0, ) * len(padding)
        output = g.op(
            'AveragePool',
            input,
            kernel_shape_i=tuple_fn(kernel_size),
            strides_i=tuple_fn(stride),
            pads_i=padding * 2,
            ceil_mode_i=ceil_mode)
        return output

    return symbolic_fn


avg_pool1d = _avg_pool('avg_pool1d', _single)
avg_pool2d = _avg_pool('avg_pool2d', _pair)
avg_pool3d = _avg_pool('avg_pool3d', _triple)


def _get_im2col_indices_along_dim(g, input_d, kernel_size_d, dilation_d,
                                  padding_d, stride_d):
    # Input is always 4-D (N, C, H, W)
    # Calculate indices of sliding blocks along spatial dimension
    # Slide kernel over input each dim d:
    # each dimension d ranges from 0 to
    # input[d]+2xpadding[d]-dilation[d]x(kernel_size[d]-1)
    # with steps = stride

    blocks_d = g.op('Add', input_d,
                    g.op('Constant', value_t=torch.tensor(padding_d * 2)))
    blocks_d = g.op(
        'Sub', blocks_d,
        g.op(
            'Constant',
            value_t=torch.tensor(dilation_d * (kernel_size_d - 1))))

    # Stride kernel over input and find starting indices along dim d
    blocks_d_indices = g.op('Range', g.op('Constant', value_t=torch.tensor(0)),
                            blocks_d,
                            g.op('Constant', value_t=torch.tensor(stride_d)))

    # Apply dilation on kernel and find its indices along dim d
    kernel_grid = np.arange(0, kernel_size_d * dilation_d, dilation_d)
    kernel_grid = g.op('Constant', value_t=torch.tensor([kernel_grid]))

    # Broadcast and add kernel staring positions (indices) with
    # kernel_grid along dim d, to get block indices along dim d
    blocks_d_indices = g.op(
        'Unsqueeze', blocks_d_indices, axes_i=[0])  # Reshape to [1, -1]
    kernel_mask = g.op('Reshape', kernel_grid,
                       g.op('Constant', value_t=torch.tensor([-1, 1])))
    block_mask = g.op('Add', blocks_d_indices, kernel_mask)

    return block_mask


def _get_im2col_padded_input(g, input, padding_h, padding_w):
    # Input is always 4-D tensor (N, C, H, W)
    # Padding tensor has the following format: (padding_h, padding_w)
    # Reshape the padding to follow ONNX format:
    # (dim1_begin, dim2_begin,...,dim1_end, dim2_end,...)
    pad = g.op(
        'Constant', value_t=torch.LongTensor([0, 0, padding_h, padding_w] * 2))
    return g.op('Pad', input, pad)


def _get_im2col_output_shape(g, input, kernel_h, kernel_w):
    batch_dim = size(g, input, g.op('Constant', value_t=torch.tensor(0)))
    channel_dim = size(g, input, g.op('Constant', value_t=torch.tensor(1)))
    channel_unfolded = g.op(
        'Mul', channel_dim,
        g.op('Constant', value_t=torch.tensor(kernel_h * kernel_w)))

    return g.op(
        'Concat',
        g.op('Unsqueeze', batch_dim, axes_i=[0]),
        g.op('Unsqueeze', channel_unfolded, axes_i=[0]),
        g.op('Constant', value_t=torch.tensor([-1])),
        axis_i=0)


def size(g, self, dim=None):
    if dim is None:
        return g.op('Shape', self)
    return sym_help._size_helper(g, self, dim)


@parse_args('v', 'is', 'is', 'is', 'is')
def im2col(g, input, kernel_size, dilation, padding, stride):
    # Input is always 4-D tensor (N, C, H, W)
    # All other args are int[2]

    input_h = size(g, input, g.op('Constant', value_t=torch.tensor(2)))
    input_w = size(g, input, g.op('Constant', value_t=torch.tensor(3)))

    stride_h, stride_w = stride[0], stride[1]
    padding_h, padding_w = padding[0], padding[1]
    dilation_h, dilation_w = dilation[0], dilation[1]
    kernel_h, kernel_w = kernel_size[0], kernel_size[1]

    blocks_row_indices = _get_im2col_indices_along_dim(g, input_h, kernel_h,
                                                       dilation_h, padding_h,
                                                       stride_h)
    blocks_col_indices = _get_im2col_indices_along_dim(g, input_w, kernel_w,
                                                       dilation_w, padding_w,
                                                       stride_w)

    output_shape = _get_im2col_output_shape(g, input, kernel_h, kernel_w)
    padded_input = _get_im2col_padded_input(g, input, padding_h, padding_w)

    output = g.op('Gather', padded_input, blocks_row_indices, axis_i=2)
    output = g.op('Gather', output, blocks_col_indices, axis_i=4)
    output = g.op('Transpose', output, perm_i=[0, 1, 2, 4, 3, 5])
    return g.op('Reshape', output, output_shape)


@parse_args('v', 'i')
def one_hot(g, self, num_classes):
    values = g.op('Constant', value_t=torch.LongTensor([0, 1]))
    depth = g.op('Constant', value_t=torch.LongTensor([num_classes]))
    return g.op('OneHot', self, depth, values, axis_i=-1)


@parse_args('v', 'i', 'none')
def softmax(g, input, dim, dtype=None):
    input_dim = input.type().dim()
    if input_dim:
        # TODO: remove this as onnx opset 11 spec allows negative axes
        if dim < 0:
            dim = input_dim + dim
        if input_dim == dim + 1:
            softmax = g.op('Softmax', input, axis_i=dim)
            if dtype and dtype.node().kind() != 'prim::Constant':
                parsed_dtype = sym_help._get_const(dtype, 'i', 'dtype')
                softmax = g.op(
                    'Cast',
                    softmax,
                    to_i=sym_help.scalar_type_to_onnx[parsed_dtype])
            return softmax

    max_value = g.op('ReduceMax', input, axes_i=[dim], keepdims_i=1)
    input = g.op('Sub', input, max_value)
    exp = g.op('Exp', input)
    sum = g.op('ReduceSum', exp, axes_i=[dim])
    softmax = g.op('Div', exp, sum)
    if dtype and dtype.node().kind() != 'prim::Constant':
        parsed_dtype = sym_help._get_const(dtype, 'i', 'dtype')
        softmax = g.op(
            'Cast', softmax, to_i=sym_help.scalar_type_to_onnx[parsed_dtype])
    return softmax


def _adaptive_pool(name, type, tuple_fn, fn=None):

    @parse_args('v', 'is')
    def symbolic_fn(g, input, output_size):
        if output_size == [1] * len(output_size) and type == 'AveragePool':
            return g.op('GlobalAveragePool', input)
        if not input.isCompleteTensor():
            if output_size == [1] * len(output_size):
                return g.op('GlobalMaxPool', input), None
            raise NotImplementedError(
                '[Adaptive pool]:input size not accessible')
        dim = input.type().sizes()[2:]
        if output_size == [1] * len(output_size) and type == 'MaxPool':
            return g.op('GlobalMaxPool', input), None

        # compute stride = floor(input_size / output_size)
        s = [int(dim[i] / output_size[i]) for i in range(0, len(dim))]

        # compute kernel_size = input_size - (output_size - 1) * stride
        k = [dim[i] - (output_size[i] - 1) * s[i] for i in range(0, len(dim))]

        # call max_poolxd_with_indices to get indices in the output
        if type == 'MaxPool':
            return fn(g, input, k, k, (0, ) * len(dim), (1, ) * len(dim),
                      False)
        output = g.op(
            type,
            input,
            kernel_shape_i=tuple_fn(k),
            strides_i=tuple_fn(s),
            ceil_mode_i=False)
        return output

    return symbolic_fn


adaptive_avg_pool1d = _adaptive_pool('adaptive_avg_pool1d', 'AveragePool',
                                     _single)
adaptive_avg_pool2d = _adaptive_pool('adaptive_avg_pool2d', 'AveragePool',
                                     _pair)
adaptive_avg_pool3d = _adaptive_pool('adaptive_avg_pool3d', 'AveragePool',
                                     _triple)


def new_full(g,
             self,
             size,
             fill_value,
             dtype,
             layout,
             device,
             pin_memory=False):
    from torch.onnx.symbolic_opset9 import full
    if dtype is None and self.isCompleteTensor():
        dtype = self.type().scalarType()
        dtype = sym_help.scalar_type_to_onnx.index(
            sym_help.cast_pytorch_to_onnx[dtype])
    return full(g, size, fill_value, dtype, layout, device, pin_memory)


@parse_args('v', 'v', 'i', 'i', 'i')
def grid_sampler(g,
                 input,
                 grid,
                 interpolation_mode,
                 padding_mode,
                 align_corners=False):
    return g.op(
        'mmcv::grid_sampler',
        input,
        grid,
        interpolation_mode_i=interpolation_mode,
        padding_mode_i=padding_mode,
        align_corners_i=align_corners)


@parse_args('v', 'i')
def cummax(g, input, dim):
    return g.op('mmcv::cummax', input, dim_i=dim, outputs=2)


@parse_args('v', 'i')
def cummin(g, input, dim):
    return g.op('mmcv::cummin', input, dim_i=dim, outputs=2)


@parse_args('v', 'v', 'is')
def roll(g, input, shifts, dims):
    from packaging import version
    from torch.onnx.symbolic_opset9 import squeeze
    input_shape = g.op('Shape', input)

    need_flatten = len(dims) == 0
    # If dims is not specified, the tensor will be flattened before
    # rolling and then restored to the original shape.
    if need_flatten:
        resize_shape = input_shape
        input = g.op('Reshape', input,
                     g.op('Constant', value_t=torch.LongTensor([1, -1])))
        input_shape = g.op('Shape', input)
        dims = [1]

    for index, dim in enumerate(dims):
        end_size = sym_help._slice_helper(
            g, input_shape, axes=[0], ends=[dim + 1], starts=[dim])
        shift_size = sym_help._slice_helper(
            g, shifts, axes=[0], ends=[index + 1], starts=[index])
        slice_size = g.op('Sub', end_size, shift_size)

        # Can not use Mod because tensorrt does not support
        div_size = g.op('Div', slice_size, end_size)
        slice_size = g.op('Sub', slice_size, g.op('Mul', end_size, div_size))

        if version.parse(torch.__version__) >= version.parse('1.7.0'):
            # add dim=0 for pytorch 1.9.0
            end_size = squeeze(g, end_size, 0)
            slice_size = squeeze(g, slice_size, 0)
        else:
            end_size = g.op('Squeeze', end_size)
            slice_size = g.op('Squeeze', slice_size)
            dim = torch.LongTensor([dim])

        input_slice0 = sym_help._slice_helper(
            g,
            input,
            axes=dim,
            starts=torch.LongTensor([0]),
            ends=slice_size,
            dynamic_slice=True)
        input_slice1 = sym_help._slice_helper(
            g,
            input,
            axes=dim,
            ends=end_size,
            starts=slice_size,
            dynamic_slice=True)

        input = g.op('Concat', input_slice1, input_slice0, axis_i=dim)

    if need_flatten:
        input = g.op('Reshape', input, resize_shape)

    return input


def register_extra_symbolics(opset=11):
    # Following strings of text style are from colorama package
    bright_style, reset_style = '\x1b[1m', '\x1b[0m'
    red_text, blue_text = '\x1b[31m', '\x1b[34m'
    white_background = '\x1b[107m'

    msg = white_background + bright_style + red_text
    msg += 'DeprecationWarning: This function will be deprecated in future. '
    msg += blue_text + 'Welcome to use the unified model deployment toolbox '
    msg += 'MMDeploy: https://github.com/open-mmlab/mmdeploy'
    msg += reset_style
    warnings.warn(msg)

    register_custom_op_symbolic('::one_hot', one_hot, opset)
    register_custom_op_symbolic('::im2col', im2col, opset)
    register_custom_op_symbolic('::topk', topk, opset)
    register_custom_op_symbolic('::softmax', softmax, opset)
    register_custom_op_symbolic('::constant_pad_nd', constant_pad_nd, opset)
    register_custom_op_symbolic('::reflection_pad1d', reflection_pad1d, opset)
    register_custom_op_symbolic('::reflection_pad2d', reflection_pad2d, opset)
    register_custom_op_symbolic('::reflection_pad3d', reflection_pad3d, opset)
    register_custom_op_symbolic('::avg_pool1d', avg_pool1d, opset)
    register_custom_op_symbolic('::avg_pool2d', avg_pool2d, opset)
    register_custom_op_symbolic('::avg_pool3d', avg_pool3d, opset)
    register_custom_op_symbolic('::adaptive_avg_pool1d', adaptive_avg_pool1d,
                                opset)
    register_custom_op_symbolic('::adaptive_avg_pool2d', adaptive_avg_pool2d,
                                opset)
    register_custom_op_symbolic('::adaptive_avg_pool3d', adaptive_avg_pool3d,
                                opset)
    register_custom_op_symbolic('::masked_select', masked_select, opset)
    register_custom_op_symbolic('::upsample_nearest1d', upsample_nearest1d,
                                opset)
    register_custom_op_symbolic('::upsample_nearest2d', upsample_nearest2d,
                                opset)
    register_custom_op_symbolic('::upsample_nearest3d', upsample_nearest3d,
                                opset)
    register_custom_op_symbolic('::upsample_linear1d', upsample_linear1d,
                                opset)
    register_custom_op_symbolic('::upsample_bilinear2d', upsample_bilinear2d,
                                opset)
    register_custom_op_symbolic('::upsample_trilinear3d', upsample_trilinear3d,
                                opset)
    register_custom_op_symbolic('::upsample_bicubic2d', upsample_bicubic2d,
                                opset)
    register_custom_op_symbolic('::new_full', new_full, opset)
    register_custom_op_symbolic('::grid_sampler', grid_sampler, opset)
    register_custom_op_symbolic('::cummax', cummax, opset)
    register_custom_op_symbolic('::cummin', cummin, opset)
    register_custom_op_symbolic('::roll', roll, opset)
