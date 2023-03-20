# Copyright 2019 Yan Yan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', [
    'get_indice_pairs_2d_forward', 'get_indice_pairs_3d_forward',
    'get_indice_pairs_4d_forward', 'get_indice_pairs_2d_backward',
    'get_indice_pairs_3d_backward', 'indice_conv_forward',
    'indice_conv_backward', 'fused_indice_conv_forward',
    'indice_maxpool_forward', 'indice_maxpool_backward'
])


def get_conv_output_size(input_size, kernel_size, stride, padding, dilation):
    ndim = len(input_size)
    output_size = []
    for i in range(ndim):
        size = (input_size[i] + 2 * padding[i] - dilation[i] *
                (kernel_size[i] - 1) - 1) // stride[i] + 1
        if kernel_size[i] == -1:
            output_size.append(1)
        else:
            output_size.append(size)
    return output_size


def get_deconv_output_size(input_size, kernel_size, stride, padding, dilation,
                           output_padding):
    ndim = len(input_size)
    output_size = []
    for i in range(ndim):
        if kernel_size[i] == -1:
            raise ValueError("deconv don't support kernel_size < 0")
        size = (input_size[i] - 1) * stride[i] - 2 * padding[i] + kernel_size[
            i] + output_padding[i]
        output_size.append(size)
    return output_size


def get_indice_pairs(indices,
                     batch_size,
                     spatial_shape,
                     ksize=3,
                     stride=1,
                     padding=0,
                     dilation=1,
                     out_padding=0,
                     subm=False,
                     transpose=False,
                     grid=None):
    ndim = indices.shape[1] - 1
    if not isinstance(ksize, (list, tuple)):
        ksize = [ksize] * ndim
    if not isinstance(stride, (list, tuple)):
        stride = [stride] * ndim
    if not isinstance(padding, (list, tuple)):
        padding = [padding] * ndim
    if not isinstance(dilation, (list, tuple)):
        dilation = [dilation] * ndim
    if not isinstance(out_padding, (list, tuple)):
        out_padding = [out_padding] * ndim

    for d, s in zip(dilation, stride):
        assert any([s == 1, d == 1]), "don't support this."

    if not subm:
        if transpose:
            out_shape = get_deconv_output_size(spatial_shape, ksize, stride,
                                               padding, dilation, out_padding)
        else:
            out_shape = get_conv_output_size(spatial_shape, ksize, stride,
                                             padding, dilation)

    else:
        out_shape = spatial_shape
    if grid is None:
        if ndim == 2:
            get_indice_pairs_func = ext_module.get_indice_pairs_2d_forward
        elif ndim == 3:
            get_indice_pairs_func = ext_module.get_indice_pairs_3d_forward
        elif ndim == 4:
            get_indice_pairs_func = ext_module.get_indice_pairs_4d_forward
        else:
            raise NotImplementedError
        return get_indice_pairs_func(indices, batch_size, out_shape,
                                     spatial_shape, ksize, stride, padding,
                                     dilation, out_padding, int(subm),
                                     int(transpose))
    else:
        if ndim == 2:
            get_indice_pairs_func = ext_module.get_indice_pairs_2d_backward
        elif ndim == 3:
            get_indice_pairs_func = ext_module.get_indice_pairs_3d_backward
        else:
            raise NotImplementedError
        return get_indice_pairs_func(indices, grid, batch_size, out_shape,
                                     spatial_shape, ksize, stride, padding,
                                     dilation, out_padding, int(subm),
                                     int(transpose))


def indice_conv(features,
                filters,
                indice_pairs,
                indice_pair_num,
                num_activate_out,
                inverse=False,
                subm=False):
    if filters.dtype == torch.float32 or filters.dtype == torch.half:
        return ext_module.indice_conv_forward(features, filters, indice_pairs,
                                              indice_pair_num,
                                              num_activate_out, int(inverse),
                                              int(subm))
    else:
        raise NotImplementedError


def fused_indice_conv(features, filters, bias, indice_pairs, indice_pair_num,
                      num_activate_out, inverse, subm):
    if features.dtype == torch.half or filters.dtypes == torch.float32:
        func = ext_module.fused_indice_conv_forward
    else:
        raise NotImplementedError

    return func(features, filters, bias, indice_pairs, indice_pair_num,
                num_activate_out, int(inverse), int(subm))


def indice_conv_backward(features,
                         filters,
                         out_bp,
                         indice_pairs,
                         indice_pair_num,
                         inverse=False,
                         subm=False):
    if filters.dtype == torch.float32 or filters.dtype == torch.half:
        return ext_module.indice_conv_backward(features, filters, out_bp,
                                               indice_pairs, indice_pair_num,
                                               int(inverse), int(subm))
    else:
        raise NotImplementedError


def indice_maxpool(features, indice_pairs, indice_pair_num, num_activate_out):
    if features.dtype == torch.float32 or features.dtype == torch.half:
        return ext_module.indice_maxpool_forward(features, indice_pairs,
                                                 indice_pair_num,
                                                 num_activate_out)
    else:
        raise NotImplementedError


def indice_maxpool_backward(features, out_features, out_bp, indice_pairs,
                            indice_pair_num):
    if features.dtype == torch.float32 or features.dtype == torch.half:
        return ext_module.indice_maxpool_backward(features, out_features,
                                                  out_bp, indice_pairs,
                                                  indice_pair_num)
    else:
        raise NotImplementedError
