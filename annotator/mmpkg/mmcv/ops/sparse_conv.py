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
import math

import numpy as np
import torch
from torch.nn import init
from torch.nn.parameter import Parameter

from ..cnn import CONV_LAYERS
from . import sparse_functional as Fsp
from . import sparse_ops as ops
from .sparse_modules import SparseModule
from .sparse_structure import SparseConvTensor


def _calculate_fan_in_and_fan_out_hwio(tensor):
    dimensions = tensor.ndimension()
    if dimensions < 2:
        raise ValueError('fan in and fan out can not be computed for tensor'
                         'with fewer than 2 dimensions')

    if dimensions == 2:  # Linear
        fan_in = tensor.size(-2)
        fan_out = tensor.size(-1)
    else:
        num_input_fmaps = tensor.size(-2)
        num_output_fmaps = tensor.size(-1)
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[..., 0, 0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


class SparseConvolution(SparseModule):

    def __init__(self,
                 ndim,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 subm=False,
                 output_padding=0,
                 transposed=False,
                 inverse=False,
                 indice_key=None,
                 fused_bn=False):
        super().__init__()
        assert groups == 1
        if not isinstance(kernel_size, (list, tuple)):
            kernel_size = [kernel_size] * ndim
        if not isinstance(stride, (list, tuple)):
            stride = [stride] * ndim
        if not isinstance(padding, (list, tuple)):
            padding = [padding] * ndim
        if not isinstance(dilation, (list, tuple)):
            dilation = [dilation] * ndim
        if not isinstance(output_padding, (list, tuple)):
            output_padding = [output_padding] * ndim

        for d, s in zip(dilation, stride):
            assert any([s == 1, d == 1]), "don't support this."

        self.ndim = ndim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv1x1 = np.prod(kernel_size) == 1
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.inverse = inverse
        self.output_padding = output_padding
        self.groups = groups
        self.subm = subm
        self.indice_key = indice_key
        self.fused_bn = fused_bn

        self.weight = Parameter(
            torch.Tensor(*kernel_size, in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out_hwio(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        assert isinstance(input, SparseConvTensor)
        features = input.features
        device = features.device
        indices = input.indices
        spatial_shape = input.spatial_shape
        batch_size = input.batch_size
        if not self.subm:
            if self.transposed:
                out_spatial_shape = ops.get_deconv_output_size(
                    spatial_shape, self.kernel_size, self.stride, self.padding,
                    self.dilation, self.output_padding)
            else:
                out_spatial_shape = ops.get_conv_output_size(
                    spatial_shape, self.kernel_size, self.stride, self.padding,
                    self.dilation)

        else:
            out_spatial_shape = spatial_shape

        if self.conv1x1:
            features = torch.mm(
                input.features,
                self.weight.view(self.in_channels, self.out_channels))
            if self.bias is not None:
                features += self.bias
            out_tensor = SparseConvTensor(features, input.indices,
                                          input.spatial_shape,
                                          input.batch_size)
            out_tensor.indice_dict = input.indice_dict
            out_tensor.grid = input.grid
            return out_tensor
        data = input.find_indice_pair(self.indice_key)
        if self.inverse:
            assert data is not None and self.indice_key is not None
            _, outids, indice_pairs, indice_pair_num, out_spatial_shape = data
            assert indice_pairs.shape[0] == np.prod(
                self.kernel_size
            ), 'inverse conv must have same kernel size as its couple conv'
        else:
            if self.indice_key is not None and data is not None:
                outids, _, indice_pairs, indice_pair_num, _ = data
            else:
                outids, indice_pairs, indice_pair_num = ops.get_indice_pairs(
                    indices,
                    batch_size,
                    spatial_shape,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.output_padding,
                    self.subm,
                    self.transposed,
                    grid=input.grid)
                input.indice_dict[self.indice_key] = (outids, indices,
                                                      indice_pairs,
                                                      indice_pair_num,
                                                      spatial_shape)
        if self.fused_bn:
            assert self.bias is not None
            out_features = ops.fused_indice_conv(features, self.weight,
                                                 self.bias,
                                                 indice_pairs.to(device),
                                                 indice_pair_num,
                                                 outids.shape[0], self.inverse,
                                                 self.subm)
        else:
            if self.subm:
                out_features = Fsp.indice_subm_conv(features, self.weight,
                                                    indice_pairs.to(device),
                                                    indice_pair_num,
                                                    outids.shape[0])
            else:
                if self.inverse:
                    out_features = Fsp.indice_inverse_conv(
                        features, self.weight, indice_pairs.to(device),
                        indice_pair_num, outids.shape[0])
                else:
                    out_features = Fsp.indice_conv(features, self.weight,
                                                   indice_pairs.to(device),
                                                   indice_pair_num,
                                                   outids.shape[0])

            if self.bias is not None:
                out_features += self.bias
        out_tensor = SparseConvTensor(out_features, outids, out_spatial_shape,
                                      batch_size)
        out_tensor.indice_dict = input.indice_dict
        out_tensor.grid = input.grid
        return out_tensor


@CONV_LAYERS.register_module()
class SparseConv2d(SparseConvolution):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 indice_key=None):
        super().__init__(
            2,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            indice_key=indice_key)


@CONV_LAYERS.register_module()
class SparseConv3d(SparseConvolution):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 indice_key=None):
        super().__init__(
            3,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            indice_key=indice_key)


@CONV_LAYERS.register_module()
class SparseConv4d(SparseConvolution):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 indice_key=None):
        super().__init__(
            4,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            indice_key=indice_key)


@CONV_LAYERS.register_module()
class SparseConvTranspose2d(SparseConvolution):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 indice_key=None):
        super().__init__(
            2,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            transposed=True,
            indice_key=indice_key)


@CONV_LAYERS.register_module()
class SparseConvTranspose3d(SparseConvolution):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 indice_key=None):
        super().__init__(
            3,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            transposed=True,
            indice_key=indice_key)


@CONV_LAYERS.register_module()
class SparseInverseConv2d(SparseConvolution):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 indice_key=None,
                 bias=True):
        super().__init__(
            2,
            in_channels,
            out_channels,
            kernel_size,
            bias=bias,
            inverse=True,
            indice_key=indice_key)


@CONV_LAYERS.register_module()
class SparseInverseConv3d(SparseConvolution):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 indice_key=None,
                 bias=True):
        super().__init__(
            3,
            in_channels,
            out_channels,
            kernel_size,
            bias=bias,
            inverse=True,
            indice_key=indice_key)


@CONV_LAYERS.register_module()
class SubMConv2d(SparseConvolution):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 indice_key=None):
        super().__init__(
            2,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            True,
            indice_key=indice_key)


@CONV_LAYERS.register_module()
class SubMConv3d(SparseConvolution):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 indice_key=None):
        super().__init__(
            3,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            True,
            indice_key=indice_key)


@CONV_LAYERS.register_module()
class SubMConv4d(SparseConvolution):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 indice_key=None):
        super().__init__(
            4,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            True,
            indice_key=indice_key)
