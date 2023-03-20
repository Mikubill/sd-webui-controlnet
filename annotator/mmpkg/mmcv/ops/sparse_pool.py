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

# import sparse_functional as Fsp
# import sparse_ops as ops
from .sparse_functional import indice_maxpool
from .sparse_modules import SparseModule
from .sparse_ops import get_conv_output_size, get_indice_pairs
from .sparse_structure import SparseConvTensor


class SparseMaxPool(SparseModule):

    def __init__(self,
                 ndim,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 subm=False):
        super().__init__()
        if not isinstance(kernel_size, (list, tuple)):
            kernel_size = [kernel_size] * ndim
        if not isinstance(stride, (list, tuple)):
            stride = [stride] * ndim
        if not isinstance(padding, (list, tuple)):
            padding = [padding] * ndim
        if not isinstance(dilation, (list, tuple)):
            dilation = [dilation] * ndim

        self.ndim = ndim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.subm = subm
        self.dilation = dilation

    def forward(self, input):
        assert isinstance(input, SparseConvTensor)
        features = input.features
        device = features.device
        indices = input.indices
        spatial_shape = input.spatial_shape
        batch_size = input.batch_size
        if not self.subm:
            out_spatial_shape = get_conv_output_size(spatial_shape,
                                                     self.kernel_size,
                                                     self.stride, self.padding,
                                                     self.dilation)
        else:
            out_spatial_shape = spatial_shape
        outids, indice_pairs, indice_pairs_num = get_indice_pairs(
            indices, batch_size, spatial_shape, self.kernel_size, self.stride,
            self.padding, self.dilation, 0, self.subm)

        out_features = indice_maxpool(features, indice_pairs.to(device),
                                      indice_pairs_num.to(device),
                                      outids.shape[0])
        out_tensor = SparseConvTensor(out_features, outids, out_spatial_shape,
                                      batch_size)
        out_tensor.indice_dict = input.indice_dict
        out_tensor.grid = input.grid
        return out_tensor


class SparseMaxPool2d(SparseMaxPool):

    def __init__(self, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__(2, kernel_size, stride, padding, dilation)


class SparseMaxPool3d(SparseMaxPool):

    def __init__(self, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__(3, kernel_size, stride, padding, dilation)
