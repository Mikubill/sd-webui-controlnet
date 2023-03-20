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
import sys
from collections import OrderedDict
from typing import Any, List, Optional, Union

import torch
from torch import nn

from .sparse_structure import SparseConvTensor


def is_spconv_module(module: nn.Module) -> bool:
    spconv_modules = (SparseModule, )
    return isinstance(module, spconv_modules)


def is_sparse_conv(module: nn.Module) -> bool:
    from .sparse_conv import SparseConvolution
    return isinstance(module, SparseConvolution)


def _mean_update(vals: Union[int, List], m_vals: Union[int, List],
                 t: float) -> List:
    outputs = []
    if not isinstance(vals, list):
        vals = [vals]
    if not isinstance(m_vals, list):
        m_vals = [m_vals]
    for val, m_val in zip(vals, m_vals):
        output = t / float(t + 1) * m_val + 1 / float(t + 1) * val
        outputs.append(output)
    if len(outputs) == 1:
        outputs = outputs[0]
    return outputs


class SparseModule(nn.Module):
    """place holder, All module subclass from this will take sptensor in
    SparseSequential."""
    pass


class SparseSequential(SparseModule):
    r"""A sequential container. Modules will be added to it in the order they
    are passed in the constructor. Alternatively, an ordered dict of modules
    can also be passed in.

    To make it easier to understand, given is a small example::

    Example:
        >>> # using Sequential:
        >>> from annotator.mmpkg.mmcv.ops import SparseSequential
        >>> model = SparseSequential(
                    SparseConv2d(1,20,5),
                    nn.ReLU(),
                    SparseConv2d(20,64,5),
                    nn.ReLU()
                    )

        >>> # using Sequential with OrderedDict
        >>> model = SparseSequential(OrderedDict([
                      ('conv1', SparseConv2d(1,20,5)),
                      ('relu1', nn.ReLU()),
                      ('conv2', SparseConv2d(20,64,5)),
                      ('relu2', nn.ReLU())
                    ]))

        >>> # using Sequential with kwargs(python 3.6+)
        >>> model = SparseSequential(
                      conv1=SparseConv2d(1,20,5),
                      relu1=nn.ReLU(),
                      conv2=SparseConv2d(20,64,5),
                      relu2=nn.ReLU()
                    )
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError('kwargs only supported in py36+')
            if name in self._modules:
                raise ValueError('name exists.')
            self.add_module(name, module)
        self._sparity_dict = {}

    def __getitem__(self, idx: int) -> torch.Tensor:
        if not (-len(self) <= idx < len(self)):
            raise IndexError(f'index {idx} is out of range')
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    @property
    def sparity_dict(self):
        return self._sparity_dict

    def add(self, module: Any, name: Optional[str] = None) -> None:
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError('name exists')
        self.add_module(name, module)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        for k, module in self._modules.items():
            if is_spconv_module(module):
                assert isinstance(input, SparseConvTensor)
                self._sparity_dict[k] = input.sparity
                input = module(input)
            else:
                if isinstance(input, SparseConvTensor):
                    if input.indices.shape[0] != 0:
                        input.features = module(input.features)
                else:
                    input = module(input)
        return input

    def fused(self):
        from .sparse_conv import SparseConvolution
        mods = [v for k, v in self._modules.items()]
        fused_mods = []
        idx = 0
        while idx < len(mods):
            if is_sparse_conv(mods[idx]):
                if idx < len(mods) - 1 and isinstance(mods[idx + 1],
                                                      nn.BatchNorm1d):
                    new_module = SparseConvolution(
                        ndim=mods[idx].ndim,
                        in_channels=mods[idx].in_channels,
                        out_channels=mods[idx].out_channels,
                        kernel_size=mods[idx].kernel_size,
                        stride=mods[idx].stride,
                        padding=mods[idx].padding,
                        dilation=mods[idx].dilation,
                        groups=mods[idx].groups,
                        bias=True,
                        subm=mods[idx].subm,
                        output_padding=mods[idx].output_padding,
                        transposed=mods[idx].transposed,
                        inverse=mods[idx].inverse,
                        indice_key=mods[idx].indice_key,
                        fused_bn=True,
                    )
                    new_module.load_state_dict(mods[idx].state_dict(), False)
                    new_module.to(mods[idx].weight.device)
                    conv = new_module
                    bn = mods[idx + 1]
                    conv.bias.data.zero_()
                    conv.weight.data[:] = conv.weight.data * bn.weight.data / (
                        torch.sqrt(bn.running_var) + bn.eps)
                    conv.bias.data[:] = (
                        conv.bias.data - bn.running_mean) * bn.weight.data / (
                            torch.sqrt(bn.running_var) + bn.eps) + bn.bias.data
                    fused_mods.append(conv)
                    idx += 2
                else:
                    fused_mods.append(mods[idx])
                    idx += 1
            else:
                fused_mods.append(mods[idx])
                idx += 1
        return SparseSequential(*fused_mods)


class ToDense(SparseModule):
    """convert SparseConvTensor to NCHW dense tensor."""

    def forward(self, x: SparseConvTensor):
        return x.dense()


class RemoveGrid(SparseModule):
    """remove pre-allocated grid buffer."""

    def forward(self, x: SparseConvTensor):
        x.grid = None
        return x
