# ComfyUI (https://github.com/comfyanonymous/ComfyUI)
# Copyright (C) 2023

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import torch


class ControlLoraOps:
    class Linear(torch.nn.Module):
        def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            device=None,
            dtype=None,
        ) -> None:
            factory_kwargs = {"device": device, "dtype": dtype}
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = None
            self.up = None
            self.down = None
            self.bias = None

        def forward(self, input):
            if self.up is not None:
                return torch.nn.functional.linear(
                    input,
                    self.weight
                    + (
                        torch.mm(
                            self.up.flatten(start_dim=1), self.down.flatten(start_dim=1)
                        )
                    )
                    .reshape(self.weight.shape)
                    .type(self.weight.dtype),
                    self.bias,
                )
            else:
                return torch.nn.functional.linear(input, self.weight, self.bias)

    class Conv2d(torch.nn.Module):
        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
            device=None,
            dtype=None,
        ):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding
            self.dilation = dilation
            self.transposed = False
            self.output_padding = 0
            self.groups = groups
            self.padding_mode = padding_mode

            self.weight = None
            self.bias = None
            self.up = None
            self.down = None

        def forward(self, input):
            if self.up is not None:
                return torch.nn.functional.conv2d(
                    input,
                    self.weight
                    + (
                        torch.mm(
                            self.up.flatten(start_dim=1), self.down.flatten(start_dim=1)
                        )
                    )
                    .reshape(self.weight.shape)
                    .type(self.weight.dtype),
                    self.bias,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                )
            else:
                return torch.nn.functional.conv2d(
                    input,
                    self.weight,
                    self.bias,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                )

    def conv_nd(self, dims, *args, **kwargs):
        if dims == 2:
            return self.Conv2d(*args, **kwargs)
        else:
            raise ValueError(f"unsupported dimensions: {dims}")
