# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair, _single

from annotator.mmpkg.mmcv.utils import IS_MLU_AVAILABLE, deprecated_api_warning
from ..cnn import CONV_LAYERS
from ..utils import ext_loader, print_log

ext_module = ext_loader.load_ext(
    '_ext',
    ['modulated_deform_conv_forward', 'modulated_deform_conv_backward'])


class ModulatedDeformConv2dFunction(Function):

    @staticmethod
    def symbolic(g, input, offset, mask, weight, bias, stride, padding,
                 dilation, groups, deform_groups):
        input_tensors = [input, offset, mask, weight]
        if bias is not None:
            input_tensors.append(bias)
        return g.op(
            'mmcv::MMCVModulatedDeformConv2d',
            *input_tensors,
            stride_i=stride,
            padding_i=padding,
            dilation_i=dilation,
            groups_i=groups,
            deform_groups_i=deform_groups)

    @staticmethod
    def _calculate_sort_index(kernel_h, kernel_w, deformable_group):
        split_num = deformable_group * 2 * kernel_h * kernel_w
        sort_index = list(range(split_num))
        sort_index_fp = (sort_index[1::2] + sort_index[::2])
        sort_index_bp_dict = {i: idx for idx, i in enumerate(sort_index_fp)}
        sort_index_bp = [sort_index_bp_dict[i] for i in sort_index]
        sort_index_fp = torch.IntTensor(sort_index_fp)
        sort_index_bp = torch.IntTensor(sort_index_bp)
        sort_index_fp = sort_index_fp.npu()
        sort_index_bp = sort_index_bp.npu()
        return sort_index_fp, sort_index_bp

    @staticmethod
    def _npu_forward(ctx, input_tensor, offset, mask, weight, bias):
        _, _, kernel_h, kernel_w = weight.shape
        conv2d_bias = bias if len(bias) > 0 else None
        sort_index_fp, sort_index_bp = \
            ModulatedDeformConv2dFunction._calculate_sort_index(
                kernel_w, kernel_h, ctx.deform_groups)
        select_offset = offset.index_select(1, sort_index_fp)
        offset_all = torch.cat([select_offset, mask], dim=1)
        output, offset_out = torch.npu_deformable_conv2d(
            input_tensor,
            weight,
            offset_all,
            conv2d_bias,
            kernel_size=[kernel_w, kernel_h],
            stride=[1, 1, ctx.stride[0], ctx.stride[1]],
            padding=[1, 1, ctx.padding[0], ctx.padding[1]],
            dilation=[1, 1, ctx.dilation[0], ctx.dilation[1]],
            groups=ctx.groups,
            deformable_groups=ctx.deform_groups,
            modulated=True)
        if weight.requires_grad or mask.requires_grad or offset.requires_grad \
                or input_tensor.requires_grad:
            ctx.save_for_backward(input_tensor, weight, offset_out, offset_all,
                                  sort_index_bp)
        return output

    @staticmethod
    def _npu_backward(ctx, grad_output):
        input_tensor, weight, offset_out, offset_all, sort_index_bp = \
            ctx.saved_tensors
        grad_input, grad_weight, grad_offset_all, grad_bias = \
            torch.npu_deformable_conv2dbk(
                input_tensor, grad_output, offset_out, weight, offset_all,
                kernel_size=[weight.shape[3], weight.shape[2]],
                stride=[1, 1, ctx.stride[0], ctx.stride[1]],
                padding=[1, 1, ctx.padding[0], ctx.padding[1]],
                dilation=[1, 1, ctx.dilation[0], ctx.dilation[1]],
                groups=ctx.groups, deformable_groups=ctx.deform_groups,
                modulated=True)
        grad_offset = grad_offset_all.index_select(1, sort_index_bp)
        grad_mask = grad_offset_all[:, grad_offset.shape[1]:, :, :]
        if not ctx.with_bias:
            grad_bias = None
        return (grad_input, grad_offset, grad_mask, grad_weight, grad_bias,
                None, None, None, None, None, None, None, None)

    @staticmethod
    def forward(ctx,
                input: torch.Tensor,
                offset: torch.Tensor,
                mask: torch.Tensor,
                weight: nn.Parameter,
                bias: Optional[nn.Parameter] = None,
                stride: int = 1,
                padding: int = 0,
                dilation: int = 1,
                groups: int = 1,
                deform_groups: int = 1) -> torch.Tensor:
        if input is not None and input.dim() != 4:
            raise ValueError(
                f'Expected 4D tensor as input, got {input.dim()}D tensor \
                  instead.')
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        ctx.deform_groups = deform_groups
        ctx.with_bias = bias is not None
        ctx.device = input.device.type
        if not ctx.with_bias:
            bias = input.new_empty(0)  # fake tensor
        # When pytorch version >= 1.6.0, amp is adopted for fp16 mode;
        # amp won't cast the type of model (float32), but "offset" is cast
        # to float16 by nn.Conv2d automatically, leading to the type
        # mismatch with input (when it is float32) or weight.
        # The flag for whether to use fp16 or amp is the type of "offset",
        # we cast weight and input to temporarily support fp16 and amp
        # whatever the pytorch version is.
        input = input.type_as(offset)
        weight = weight.type_as(input)
        bias = bias.type_as(input)  # type: ignore
        mask = mask.type_as(input)
        if ctx.device == 'npu':
            output = ModulatedDeformConv2dFunction._npu_forward(
                ctx, input, offset, mask, weight, bias)
            return output
        ctx.save_for_backward(input, offset, mask, weight, bias)
        output = input.new_empty(
            ModulatedDeformConv2dFunction._output_size(ctx, input, weight))
        ctx._bufs = [input.new_empty(0), input.new_empty(0)]
        ext_module.modulated_deform_conv_forward(
            input,
            weight,
            bias,
            ctx._bufs[0],
            offset,
            mask,
            output,
            ctx._bufs[1],
            kernel_h=weight.size(2),
            kernel_w=weight.size(3),
            stride_h=ctx.stride[0],
            stride_w=ctx.stride[1],
            pad_h=ctx.padding[0],
            pad_w=ctx.padding[1],
            dilation_h=ctx.dilation[0],
            dilation_w=ctx.dilation[1],
            group=ctx.groups,
            deformable_group=ctx.deform_groups,
            with_bias=ctx.with_bias)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        if ctx.device == 'npu':
            return ModulatedDeformConv2dFunction._npu_backward(
                ctx, grad_output)
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_offset = torch.zeros_like(offset)
        grad_mask = torch.zeros_like(mask)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
        grad_output = grad_output.contiguous()
        ext_module.modulated_deform_conv_backward(
            input,
            weight,
            bias,
            ctx._bufs[0],
            offset,
            mask,
            ctx._bufs[1],
            grad_input,
            grad_weight,
            grad_bias,
            grad_offset,
            grad_mask,
            grad_output,
            kernel_h=weight.size(2),
            kernel_w=weight.size(3),
            stride_h=ctx.stride[0],
            stride_w=ctx.stride[1],
            pad_h=ctx.padding[0],
            pad_w=ctx.padding[1],
            dilation_h=ctx.dilation[0],
            dilation_w=ctx.dilation[1],
            group=ctx.groups,
            deformable_group=ctx.deform_groups,
            with_bias=ctx.with_bias)
        if not ctx.with_bias:
            grad_bias = None

        return (grad_input, grad_offset, grad_mask, grad_weight, grad_bias,
                None, None, None, None, None)

    @staticmethod
    def _output_size(ctx, input, weight):
        channels = weight.size(0)
        output_size = (input.size(0), channels)
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = ctx.padding[d]
            kernel = ctx.dilation[d] * (weight.size(d + 2) - 1) + 1
            stride_ = ctx.stride[d]
            output_size += ((in_size + (2 * pad) - kernel) // stride_ + 1, )
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError(
                'convolution input is too small (output would be ' +
                'x'.join(map(str, output_size)) + ')')
        return output_size


modulated_deform_conv2d = ModulatedDeformConv2dFunction.apply


class ModulatedDeformConv2d(nn.Module):

    @deprecated_api_warning({'deformable_groups': 'deform_groups'},
                            cls_name='ModulatedDeformConv2d')
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int]],
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 deform_groups: int = 1,
                 bias: Union[bool, str] = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deform_groups = deform_groups
        # enable compatibility with nn.Conv2d
        self.transposed = False
        self.output_padding = _single(0)

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups,
                         *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.init_weights()

    def init_weights(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x: torch.Tensor, offset: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)


@CONV_LAYERS.register_module('DCNv2')
class ModulatedDeformConv2dPack(ModulatedDeformConv2d):
    """A ModulatedDeformable Conv Encapsulation that acts as normal Conv
    layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int): Same as nn.Conv2d, while tuple is not supported.
        padding (int): Same as nn.Conv2d, while tuple is not supported.
        dilation (int): Same as nn.Conv2d, while tuple is not supported.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    _version = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True)
        self.init_weights()

    def init_weights(self) -> None:
        super().init_weights()
        if hasattr(self, 'conv_offset'):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if version is None or version < 2:
            # the key is different in early versions
            # In version < 2, ModulatedDeformConvPack
            # loads previous benchmark models.
            if (prefix + 'conv_offset.weight' not in state_dict
                    and prefix[:-1] + '_offset.weight' in state_dict):
                state_dict[prefix + 'conv_offset.weight'] = state_dict.pop(
                    prefix[:-1] + '_offset.weight')
            if (prefix + 'conv_offset.bias' not in state_dict
                    and prefix[:-1] + '_offset.bias' in state_dict):
                state_dict[prefix +
                           'conv_offset.bias'] = state_dict.pop(prefix[:-1] +
                                                                '_offset.bias')

        if version is not None and version > 1:
            print_log(
                f'ModulatedDeformConvPack {prefix.rstrip(".")} is upgraded to '
                'version 2.',
                logger='root')

        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)


if IS_MLU_AVAILABLE:
    import torchvision
    from torchvision.ops import deform_conv2d as tv_deform_conv2d

    from annotator.mmpkg.mmcv.utils import digit_version

    @CONV_LAYERS.register_module('DCNv2', force=True)
    class ModulatedDeformConv2dPack_MLU(ModulatedDeformConv2d):
        """This class is the DCNv2 implementation of the MLU device. The MLU
        backend support of the operator has been implemented in torchvision.
        The mmcv registration mechanism is used for multiplexing here. The
        torchvision implementation of DCNv2 is called.

        Args:
            in_channels (int): Same as nn.Conv2d.
            out_channels (int): Same as nn.Conv2d.
            kernel_size (int or tuple[int]): Same as nn.Conv2d.
            stride (int): Same as nn.Conv2d, while tuple is not supported.
            padding (int): Same as nn.Conv2d, while tuple is not supported.
            dilation (int): Same as nn.Conv2d, while tuple is not supported.
            groups (int): Same as nn.Conv2d.
            bias (bool or str): If specified as `auto`, it will be decided by
                the norm_cfg. Bias will be set as True if norm_cfg is None,
                otherwise False.
        """

        def __init__(self, *args, **kwargs):
            assert digit_version(torchvision.__version__) >= digit_version(
                '0.10.0a0'), 'the version of torchvision should be >= 0.10.0'
            super().__init__(*args, **kwargs)
            self.conv_offset = nn.Conv2d(
                self.in_channels,
                self.deform_groups * 3 * self.kernel_size[0] *
                self.kernel_size[1],
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                bias=True)
            self.init_weights()

        def init_weights(self):
            super().init_weights()
            if hasattr(self, 'conv_offset'):
                self.conv_offset.weight.data.zero_()
                self.conv_offset.bias.data.zero_()

        def forward(self, x):
            out = self.conv_offset(x)
            o1, o2, mask = torch.chunk(out, 3, dim=1)
            offset = torch.cat((o1, o2), dim=1)
            mask = torch.sigmoid(mask)
            x = x.type_as(offset)
            weight = self.weight.type_as(x)
            mask = mask.type_as(x)
            return tv_deform_conv2d(
                x,
                offset,
                weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                mask=mask)
