# Copyright (c) OpenMMLab. All rights reserved.
import annotator.mmpkg.mmcv as mmcv
import torch
import torch.nn as nn
from annotator.mmpkg.mmcv.cnn import (build_conv_layer, build_norm_layer, constant_init,
                      normal_init)
from annotator.mmpkg.mmcv.utils import digit_version
from torch.nn.modules.batchnorm import _BatchNorm

from annotator.mmpkg.mmpose.models.utils.ops import resize
from ..backbones.resnet import BasicBlock, Bottleneck
from ..builder import NECKS

try:
    from annotator.mmpkg.mmcv.ops import DeformConv2d
    has_mmcv_full = True
except (ImportError, ModuleNotFoundError):
    has_mmcv_full = False


@NECKS.register_module()
class PoseWarperNeck(nn.Module):
    """PoseWarper neck.

    `"Learning temporal pose estimation from sparsely-labeled videos"
    <https://arxiv.org/abs/1906.04016>`_.

    Args:
        in_channels (int): Number of input channels from backbone
        out_channels (int): Number of output channels
        inner_channels (int): Number of intermediate channels of the res block
        deform_groups (int): Number of groups in the deformable conv
        dilations (list|tuple): different dilations of the offset conv layers
        trans_conv_kernel (int): the kernel of the trans conv layer, which is
            used to get heatmap from the output of backbone. Default: 1
        res_blocks_cfg (dict|None): config of residual blocks. If None,
            use the default values. If not None, it should contain the
            following keys:

            - block (str): the type of residual block, Default: 'BASIC'.
            - num_blocks (int):  the number of blocks, Default: 20.

        offsets_kernel (int): the kernel of offset conv layer.
        deform_conv_kernel (int): the kernel of defomrable conv layer.
        in_index (int|Sequence[int]): Input feature index. Default: 0
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            Default: None.

            - 'resize_concat': Multiple feature maps will be resize to \
                the same size as first one and than concat together. \
                Usually used in FCN head of HRNet.
            - 'multiple_select': Multiple feature maps will be bundle into \
                a list and passed into decode head.
            - None: Only one select feature map is allowed.

        freeze_trans_layer (bool): Whether to freeze the transition layer
            (stop grad and set eval mode). Default: True.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        im2col_step (int): the argument `im2col_step` in deformable conv,
            Default: 80.
    """
    blocks_dict = {'BASIC': BasicBlock, 'BOTTLENECK': Bottleneck}
    minimum_mmcv_version = '1.3.17'

    def __init__(self,
                 in_channels,
                 out_channels,
                 inner_channels,
                 deform_groups=17,
                 dilations=(3, 6, 12, 18, 24),
                 trans_conv_kernel=1,
                 res_blocks_cfg=None,
                 offsets_kernel=3,
                 deform_conv_kernel=3,
                 in_index=0,
                 input_transform=None,
                 freeze_trans_layer=True,
                 norm_eval=False,
                 im2col_step=80):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inner_channels = inner_channels
        self.deform_groups = deform_groups
        self.dilations = dilations
        self.trans_conv_kernel = trans_conv_kernel
        self.res_blocks_cfg = res_blocks_cfg
        self.offsets_kernel = offsets_kernel
        self.deform_conv_kernel = deform_conv_kernel
        self.in_index = in_index
        self.input_transform = input_transform
        self.freeze_trans_layer = freeze_trans_layer
        self.norm_eval = norm_eval
        self.im2col_step = im2col_step

        identity_trans_layer = False

        assert trans_conv_kernel in [0, 1, 3]
        kernel_size = trans_conv_kernel
        if kernel_size == 3:
            padding = 1
        elif kernel_size == 1:
            padding = 0
        else:
            # 0 for Identity mapping.
            identity_trans_layer = True

        if identity_trans_layer:
            self.trans_layer = nn.Identity()
        else:
            self.trans_layer = build_conv_layer(
                cfg=dict(type='Conv2d'),
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding)

        # build chain of residual blocks
        if res_blocks_cfg is not None and not isinstance(res_blocks_cfg, dict):
            raise TypeError('res_blocks_cfg should be dict or None.')

        if res_blocks_cfg is None:
            block_type = 'BASIC'
            num_blocks = 20
        else:
            block_type = res_blocks_cfg.get('block', 'BASIC')
            num_blocks = res_blocks_cfg.get('num_blocks', 20)

        block = self.blocks_dict[block_type]

        res_layers = []
        downsample = nn.Sequential(
            build_conv_layer(
                cfg=dict(type='Conv2d'),
                in_channels=out_channels,
                out_channels=inner_channels,
                kernel_size=1,
                stride=1,
                bias=False),
            build_norm_layer(dict(type='BN'), inner_channels)[1])
        res_layers.append(
            block(
                in_channels=out_channels,
                out_channels=inner_channels,
                downsample=downsample))

        for _ in range(1, num_blocks):
            res_layers.append(block(inner_channels, inner_channels))
        self.offset_feats = nn.Sequential(*res_layers)

        # build offset layers
        self.num_offset_layers = len(dilations)
        assert self.num_offset_layers > 0, 'Number of offset layers ' \
            'should be larger than 0.'

        target_offset_channels = 2 * offsets_kernel**2 * deform_groups

        offset_layers = [
            build_conv_layer(
                cfg=dict(type='Conv2d'),
                in_channels=inner_channels,
                out_channels=target_offset_channels,
                kernel_size=offsets_kernel,
                stride=1,
                dilation=dilations[i],
                padding=dilations[i],
                bias=False,
            ) for i in range(self.num_offset_layers)
        ]
        self.offset_layers = nn.ModuleList(offset_layers)

        # build deformable conv layers
        assert digit_version(mmcv.__version__) >= \
            digit_version(self.minimum_mmcv_version), \
            f'Current MMCV version: {mmcv.__version__}, ' \
            f'but MMCV >= {self.minimum_mmcv_version} is required, see ' \
            f'https://github.com/open-mmlab/mmcv/issues/1440, ' \
            f'Please install the latest MMCV.'

        if has_mmcv_full:
            deform_conv_layers = [
                DeformConv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=deform_conv_kernel,
                    stride=1,
                    padding=int(deform_conv_kernel / 2) * dilations[i],
                    dilation=dilations[i],
                    deform_groups=deform_groups,
                    im2col_step=self.im2col_step,
                ) for i in range(self.num_offset_layers)
            ]
        else:
            raise ImportError('Please install the full version of mmcv '
                              'to use `DeformConv2d`.')

        self.deform_conv_layers = nn.ModuleList(deform_conv_layers)

        self.freeze_layers()

    def freeze_layers(self):
        if self.freeze_trans_layer:
            self.trans_layer.eval()

            for param in self.trans_layer.parameters():
                param.requires_grad = False

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, DeformConv2d):
                filler = torch.zeros([
                    m.weight.size(0),
                    m.weight.size(1),
                    m.weight.size(2),
                    m.weight.size(3)
                ],
                                     dtype=torch.float32,
                                     device=m.weight.device)
                for k in range(m.weight.size(0)):
                    filler[k, k,
                           int(m.weight.size(2) / 2),
                           int(m.weight.size(3) / 2)] = 1.0
                m.weight = torch.nn.Parameter(filler)
                m.weight.requires_grad = True

        # posewarper offset layer weight initialization
        for m in self.offset_layers.modules():
            constant_init(m, 0)

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor] | Tensor): multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """
        if not isinstance(inputs, list):
            return inputs

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs, frame_weight):
        assert isinstance(inputs, (list, tuple)), 'PoseWarperNeck inputs ' \
            'should be list or tuple, even though the length is 1, ' \
            'for unified processing.'

        output_heatmap = 0
        if len(inputs) > 1:
            inputs = [self._transform_inputs(input) for input in inputs]
            inputs = [self.trans_layer(input) for input in inputs]

            # calculate difference features
            diff_features = [
                self.offset_feats(inputs[0] - input) for input in inputs
            ]

            for i in range(len(inputs)):
                if frame_weight[i] == 0:
                    continue
                warped_heatmap = 0
                for j in range(self.num_offset_layers):
                    offset = (self.offset_layers[j](diff_features[i]))
                    warped_heatmap_tmp = self.deform_conv_layers[j](inputs[i],
                                                                    offset)
                    warped_heatmap += warped_heatmap_tmp / \
                        self.num_offset_layers

                output_heatmap += warped_heatmap * frame_weight[i]

        else:
            inputs = inputs[0]
            inputs = self._transform_inputs(inputs)
            inputs = self.trans_layer(inputs)

            num_frames = len(frame_weight)
            batch_size = inputs.size(0) // num_frames
            ref_x = inputs[:batch_size]
            ref_x_tiled = ref_x.repeat(num_frames, 1, 1, 1)

            offset_features = self.offset_feats(ref_x_tiled - inputs)

            warped_heatmap = 0
            for j in range(self.num_offset_layers):
                offset = self.offset_layers[j](offset_features)

                warped_heatmap_tmp = self.deform_conv_layers[j](inputs, offset)
                warped_heatmap += warped_heatmap_tmp / self.num_offset_layers

            for i in range(num_frames):
                if frame_weight[i] == 0:
                    continue
                output_heatmap += warped_heatmap[i * batch_size:(i + 1) *
                                                 batch_size] * frame_weight[i]

        return output_heatmap

    def train(self, mode=True):
        """Convert the model into training mode."""
        super().train(mode)
        self.freeze_layers()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
