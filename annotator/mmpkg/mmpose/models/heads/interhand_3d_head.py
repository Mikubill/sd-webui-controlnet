# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from annotator.mmpkg.mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, normal_init)

from annotator.mmpkg.mmpose.core.evaluation.top_down_eval import (
    keypoints_from_heatmaps3d, multilabel_classification_accuracy)
from annotator.mmpkg.mmpose.core.post_processing import flip_back
from annotator.mmpkg.mmpose.models.builder import build_loss
from annotator.mmpkg.mmpose.models.necks import GlobalAveragePooling
from ..builder import HEADS


class Heatmap3DHead(nn.Module):
    """Heatmap3DHead is a sub-module of Interhand3DHead, and outputs 3D
    heatmaps. Heatmap3DHead is composed of (>=0) number of deconv layers and a
    simple conv2d layer.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        depth_size (int): Number of depth discretization size
        num_deconv_layers (int): Number of deconv layers.
        num_deconv_layers should >= 0. Note that 0 means no deconv layers.
        num_deconv_filters (list|tuple): Number of filters.
        num_deconv_kernels (list|tuple): Kernel sizes.
        extra (dict): Configs for extra conv layers. Default: None
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 depth_size=64,
                 num_deconv_layers=3,
                 num_deconv_filters=(256, 256, 256),
                 num_deconv_kernels=(4, 4, 4),
                 extra=None):

        super().__init__()

        assert out_channels % depth_size == 0
        self.depth_size = depth_size
        self.in_channels = in_channels

        if extra is not None and not isinstance(extra, dict):
            raise TypeError('extra should be dict or None.')

        if num_deconv_layers > 0:
            self.deconv_layers = self._make_deconv_layer(
                num_deconv_layers,
                num_deconv_filters,
                num_deconv_kernels,
            )
        elif num_deconv_layers == 0:
            self.deconv_layers = nn.Identity()
        else:
            raise ValueError(
                f'num_deconv_layers ({num_deconv_layers}) should >= 0.')

        identity_final_layer = False
        if extra is not None and 'final_conv_kernel' in extra:
            assert extra['final_conv_kernel'] in [0, 1, 3]
            if extra['final_conv_kernel'] == 3:
                padding = 1
            elif extra['final_conv_kernel'] == 1:
                padding = 0
            else:
                # 0 for Identity mapping.
                identity_final_layer = True
            kernel_size = extra['final_conv_kernel']
        else:
            kernel_size = 1
            padding = 0

        if identity_final_layer:
            self.final_layer = nn.Identity()
        else:
            conv_channels = num_deconv_filters[
                -1] if num_deconv_layers > 0 else self.in_channels

            layers = []
            if extra is not None:
                num_conv_layers = extra.get('num_conv_layers', 0)
                num_conv_kernels = extra.get('num_conv_kernels',
                                             [1] * num_conv_layers)

                for i in range(num_conv_layers):
                    layers.append(
                        build_conv_layer(
                            dict(type='Conv2d'),
                            in_channels=conv_channels,
                            out_channels=conv_channels,
                            kernel_size=num_conv_kernels[i],
                            stride=1,
                            padding=(num_conv_kernels[i] - 1) // 2))
                    layers.append(
                        build_norm_layer(dict(type='BN'), conv_channels)[1])
                    layers.append(nn.ReLU(inplace=True))

            layers.append(
                build_conv_layer(
                    cfg=dict(type='Conv2d'),
                    in_channels=conv_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding))

            if len(layers) > 1:
                self.final_layer = nn.Sequential(*layers)
            else:
                self.final_layer = layers[0]

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""
        if num_layers != len(num_filters):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_filters({len(num_filters)})'
            raise ValueError(error_msg)
        if num_layers != len(num_kernels):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_kernels({len(num_kernels)})'
            raise ValueError(error_msg)

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                build_upsample_layer(
                    dict(type='deconv'),
                    in_channels=self.in_channels,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = planes

        return nn.Sequential(*layers)

    @staticmethod
    def _get_deconv_cfg(deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

        return deconv_kernel, padding, output_padding

    def forward(self, x):
        """Forward function."""
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        N, C, H, W = x.shape
        # reshape the 2D heatmap to 3D heatmap
        x = x.reshape(N, C // self.depth_size, self.depth_size, H, W)
        return x

    def init_weights(self):
        """Initialize model weights."""
        for _, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)


class Heatmap1DHead(nn.Module):
    """Heatmap1DHead is a sub-module of Interhand3DHead, and outputs 1D
    heatmaps.

    Args:
        in_channels (int): Number of input channels
        heatmap_size (int): Heatmap size
        hidden_dims (list|tuple): Number of feature dimension of FC layers.
    """

    def __init__(self, in_channels=2048, heatmap_size=64, hidden_dims=(512, )):
        super().__init__()

        self.in_channels = in_channels
        self.heatmap_size = heatmap_size

        feature_dims = [in_channels, *hidden_dims, heatmap_size]
        self.fc = self._make_linear_layers(feature_dims, relu_final=False)

    def soft_argmax_1d(self, heatmap1d):
        heatmap1d = F.softmax(heatmap1d, 1)
        accu = heatmap1d * torch.arange(
            self.heatmap_size, dtype=heatmap1d.dtype,
            device=heatmap1d.device)[None, :]
        coord = accu.sum(dim=1)
        return coord

    def _make_linear_layers(self, feat_dims, relu_final=False):
        """Make linear layers."""
        layers = []
        for i in range(len(feat_dims) - 1):
            layers.append(nn.Linear(feat_dims[i], feat_dims[i + 1]))
            if i < len(feat_dims) - 2 or \
                    (i == len(feat_dims) - 2 and relu_final):
                layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward function."""
        heatmap1d = self.fc(x)
        value = self.soft_argmax_1d(heatmap1d).view(-1, 1)
        return value

    def init_weights(self):
        """Initialize model weights."""
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                normal_init(m, mean=0, std=0.01, bias=0)


class MultilabelClassificationHead(nn.Module):
    """MultilabelClassificationHead is a sub-module of Interhand3DHead, and
    outputs hand type classification.

    Args:
        in_channels (int): Number of input channels
        num_labels (int): Number of labels
        hidden_dims (list|tuple): Number of hidden dimension of FC layers.
    """

    def __init__(self, in_channels=2048, num_labels=2, hidden_dims=(512, )):
        super().__init__()

        self.in_channels = in_channels
        self.num_labesl = num_labels

        feature_dims = [in_channels, *hidden_dims, num_labels]
        self.fc = self._make_linear_layers(feature_dims, relu_final=False)

    def _make_linear_layers(self, feat_dims, relu_final=False):
        """Make linear layers."""
        layers = []
        for i in range(len(feat_dims) - 1):
            layers.append(nn.Linear(feat_dims[i], feat_dims[i + 1]))
            if i < len(feat_dims) - 2 or \
                    (i == len(feat_dims) - 2 and relu_final):
                layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward function."""
        labels = torch.sigmoid(self.fc(x))
        return labels

    def init_weights(self):
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                normal_init(m, mean=0, std=0.01, bias=0)


@HEADS.register_module()
class Interhand3DHead(nn.Module):
    """Interhand 3D head of paper ref: Gyeongsik Moon. "InterHand2.6M: A
    Dataset and Baseline for 3D Interacting Hand Pose Estimation from a Single
    RGB Image".

    Args:
        keypoint_head_cfg (dict): Configs of Heatmap3DHead for hand
            keypoint estimation.
        root_head_cfg (dict): Configs of Heatmap1DHead for relative
            hand root depth estimation.
        hand_type_head_cfg (dict): Configs of MultilabelClassificationHead
            for hand type classification.
        loss_keypoint (dict): Config for keypoint loss. Default: None.
        loss_root_depth (dict): Config for relative root depth loss.
            Default: None.
        loss_hand_type (dict): Config for hand type classification
            loss. Default: None.
    """

    def __init__(self,
                 keypoint_head_cfg,
                 root_head_cfg,
                 hand_type_head_cfg,
                 loss_keypoint=None,
                 loss_root_depth=None,
                 loss_hand_type=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()

        # build sub-module heads
        self.right_hand_head = Heatmap3DHead(**keypoint_head_cfg)
        self.left_hand_head = Heatmap3DHead(**keypoint_head_cfg)
        self.root_head = Heatmap1DHead(**root_head_cfg)
        self.hand_type_head = MultilabelClassificationHead(
            **hand_type_head_cfg)
        self.neck = GlobalAveragePooling()

        # build losses
        self.keypoint_loss = build_loss(loss_keypoint)
        self.root_depth_loss = build_loss(loss_root_depth)
        self.hand_type_loss = build_loss(loss_hand_type)
        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg
        self.target_type = self.test_cfg.get('target_type', 'GaussianHeatmap')

    def init_weights(self):
        self.left_hand_head.init_weights()
        self.right_hand_head.init_weights()
        self.root_head.init_weights()
        self.hand_type_head.init_weights()

    def get_loss(self, output, target, target_weight):
        """Calculate loss for hand keypoint heatmaps, relative root depth and
        hand type.

        Args:
            output (list[Tensor]): a list of outputs from multiple heads.
            target (list[Tensor]): a list of targets for multiple heads.
            target_weight (list[Tensor]): a list of targets weight for
                multiple heads.
        """
        losses = dict()

        # hand keypoint loss
        assert not isinstance(self.keypoint_loss, nn.Sequential)
        out, tar, tar_weight = output[0], target[0], target_weight[0]
        assert tar.dim() == 5 and tar_weight.dim() == 3
        losses['hand_loss'] = self.keypoint_loss(out, tar, tar_weight)

        # relative root depth loss
        assert not isinstance(self.root_depth_loss, nn.Sequential)
        out, tar, tar_weight = output[1], target[1], target_weight[1]
        assert tar.dim() == 2 and tar_weight.dim() == 2
        losses['rel_root_loss'] = self.root_depth_loss(out, tar, tar_weight)

        # hand type loss
        assert not isinstance(self.hand_type_loss, nn.Sequential)
        out, tar, tar_weight = output[2], target[2], target_weight[2]
        assert tar.dim() == 2 and tar_weight.dim() in [1, 2]
        losses['hand_type_loss'] = self.hand_type_loss(out, tar, tar_weight)

        return losses

    def get_accuracy(self, output, target, target_weight):
        """Calculate accuracy for hand type.

        Args:
            output (list[Tensor]): a list of outputs from multiple heads.
            target (list[Tensor]): a list of targets for multiple heads.
            target_weight (list[Tensor]): a list of targets weight for
                multiple heads.
        """
        accuracy = dict()
        avg_acc = multilabel_classification_accuracy(
            output[2].detach().cpu().numpy(),
            target[2].detach().cpu().numpy(),
            target_weight[2].detach().cpu().numpy(),
        )
        accuracy['acc_classification'] = float(avg_acc)
        return accuracy

    def forward(self, x):
        """Forward function."""
        outputs = []
        outputs.append(
            torch.cat([self.right_hand_head(x),
                       self.left_hand_head(x)], dim=1))
        x = self.neck(x)
        outputs.append(self.root_head(x))
        outputs.append(self.hand_type_head(x))
        return outputs

    def inference_model(self, x, flip_pairs=None):
        """Inference function.

        Returns:
            output (list[np.ndarray]): list of output hand keypoint
            heatmaps, relative root depth and hand type.

        Args:
            x (torch.Tensor[N,K,H,W]): Input features.
            flip_pairs (None | list[tuple()):
                Pairs of keypoints which are mirrored.
        """

        output = self.forward(x)

        if flip_pairs is not None:
            # flip 3D heatmap
            heatmap_3d = output[0]
            N, K, D, H, W = heatmap_3d.shape
            # reshape 3D heatmap to 2D heatmap
            heatmap_3d = heatmap_3d.reshape(N, K * D, H, W)
            # 2D heatmap flip
            heatmap_3d_flipped_back = flip_back(
                heatmap_3d.detach().cpu().numpy(),
                flip_pairs,
                target_type=self.target_type)
            # reshape back to 3D heatmap
            heatmap_3d_flipped_back = heatmap_3d_flipped_back.reshape(
                N, K, D, H, W)
            # feature is not aligned, shift flipped heatmap for higher accuracy
            if self.test_cfg.get('shift_heatmap', False):
                heatmap_3d_flipped_back[...,
                                        1:] = heatmap_3d_flipped_back[..., :-1]
            output[0] = heatmap_3d_flipped_back

            # flip relative hand root depth
            output[1] = -output[1].detach().cpu().numpy()

            # flip hand type
            hand_type = output[2].detach().cpu().numpy()
            hand_type_flipped_back = hand_type.copy()
            hand_type_flipped_back[:, 0] = hand_type[:, 1]
            hand_type_flipped_back[:, 1] = hand_type[:, 0]
            output[2] = hand_type_flipped_back
        else:
            output = [out.detach().cpu().numpy() for out in output]

        return output

    def decode(self, img_metas, output, **kwargs):
        """Decode hand keypoint, relative root depth and hand type.

        Args:
            img_metas (list(dict)): Information about data augmentation
                By default this includes:

                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
                - "heatmap3d_depth_bound": depth bound of hand keypoint
                    3D heatmap
                - "root_depth_bound": depth bound of relative root depth
                    1D heatmap
            output (list[np.ndarray]): model predicted 3D heatmaps, relative
                root depth and hand type.
        """

        batch_size = len(img_metas)
        result = {}

        heatmap3d_depth_bound = np.ones(batch_size, dtype=np.float32)
        root_depth_bound = np.ones(batch_size, dtype=np.float32)
        center = np.zeros((batch_size, 2), dtype=np.float32)
        scale = np.zeros((batch_size, 2), dtype=np.float32)
        image_paths = []
        score = np.ones(batch_size, dtype=np.float32)
        if 'bbox_id' in img_metas[0]:
            bbox_ids = []
        else:
            bbox_ids = None

        for i in range(batch_size):
            heatmap3d_depth_bound[i] = img_metas[i]['heatmap3d_depth_bound']
            root_depth_bound[i] = img_metas[i]['root_depth_bound']
            center[i, :] = img_metas[i]['center']
            scale[i, :] = img_metas[i]['scale']
            image_paths.append(img_metas[i]['image_file'])

            if 'bbox_score' in img_metas[i]:
                score[i] = np.array(img_metas[i]['bbox_score']).reshape(-1)
            if bbox_ids is not None:
                bbox_ids.append(img_metas[i]['bbox_id'])

        all_boxes = np.zeros((batch_size, 6), dtype=np.float32)
        all_boxes[:, 0:2] = center[:, 0:2]
        all_boxes[:, 2:4] = scale[:, 0:2]
        # scale is defined as: bbox_size / 200.0, so we
        # need multiply 200.0 to get bbox size
        all_boxes[:, 4] = np.prod(scale * 200.0, axis=1)
        all_boxes[:, 5] = score
        result['boxes'] = all_boxes
        result['image_paths'] = image_paths
        result['bbox_ids'] = bbox_ids

        # decode 3D heatmaps of hand keypoints
        heatmap3d = output[0]
        preds, maxvals = keypoints_from_heatmaps3d(heatmap3d, center, scale)
        keypoints_3d = np.zeros((batch_size, preds.shape[1], 4),
                                dtype=np.float32)
        keypoints_3d[:, :, 0:3] = preds[:, :, 0:3]
        keypoints_3d[:, :, 3:4] = maxvals
        # transform keypoint depth to camera space
        keypoints_3d[:, :, 2] = \
            (keypoints_3d[:, :, 2] / self.right_hand_head.depth_size - 0.5) \
            * heatmap3d_depth_bound[:, np.newaxis]

        result['preds'] = keypoints_3d

        # decode relative hand root depth
        # transform relative root depth to camera space
        result['rel_root_depth'] = (output[1] / self.root_head.heatmap_size -
                                    0.5) * root_depth_bound

        # decode hand type
        result['hand_type'] = output[2] > 0.5
        return result
