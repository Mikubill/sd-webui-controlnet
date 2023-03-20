# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch.nn as nn

from .. import builder
from ..builder import POSENETS
from .base import BasePose

try:
    from annotator.mmpkg.mmcv.runner import auto_fp16
except ImportError:
    warnings.warn('auto_fp16 from annotator.mmpkg.mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from annotator.mmpkg.mmpose.core import auto_fp16


@POSENETS.register_module()
class GestureRecognizer(BasePose):
    """Hand gesture recognizer.

    Args:
        backbone (dict): Backbone modules to extract feature.
        neck (dict): Neck Modules to process feature.
        cls_head (dict): Classification head to process feature.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        modality (str or list or tuple): Data modality. Default: None.
        pretrained (str): Path to the pretrained models.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 cls_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 modality='rgb',
                 pretrained=None):
        super().__init__()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if isinstance(modality, (tuple, list)):
            self.modality = modality
        else:
            self.modality = (modality, )
            backbone = {modality: backbone}
            pretrained = {modality: pretrained}

        # build backbone
        self.backbone = nn.Module()
        for modal in self.modality:
            setattr(self.backbone, modal,
                    builder.build_backbone(backbone[modal]))

        # build neck
        if neck is not None:
            self.neck = builder.build_neck(neck)

        # build head
        cls_head['train_cfg'] = train_cfg
        cls_head['test_cfg'] = test_cfg
        cls_head['modality'] = self.modality
        self.cls_head = builder.build_head(cls_head)

        self.pretrained = dict() if pretrained is None else pretrained
        self.init_weights()

    def init_weights(self, pretrained=None):
        """Weight initialization for model."""
        if pretrained is not None:
            self.pretrained = pretrained
        for modal in self.modality:
            getattr(self.backbone,
                    modal).init_weights(self.pretrained.get(modal, None))
        if hasattr(self, 'neck'):
            self.neck.init_weights()
        if hasattr(self, 'cls_head'):
            self.cls_head.init_weights()

    @auto_fp16(apply_to=('video', ))
    def forward(self,
                video,
                label=None,
                img_metas=None,
                return_loss=True,
                **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.

            Note:
                - batch_size: N
                - num_vid_channel: C (Default: 3)
                - video height: vidH
                - video width: vidW
                - video length: vidL

            Args:
                video (list[torch.Tensor[NxCxvidLxvidHxvidW]]): Input videos.
                label (torch.Tensor[N]): Category label of videos.
                img_metas (list(dict)): Information about data.
                    By default this includes:
                    - "fps: video frame rate
                    - "modality": modality of input videos
                return_loss (bool): Option to `return loss`. `return loss=True`
                    for training, `return loss=False` for validation & test.

            Returns:
                dict|tuple: if `return loss` is true, then return losses. \
                    Otherwise, return predicted gestures for clips with \
                    a certain length. \
        .
        """
        if not isinstance(img_metas, (tuple, list)):
            img_metas = [img_metas.data]
        if return_loss:
            return self.forward_train(video, label, img_metas[0], **kwargs)
        return self.forward_test(video, label, img_metas[0], **kwargs)

    def _feed_forward(self, video, img_metas):
        """Feed videos into network to compute feature maps and logits.

        Note:
            - batch_size: N
            - num_vid_channel: C (Default: 3)
            - video height: vidH
            - video width: vidW
            - video length: vidL

        Args:
            video (list[torch.Tensor[NxCxvidLxvidHxvidW]]): Input videos.
            img_metas (list(dict)): Information about data.
                By default this includes:
                - "fps: video frame rate
                - "modality": modality of input videos

        Returns:
            tuple[Tensor, Tensor]: output logit and feature map.
        """
        fmaps = []
        for i, modal in enumerate(img_metas['modality']):
            fmaps.append(getattr(self.backbone, modal)(video[i]))

        if hasattr(self, 'neck'):
            fmaps = [self.neck(fmap) for fmap in fmaps]

        if hasattr(self, 'cls_head'):
            logits = self.cls_head(fmaps, img_metas)
        else:
            return None, fmaps

        return logits, fmaps

    def forward_train(self, video, label, img_metas, **kwargs):
        """Defines the computation performed at every call when training."""
        logits, fmaps = self._feed_forward(video, img_metas)

        # if return loss
        losses = dict()
        if hasattr(self, 'cls_head'):
            cls_losses = self.cls_head.get_loss(logits, label, fmaps=fmaps)
            losses.update(cls_losses)
            cls_accuracy = self.cls_head.get_accuracy(logits, label, img_metas)
            losses.update(cls_accuracy)

        return losses

    def forward_test(self, video, label, img_metas, **kwargs):
        """Defines the computation performed at every call when testing."""
        results = dict(logits=dict())
        logits, _ = self._feed_forward(video, img_metas)
        for i, modal in enumerate(img_metas['modality']):
            results['logits'][modal] = logits[i]
        results['label'] = label
        return results

    def set_train_epoch(self, epoch: int):
        """set the training epoch of heads to support customized behaviour."""
        if hasattr(self, 'cls_head'):
            self.cls_head.set_train_epoch(epoch)

    def forward_dummy(self, video):
        raise NotImplementedError

    def show_result(self, video, result, **kwargs):
        raise NotImplementedError
