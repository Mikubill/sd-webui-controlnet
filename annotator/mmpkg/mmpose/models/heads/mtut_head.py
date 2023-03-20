# Copyright (c) OpenMMLab. All rights reserved.
import itertools

import torch
import torch.nn as nn
from annotator.mmpkg.mmcv.cnn import xavier_init

from ..builder import HEADS


@HEADS.register_module()
class MultiModalSSAHead(nn.Module):
    """Sparial-temporal Semantic Alignment Head proposed in "Improving the
    performance of unimodal dynamic hand-gesture recognition with multimodal
    training",

    Please refer to the `paper <https://arxiv.org/abs/1812.06145>`__ for
    details.

    Args:
        num_classes (int): number of classes.
        modality (list[str]): modalities of input videos for backbone.
        in_channels (int): number of channels of feature maps. Default: 1024
        avg_pool_kernel (tuple[int]): kernel size of pooling layer.
            Default: (1, 7, 7)
        dropout_prob (float): probablity to use dropout on input feature map.
            Default: 0
        train_cfg (dict): training config.
        test_cfg (dict): testing config.
    """

    def __init__(self,
                 num_classes,
                 modality,
                 in_channels=1024,
                 avg_pool_kernel=(1, 7, 7),
                 dropout_prob=0.0,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):
        super().__init__()

        self.modality = modality
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # build sub modules
        self.avg_pool = nn.AvgPool3d(avg_pool_kernel, (1, 1, 1))
        self.dropout = nn.Dropout(dropout_prob)
        self.output_conv = nn.Module()
        for modal in self.modality:
            conv3d = nn.Conv3d(in_channels, num_classes, (1, 1, 1))
            setattr(self.output_conv, modal, conv3d)
        self.loss = nn.CrossEntropyLoss(reduction='none')

        # parameters for ssa loss
        self.beta = self.train_cfg.get('beta', 2.0)
        self.lambda_ = self.train_cfg.get('lambda_', 5e-3)
        self.start_epoch = self.train_cfg.get('ssa_start_epoch', 1e6)
        self._train_epoch = 0

    def init_weights(self):
        """Initialize model weights."""
        for m in self.output_conv.modules():
            if isinstance(m, nn.Conv3d):
                xavier_init(m)

    def set_train_epoch(self, epoch: int):
        """set the epoch to control the activation of SSA loss."""
        self._train_epoch = epoch

    def forward(self, x, img_metas):
        """Forward function."""
        logits = []
        for i, modal in enumerate(img_metas['modality']):
            out = self.avg_pool(x[i])
            out = self.dropout(out)
            out = getattr(self.output_conv, modal)(out)
            out = out.mean(3).mean(3)
            logits.append(out)
        return logits

    @staticmethod
    def _compute_corr(fmap):
        """compute the self-correlation matrix of feature map."""
        fmap = fmap.view(fmap.size(0), fmap.size(1), -1)
        fmap = nn.functional.normalize(fmap, dim=2, eps=1e-8)
        corr = torch.bmm(fmap.permute(0, 2, 1), fmap)
        return corr.view(corr.size(0), -1)

    def get_loss(self, logits, label, fmaps=None):
        """Compute the Cross Entropy loss and SSA loss.

        Note:
            - batch_size: N
            - number of classes: nC
            - feature map channel: C
            - feature map height: H
            - feature map width: W
            - feature map length: L
            - logit length: Lg

        Args:
            logits (list[NxnCxLg]): predicted logits for each modality.
            label (list(dict)): Category label.
            fmaps (list[torch.Tensor[NxCxLxHxW]]): feature maps for each
                modality.

        Returns:
            dict[str, torch.tensor]: computed losses.
        """
        losses = {}
        ce_loss = [self.loss(logit.mean(dim=2), label) for logit in logits]

        if self._train_epoch >= self.start_epoch:
            ssa_loss = []
            corrs = [self._compute_corr(fmap) for fmap in fmaps]
            for idx1, idx2 in itertools.combinations(range(len(fmaps)), 2):
                for i, j in ((idx1, idx2), (idx2, idx1)):
                    rho = (ce_loss[i] - ce_loss[j]).clamp(min=0)
                    rho = (torch.exp(self.beta * rho) - 1).detach()
                    ssa = corrs[i] - corrs[j].detach()
                    ssa = rho * ssa.pow(2).mean(dim=1).pow(0.5)
                    ssa_loss.append((ssa.mean() * self.lambda_).clamp(max=10))
            losses['ssa_loss'] = sum(ssa_loss)
        ce_loss = [loss.mean() for loss in ce_loss]
        losses['ce_loss'] = sum(ce_loss)

        return losses

    def get_accuracy(self, logits, label, img_metas):
        """Compute the accuracy of predicted gesture.

        Note:
            - batch_size: N
            - number of classes: nC
            - logit length: L

        Args:
            logits (list[NxnCxL]): predicted logits for each modality.
            label (list(dict)): Category label.
            img_metas (list(dict)): Information about data.
                By default this includes:
                - "fps: video frame rate
                - "modality": modality of input videos

        Returns:
            dict[str, torch.tensor]: computed accuracy for each modality.
        """
        results = {}
        for i, modal in enumerate(img_metas['modality']):
            logit = logits[i].mean(dim=2)
            acc = (logit.argmax(dim=1) == label).float().mean()
            results[f'acc_{modal}'] = acc.item()
        return results
