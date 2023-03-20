# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

from .. import builder
from ..builder import POSENETS


@POSENETS.register_module()
class MultiTask(nn.Module):
    """Multi-task detectors.

    Args:
        backbone (dict): Backbone modules to extract feature.
        heads (list[dict]): heads to output predictions.
        necks (list[dict] | None): necks to process feature.
        head2neck (dict{int:int}): head index to neck index.
        pretrained (str): Path to the pretrained models.
    """

    def __init__(self,
                 backbone,
                 heads,
                 necks=None,
                 head2neck=None,
                 pretrained=None):
        super().__init__()

        self.backbone = builder.build_backbone(backbone)

        if head2neck is None:
            assert necks is None
            head2neck = {}

        self.head2neck = {}
        for i in range(len(heads)):
            self.head2neck[i] = head2neck[i] if i in head2neck else -1

        self.necks = nn.ModuleList([])
        if necks is not None:
            for neck in necks:
                self.necks.append(builder.build_neck(neck))
        self.necks.append(nn.Identity())

        self.heads = nn.ModuleList([])
        assert heads is not None
        for head in heads:
            assert head is not None
            self.heads.append(builder.build_head(head))
        self.pretrained = pretrained
        self.init_weights()

    @property
    def with_necks(self):
        """Check if has keypoint_head."""
        return hasattr(self, 'necks')

    def init_weights(self, pretrained=None):
        """Weight initialization for model."""
        if pretrained is not None:
            self.pretrained = pretrained
        self.backbone.init_weights(self.pretrained)
        if self.with_necks:
            for neck in self.necks:
                if hasattr(neck, 'init_weights'):
                    neck.init_weights()

        for head in self.heads:
            if hasattr(head, 'init_weights'):
                head.init_weights()

    def forward(self,
                img,
                target=None,
                target_weight=None,
                img_metas=None,
                return_loss=True,
                **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=True`, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=False`, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.

        Note:
            - batch_size: N
            - num_keypoints: K
            - num_img_channel: C (Default: 3)
            - img height: imgH
            - img weight: imgW
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            img (torch.Tensor[N,C,imgH,imgW]): Input images.
            target (list[torch.Tensor]): Targets.
            target_weight (List[torch.Tensor]): Weights.
            img_metas (list(dict)): Information about data augmentation
                By default this includes:

                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
            return_loss (bool): Option to `return loss`. `return loss=True`
                for training, `return loss=False` for validation & test.

        Returns:
            dict|tuple: if `return loss` is true, then return losses. \
                Otherwise, return predicted poses, boxes, image paths \
                and heatmaps.
        """
        if return_loss:
            return self.forward_train(img, target, target_weight, img_metas,
                                      **kwargs)
        return self.forward_test(img, img_metas, **kwargs)

    def forward_train(self, img, target, target_weight, img_metas, **kwargs):
        """Defines the computation performed at every call when training."""
        features = self.backbone(img)
        outputs = []

        for head_id, head in enumerate(self.heads):
            neck_id = self.head2neck[head_id]
            outputs.append(head(self.necks[neck_id](features)))

        # if return loss
        losses = dict()

        for head, output, gt, gt_weight in zip(self.heads, outputs, target,
                                               target_weight):
            loss = head.get_loss(output, gt, gt_weight)
            assert len(set(losses.keys()).intersection(set(loss.keys()))) == 0
            losses.update(loss)

            if hasattr(head, 'get_accuracy'):
                acc = head.get_accuracy(output, gt, gt_weight)
                assert len(set(losses.keys()).intersection(set(
                    acc.keys()))) == 0
                losses.update(acc)

        return losses

    def forward_test(self, img, img_metas, **kwargs):
        """Defines the computation performed at every call when testing."""
        assert img.size(0) == len(img_metas)
        batch_size, _, img_height, img_width = img.shape
        if batch_size > 1:
            assert 'bbox_id' in img_metas[0]

        results = {}

        features = self.backbone(img)
        outputs = []

        for head_id, head in enumerate(self.heads):
            neck_id = self.head2neck[head_id]
            if hasattr(head, 'inference_model'):
                head_output = head.inference_model(
                    self.necks[neck_id](features), flip_pairs=None)
            else:
                head_output = head(
                    self.necks[neck_id](features)).detach().cpu().numpy()
            outputs.append(head_output)

        for head, output in zip(self.heads, outputs):
            result = head.decode(
                img_metas, output, img_size=[img_width, img_height])
            results.update(result)
        return results

    def forward_dummy(self, img):
        """Used for computing network FLOPs.

        See ``tools/get_flops.py``.

        Args:
            img (torch.Tensor): Input image.

        Returns:
            list[Tensor]: Outputs.
        """
        features = self.backbone(img)
        outputs = []
        for head_id, head in enumerate(self.heads):
            neck_id = self.head2neck[head_id]
            outputs.append(head(self.necks[neck_id](features)))
        return outputs
