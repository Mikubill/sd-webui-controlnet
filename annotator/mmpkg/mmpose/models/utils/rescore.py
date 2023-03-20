# Copyright (c) OpenMMLab. All rights reserved.
# Code is modified from `HRNet/DEKR <https://github.com/HRNet/DEKR>`.

import torch
from annotator.mmpkg.mmcv.runner import load_checkpoint


class DekrRescoreNet(torch.nn.Module):
    """Rescore net used to predict the OKS score of predicted pose. We use the
    off-the-shelf rescore net pretrained by authors of DEKR.

    Args:
        in_channels (int): input channels
        norm_indexes (Tuple(int)): indexes of torso in skeleton.
        pretrained (str): url or path of pretrained rescore net.
    """

    def __init__(
        self,
        in_channels,
        norm_indexes,
        pretrained=None,
    ):
        super(DekrRescoreNet, self).__init__()

        self.pretrained = pretrained
        self.norm_indexes = norm_indexes

        hidden = 256

        self.l1 = torch.nn.Linear(in_channels, hidden, bias=True)
        self.l2 = torch.nn.Linear(hidden, hidden, bias=True)
        self.l3 = torch.nn.Linear(hidden, 1, bias=True)
        self.relu = torch.nn.ReLU()

    def make_feature(self, poses, skeleton):
        """Combine original scores, joint distance and relative distance to
        make feature.

        Args:
            poses (np.ndarray): predicetd poses
            skeleton (list(list(int))): joint links

        Returns:
            torch.Tensor: feature for each instance
        """
        poses = torch.tensor(poses)
        joint_1, joint_2 = zip(*skeleton)
        num_link = len(skeleton)

        joint_relate = (poses[:, joint_1] - poses[:, joint_2])[:, :, :2]
        joint_length = joint_relate.norm(dim=2)

        # To use the torso distance to normalize
        normalize = (joint_length[:, self.norm_indexes[0]] +
                     joint_length[:, self.norm_indexes[1]]) / 2
        normalize = normalize.unsqueeze(1).expand(normalize.size(0), num_link)
        normalize = normalize.clamp(min=1).contiguous()

        joint_length = joint_length / normalize[:, :]
        joint_relate = joint_relate / normalize.unsqueeze(-1)
        joint_relate = joint_relate.flatten(1)

        feature = torch.cat((joint_relate, joint_length, poses[..., 2]),
                            dim=1).float()
        return feature

    def forward(self, poses, skeleton):
        feature = self.make_feature(poses, skeleton).to(self.l1.weight.device)
        x = self.relu(self.l1(feature))
        x = self.relu(self.l2(x))
        x = self.l3(x)
        return x.squeeze(1)

    def init_weight(self):
        if self.pretrained is not None:
            load_checkpoint(self, self.pretrained, map_location='cpu')
