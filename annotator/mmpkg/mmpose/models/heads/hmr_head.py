# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
from annotator.mmpkg.mmcv.cnn import xavier_init

from ..builder import HEADS
from ..utils.geometry import rot6d_to_rotmat


@HEADS.register_module()
class HMRMeshHead(nn.Module):
    """SMPL parameters regressor head of simple baseline. "End-to-end Recovery
    of Human Shape and Pose", CVPR'2018.

    Args:
        in_channels (int): Number of input channels
        smpl_mean_params (str): The file name of the mean SMPL parameters
        n_iter (int): The iterations of estimating delta parameters
    """

    def __init__(self, in_channels, smpl_mean_params=None, n_iter=3):
        super().__init__()

        self.in_channels = in_channels
        self.n_iter = n_iter

        npose = 24 * 6
        nbeta = 10
        ncam = 3
        hidden_dim = 1024

        self.fc1 = nn.Linear(in_channels + npose + nbeta + ncam, hidden_dim)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(hidden_dim, npose)
        self.decshape = nn.Linear(hidden_dim, nbeta)
        self.deccam = nn.Linear(hidden_dim, ncam)

        # Load mean SMPL parameters
        if smpl_mean_params is None:
            init_pose = torch.zeros([1, npose])
            init_shape = torch.zeros([1, nbeta])
            init_cam = torch.FloatTensor([[1, 0, 0]])
        else:
            mean_params = np.load(smpl_mean_params)
            init_pose = torch.from_numpy(
                mean_params['pose'][:]).unsqueeze(0).float()
            init_shape = torch.from_numpy(
                mean_params['shape'][:]).unsqueeze(0).float()
            init_cam = torch.from_numpy(
                mean_params['cam']).unsqueeze(0).float()
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def forward(self, x):
        """Forward function.

        x is the image feature map and is expected to be in shape (batch size x
        channel number x height x width)
        """
        batch_size = x.shape[0]
        # extract the global feature vector by average along
        # spatial dimension.
        x = x.mean(dim=-1).mean(dim=-1)

        init_pose = self.init_pose.expand(batch_size, -1)
        init_shape = self.init_shape.expand(batch_size, -1)
        init_cam = self.init_cam.expand(batch_size, -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for _ in range(self.n_iter):
            xc = torch.cat([x, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)
        out = (pred_rotmat, pred_shape, pred_cam)
        return out

    def init_weights(self):
        """Initialize model weights."""
        xavier_init(self.decpose, gain=0.01)
        xavier_init(self.decshape, gain=0.01)
        xavier_init(self.deccam, gain=0.01)
