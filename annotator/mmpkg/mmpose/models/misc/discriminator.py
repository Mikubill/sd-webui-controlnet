# ------------------------------------------------------------------------------
# Adapted from https://github.com/akanazawa/hmr
# Original licence: Copyright (c) 2018 akanazawa, under the MIT License.
# ------------------------------------------------------------------------------

from abc import abstractmethod

import torch
import torch.nn as nn
from annotator.mmpkg.mmcv.cnn import normal_init, xavier_init

from annotator.mmpkg.mmpose.models.utils.geometry import batch_rodrigues


class BaseDiscriminator(nn.Module):
    """Base linear module for SMPL parameter discriminator.

    Args:
        fc_layers (Tuple): Tuple of neuron count,
            such as (9, 32, 32, 1)
        use_dropout (Tuple): Tuple of bool define use dropout or not
            for each layer, such as (True, True, False)
        drop_prob (Tuple): Tuple of float defined the drop prob,
            such as (0.5, 0.5, 0)
        use_activation(Tuple): Tuple of bool define use active function
            or not, such as (True, True, False)
    """

    def __init__(self, fc_layers, use_dropout, drop_prob, use_activation):
        super().__init__()
        self.fc_layers = fc_layers
        self.use_dropout = use_dropout
        self.drop_prob = drop_prob
        self.use_activation = use_activation
        self._check()
        self.create_layers()

    def _check(self):
        """Check input to avoid ValueError."""
        if not isinstance(self.fc_layers, tuple):
            raise TypeError(f'fc_layers require tuple, '
                            f'get {type(self.fc_layers)}')

        if not isinstance(self.use_dropout, tuple):
            raise TypeError(f'use_dropout require tuple, '
                            f'get {type(self.use_dropout)}')

        if not isinstance(self.drop_prob, tuple):
            raise TypeError(f'drop_prob require tuple, '
                            f'get {type(self.drop_prob)}')

        if not isinstance(self.use_activation, tuple):
            raise TypeError(f'use_activation require tuple, '
                            f'get {type(self.use_activation)}')

        l_fc_layer = len(self.fc_layers)
        l_use_drop = len(self.use_dropout)
        l_drop_prob = len(self.drop_prob)
        l_use_activation = len(self.use_activation)

        pass_check = (
            l_fc_layer >= 2 and l_use_drop < l_fc_layer
            and l_drop_prob < l_fc_layer and l_use_activation < l_fc_layer
            and l_drop_prob == l_use_drop)

        if not pass_check:
            msg = 'Wrong BaseDiscriminator parameters!'
            raise ValueError(msg)

    def create_layers(self):
        """Create layers."""
        l_fc_layer = len(self.fc_layers)
        l_use_drop = len(self.use_dropout)
        l_use_activation = len(self.use_activation)

        self.fc_blocks = nn.Sequential()

        for i in range(l_fc_layer - 1):
            self.fc_blocks.add_module(
                name=f'regressor_fc_{i}',
                module=nn.Linear(
                    in_features=self.fc_layers[i],
                    out_features=self.fc_layers[i + 1]))

            if i < l_use_activation and self.use_activation[i]:
                self.fc_blocks.add_module(
                    name=f'regressor_af_{i}', module=nn.ReLU())

            if i < l_use_drop and self.use_dropout[i]:
                self.fc_blocks.add_module(
                    name=f'regressor_fc_dropout_{i}',
                    module=nn.Dropout(p=self.drop_prob[i]))

    @abstractmethod
    def forward(self, inputs):
        """Forward function."""
        msg = 'the base class [BaseDiscriminator] is not callable!'
        raise NotImplementedError(msg)

    def init_weights(self):
        """Initialize model weights."""
        for m in self.fc_blocks.named_modules():
            if isinstance(m, nn.Linear):
                xavier_init(m, gain=0.01)


class ShapeDiscriminator(BaseDiscriminator):
    """Discriminator for SMPL shape parameters, the inputs is (batch_size x 10)

    Args:
        fc_layers (Tuple): Tuple of neuron count, such as (10, 5, 1)
        use_dropout (Tuple): Tuple of bool define use dropout or
            not for each layer, such as (True, True, False)
        drop_prob (Tuple): Tuple of float defined the drop prob,
            such as (0.5, 0)
        use_activation(Tuple): Tuple of bool define use active
            function or not, such as (True, False)
    """

    def __init__(self, fc_layers, use_dropout, drop_prob, use_activation):
        if fc_layers[-1] != 1:
            msg = f'the neuron count of the last layer ' \
                  f'must be 1, but got {fc_layers[-1]}'
            raise ValueError(msg)

        super().__init__(fc_layers, use_dropout, drop_prob, use_activation)

    def forward(self, inputs):
        """Forward function."""
        return self.fc_blocks(inputs)


class PoseDiscriminator(nn.Module):
    """Discriminator for SMPL pose parameters of each joint. It is composed of
    discriminators for each joints. The inputs is (batch_size x joint_count x
    9)

    Args:
        channels (Tuple): Tuple of channel number,
            such as (9, 32, 32, 1)
        joint_count (int): Joint number, such as 23
    """

    def __init__(self, channels, joint_count):
        super().__init__()
        if channels[-1] != 1:
            msg = f'the neuron count of the last layer ' \
                  f'must be 1, but got {channels[-1]}'
            raise ValueError(msg)
        self.joint_count = joint_count

        self.conv_blocks = nn.Sequential()
        len_channels = len(channels)
        for idx in range(len_channels - 2):
            self.conv_blocks.add_module(
                name=f'conv_{idx}',
                module=nn.Conv2d(
                    in_channels=channels[idx],
                    out_channels=channels[idx + 1],
                    kernel_size=1,
                    stride=1))

        self.fc_layer = nn.ModuleList()
        for idx in range(joint_count):
            self.fc_layer.append(
                nn.Linear(
                    in_features=channels[len_channels - 2], out_features=1))

    def forward(self, inputs):
        """Forward function.

        The input is (batch_size x joint_count x 9).
        """
        # shape: batch_size x 9 x 1 x joint_count
        inputs = inputs.transpose(1, 2).unsqueeze(2).contiguous()
        # shape: batch_size x c x 1 x joint_count
        internal_outputs = self.conv_blocks(inputs)
        outputs = []
        for idx in range(self.joint_count):
            outputs.append(self.fc_layer[idx](internal_outputs[:, :, 0, idx]))

        return torch.cat(outputs, 1), internal_outputs

    def init_weights(self):
        """Initialize model weights."""
        for m in self.conv_blocks:
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
        for m in self.fc_layer.named_modules():
            if isinstance(m, nn.Linear):
                xavier_init(m, gain=0.01)


class FullPoseDiscriminator(BaseDiscriminator):
    """Discriminator for SMPL pose parameters of all joints.

    Args:
        fc_layers (Tuple): Tuple of neuron count,
            such as (736, 1024, 1024, 1)
        use_dropout (Tuple): Tuple of bool define use dropout or not
            for each layer, such as (True, True, False)
        drop_prob (Tuple): Tuple of float defined the drop prob,
            such as (0.5, 0.5, 0)
        use_activation(Tuple): Tuple of bool define use active
            function or not, such as (True, True, False)
    """

    def __init__(self, fc_layers, use_dropout, drop_prob, use_activation):
        if fc_layers[-1] != 1:
            msg = f'the neuron count of the last layer must be 1,' \
                  f' but got {fc_layers[-1]}'
            raise ValueError(msg)

        super().__init__(fc_layers, use_dropout, drop_prob, use_activation)

    def forward(self, inputs):
        """Forward function."""
        return self.fc_blocks(inputs)


class SMPLDiscriminator(nn.Module):
    """Discriminator for SMPL pose and shape parameters. It is composed of a
    discriminator for SMPL shape parameters, a discriminator for SMPL pose
    parameters of all joints  and a discriminator for SMPL pose parameters of
    each joint.

    Args:
        beta_channel (tuple of int): Tuple of neuron count of the
            discriminator of shape parameters. Defaults to (10, 5, 1)
        per_joint_channel (tuple of int): Tuple of neuron count of the
            discriminator of each joint. Defaults to (9, 32, 32, 1)
        full_pose_channel (tuple of int): Tuple of neuron count of the
            discriminator of full pose. Defaults to (23*32, 1024, 1024, 1)
    """

    def __init__(self,
                 beta_channel=(10, 5, 1),
                 per_joint_channel=(9, 32, 32, 1),
                 full_pose_channel=(23 * 32, 1024, 1024, 1)):
        super().__init__()
        self.joint_count = 23
        # The count of SMPL shape parameter is 10.
        assert beta_channel[0] == 10
        # Use 3 x 3 rotation matrix as the pose parameters
        # of each joint, so the input channel is 9.
        assert per_joint_channel[0] == 9
        assert self.joint_count * per_joint_channel[-2] \
            == full_pose_channel[0]

        self.beta_channel = beta_channel
        self.per_joint_channel = per_joint_channel
        self.full_pose_channel = full_pose_channel
        self._create_sub_modules()

    def _create_sub_modules(self):
        """Create sub discriminators."""

        # create theta discriminator for each joint
        self.pose_discriminator = PoseDiscriminator(self.per_joint_channel,
                                                    self.joint_count)

        # create full pose discriminator for total joints
        fc_layers = self.full_pose_channel
        use_dropout = tuple([False] * (len(fc_layers) - 1))
        drop_prob = tuple([0.5] * (len(fc_layers) - 1))
        use_activation = tuple([True] * (len(fc_layers) - 2) + [False])

        self.full_pose_discriminator = FullPoseDiscriminator(
            fc_layers, use_dropout, drop_prob, use_activation)

        # create shape discriminator for betas
        fc_layers = self.beta_channel
        use_dropout = tuple([False] * (len(fc_layers) - 1))
        drop_prob = tuple([0.5] * (len(fc_layers) - 1))
        use_activation = tuple([True] * (len(fc_layers) - 2) + [False])
        self.shape_discriminator = ShapeDiscriminator(fc_layers, use_dropout,
                                                      drop_prob,
                                                      use_activation)

    def forward(self, thetas):
        """Forward function."""
        _, poses, shapes = thetas

        batch_size = poses.shape[0]
        shape_disc_value = self.shape_discriminator(shapes)

        # The first rotation matrix is global rotation
        # and is NOT used in discriminator.
        if poses.dim() == 2:
            rotate_matrixs = \
                batch_rodrigues(poses.contiguous().view(-1, 3)
                                ).view(batch_size, 24, 9)[:, 1:, :]
        else:
            rotate_matrixs = poses.contiguous().view(batch_size, 24,
                                                     9)[:, 1:, :].contiguous()
        pose_disc_value, pose_inter_disc_value \
            = self.pose_discriminator(rotate_matrixs)
        full_pose_disc_value = self.full_pose_discriminator(
            pose_inter_disc_value.contiguous().view(batch_size, -1))
        return torch.cat(
            (pose_disc_value, full_pose_disc_value, shape_disc_value), 1)

    def init_weights(self):
        """Initialize model weights."""
        self.full_pose_discriminator.init_weights()
        self.pose_discriminator.init_weights()
        self.shape_discriminator.init_weights()
