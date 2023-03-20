# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from ..builder import LOSSES
from ..utils.geometry import batch_rodrigues


def perspective_projection(points, rotation, translation, focal_length,
                           camera_center):
    """This function computes the perspective projection of a set of 3D points.

    Note:
        - batch size: B
        - point number: N

    Args:
        points (Tensor([B, N, 3])): A set of 3D points
        rotation (Tensor([B, 3, 3])): Camera rotation matrix
        translation (Tensor([B, 3])): Camera translation
        focal_length (Tensor([B,])): Focal length
        camera_center (Tensor([B, 2])): Camera center

    Returns:
        projected_points (Tensor([B, N, 2])): Projected 2D
            points in image space.
    """

    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:, 0, 0] = focal_length
    K[:, 1, 1] = focal_length
    K[:, 2, 2] = 1.
    K[:, :-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:, :, -1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)
    projected_points = projected_points[:, :, :-1]
    return projected_points


@LOSSES.register_module()
class MeshLoss(nn.Module):
    """Mix loss for 3D human mesh. It is composed of loss on 2D joints, 3D
    joints, mesh vertices and smpl parameters (if any).

    Args:
        joints_2d_loss_weight (float): Weight for loss on 2D joints.
        joints_3d_loss_weight (float): Weight for loss on 3D joints.
        vertex_loss_weight (float): Weight for loss on 3D verteices.
        smpl_pose_loss_weight (float): Weight for loss on SMPL
            pose parameters.
        smpl_beta_loss_weight (float): Weight for loss on SMPL
            shape parameters.
        img_res (int): Input image resolution.
        focal_length (float): Focal length of camera model. Default=5000.
    """

    def __init__(self,
                 joints_2d_loss_weight,
                 joints_3d_loss_weight,
                 vertex_loss_weight,
                 smpl_pose_loss_weight,
                 smpl_beta_loss_weight,
                 img_res,
                 focal_length=5000):

        super().__init__()
        # Per-vertex loss on the mesh
        self.criterion_vertex = nn.L1Loss(reduction='none')

        # Joints (2D and 3D) loss
        self.criterion_joints_2d = nn.SmoothL1Loss(reduction='none')
        self.criterion_joints_3d = nn.SmoothL1Loss(reduction='none')

        # Loss for SMPL parameter regression
        self.criterion_regr = nn.MSELoss(reduction='none')

        self.joints_2d_loss_weight = joints_2d_loss_weight
        self.joints_3d_loss_weight = joints_3d_loss_weight
        self.vertex_loss_weight = vertex_loss_weight
        self.smpl_pose_loss_weight = smpl_pose_loss_weight
        self.smpl_beta_loss_weight = smpl_beta_loss_weight
        self.focal_length = focal_length
        self.img_res = img_res

    def joints_2d_loss(self, pred_joints_2d, gt_joints_2d, joints_2d_visible):
        """Compute 2D reprojection loss on the joints.

        The loss is weighted by joints_2d_visible.
        """
        conf = joints_2d_visible.float()
        loss = (conf *
                self.criterion_joints_2d(pred_joints_2d, gt_joints_2d)).mean()
        return loss

    def joints_3d_loss(self, pred_joints_3d, gt_joints_3d, joints_3d_visible):
        """Compute 3D joints loss for the examples that 3D joint annotations
        are available.

        The loss is weighted by joints_3d_visible.
        """
        conf = joints_3d_visible.float()
        if len(gt_joints_3d) > 0:
            gt_pelvis = (gt_joints_3d[:, 2, :] + gt_joints_3d[:, 3, :]) / 2
            gt_joints_3d = gt_joints_3d - gt_pelvis[:, None, :]
            pred_pelvis = (pred_joints_3d[:, 2, :] +
                           pred_joints_3d[:, 3, :]) / 2
            pred_joints_3d = pred_joints_3d - pred_pelvis[:, None, :]
            return (
                conf *
                self.criterion_joints_3d(pred_joints_3d, gt_joints_3d)).mean()
        return pred_joints_3d.sum() * 0

    def vertex_loss(self, pred_vertices, gt_vertices, has_smpl):
        """Compute 3D vertex loss for the examples that 3D human mesh
        annotations are available.

        The loss is weighted by the has_smpl.
        """
        conf = has_smpl.float()
        loss_vertex = self.criterion_vertex(pred_vertices, gt_vertices)
        loss_vertex = (conf[:, None, None] * loss_vertex).mean()
        return loss_vertex

    def smpl_losses(self, pred_rotmat, pred_betas, gt_pose, gt_betas,
                    has_smpl):
        """Compute SMPL parameters loss for the examples that SMPL parameter
        annotations are available.

        The loss is weighted by has_smpl.
        """
        conf = has_smpl.float()
        gt_rotmat = batch_rodrigues(gt_pose.view(-1, 3)).view(-1, 24, 3, 3)
        loss_regr_pose = self.criterion_regr(pred_rotmat, gt_rotmat)
        loss_regr_betas = self.criterion_regr(pred_betas, gt_betas)
        loss_regr_pose = (conf[:, None, None, None] * loss_regr_pose).mean()
        loss_regr_betas = (conf[:, None] * loss_regr_betas).mean()
        return loss_regr_pose, loss_regr_betas

    def project_points(self, points_3d, camera):
        """Perform orthographic projection of 3D points using the camera
        parameters, return projected 2D points in image plane.

        Note:
            - batch size: B
            - point number: N

        Args:
            points_3d (Tensor([B, N, 3])): 3D points.
            camera (Tensor([B, 3])): camera parameters with the
                3 channel as (scale, translation_x, translation_y)

        Returns:
            Tensor([B, N, 2]): projected 2D points \
                in image space.
        """
        batch_size = points_3d.shape[0]
        device = points_3d.device
        cam_t = torch.stack([
            camera[:, 1], camera[:, 2], 2 * self.focal_length /
            (self.img_res * camera[:, 0] + 1e-9)
        ],
                            dim=-1)
        camera_center = camera.new_zeros([batch_size, 2])
        rot_t = torch.eye(
            3, device=device,
            dtype=points_3d.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        joints_2d = perspective_projection(
            points_3d,
            rotation=rot_t,
            translation=cam_t,
            focal_length=self.focal_length,
            camera_center=camera_center)
        return joints_2d

    def forward(self, output, target):
        """Forward function.

        Args:
            output (dict): dict of network predicted results.
                Keys: 'vertices', 'joints_3d', 'camera',
                'pose'(optional), 'beta'(optional)
            target (dict): dict of ground-truth labels.
                Keys: 'vertices', 'joints_3d', 'joints_3d_visible',
                'joints_2d', 'joints_2d_visible', 'pose', 'beta',
                'has_smpl'

        Returns:
            dict: dict of losses.
        """
        losses = {}

        # Per-vertex loss for the shape
        pred_vertices = output['vertices']

        gt_vertices = target['vertices']
        has_smpl = target['has_smpl']
        loss_vertex = self.vertex_loss(pred_vertices, gt_vertices, has_smpl)
        losses['vertex_loss'] = loss_vertex * self.vertex_loss_weight

        # Compute loss on SMPL parameters, if available
        if 'pose' in output.keys() and 'beta' in output.keys():
            pred_rotmat = output['pose']
            pred_betas = output['beta']
            gt_pose = target['pose']
            gt_betas = target['beta']
            loss_regr_pose, loss_regr_betas = self.smpl_losses(
                pred_rotmat, pred_betas, gt_pose, gt_betas, has_smpl)
            losses['smpl_pose_loss'] = \
                loss_regr_pose * self.smpl_pose_loss_weight
            losses['smpl_beta_loss'] = \
                loss_regr_betas * self.smpl_beta_loss_weight

        # Compute 3D joints loss
        pred_joints_3d = output['joints_3d']
        gt_joints_3d = target['joints_3d']
        joints_3d_visible = target['joints_3d_visible']
        loss_joints_3d = self.joints_3d_loss(pred_joints_3d, gt_joints_3d,
                                             joints_3d_visible)
        losses['joints_3d_loss'] = loss_joints_3d * self.joints_3d_loss_weight

        # Compute 2D reprojection loss for the 2D joints
        pred_camera = output['camera']
        gt_joints_2d = target['joints_2d']
        joints_2d_visible = target['joints_2d_visible']
        pred_joints_2d = self.project_points(pred_joints_3d, pred_camera)

        # Normalize keypoints to [-1,1]
        # The coordinate origin of pred_joints_2d is
        #  the center of the input image.
        pred_joints_2d = 2 * pred_joints_2d / (self.img_res - 1)
        # The coordinate origin of gt_joints_2d is
        # the top left corner of the input image.
        gt_joints_2d = 2 * gt_joints_2d / (self.img_res - 1) - 1
        loss_joints_2d = self.joints_2d_loss(pred_joints_2d, gt_joints_2d,
                                             joints_2d_visible)
        losses['joints_2d_loss'] = loss_joints_2d * self.joints_2d_loss_weight

        return losses


@LOSSES.register_module()
class GANLoss(nn.Module):
    """Define GAN loss.

    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self,
                 gan_type,
                 real_label_val=1.0,
                 fake_label_val=0.0,
                 loss_weight=1.0):
        super().__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(
                f'GAN type {self.gan_type} is not implemented.')

    @staticmethod
    def _wgan_loss(input, target):
        """wgan loss.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return -input.mean() if target else input.mean()

    def get_target_label(self, input, target_is_real):
        """Get target label.

        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, \
                otherwise, return Tensor.
        """

        if self.gan_type == 'wgan':
            return target_is_real
        target_val = (
            self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.

        Returns:
            Tensor: GAN loss value.
        """
        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == 'hinge':
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:  # other gan types
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight
