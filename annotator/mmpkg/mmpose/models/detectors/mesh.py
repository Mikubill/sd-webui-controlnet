# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import annotator.mmpkg.mmcv as mmcv
import numpy as np
import torch

from annotator.mmpkg.mmpose.core.visualization.image import imshow_mesh_3d
from annotator.mmpkg.mmpose.models.misc.discriminator import SMPLDiscriminator
from .. import builder
from ..builder import POSENETS
from .base import BasePose


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


@POSENETS.register_module()
class ParametricMesh(BasePose):
    """Model-based 3D human mesh detector. Take a single color image as input
    and output 3D joints, SMPL parameters and camera parameters.

    Args:
        backbone (dict): Backbone modules to extract feature.
        mesh_head (dict): Mesh head to process feature.
        smpl (dict): Config for SMPL model.
        disc (dict): Discriminator for SMPL parameters. Default: None.
        loss_gan (dict): Config for adversarial loss. Default: None.
        loss_mesh (dict): Config for mesh loss. Default: None.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path to the pretrained models.
    """

    def __init__(self,
                 backbone,
                 mesh_head,
                 smpl,
                 disc=None,
                 loss_gan=None,
                 loss_mesh=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__()

        self.backbone = builder.build_backbone(backbone)
        self.mesh_head = builder.build_head(mesh_head)
        self.generator = torch.nn.Sequential(self.backbone, self.mesh_head)

        self.smpl = builder.build_mesh_model(smpl)

        self.with_gan = disc is not None and loss_gan is not None
        if self.with_gan:
            self.discriminator = SMPLDiscriminator(**disc)
            self.loss_gan = builder.build_loss(loss_gan)
        self.disc_step_count = 0

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.loss_mesh = builder.build_loss(loss_mesh)
        self.pretrained = pretrained
        self.init_weights()

    def init_weights(self, pretrained=None):
        """Weight initialization for model."""
        if pretrained is not None:
            self.pretrained = pretrained
        self.backbone.init_weights(self.pretrained)
        self.mesh_head.init_weights()
        if self.with_gan:
            self.discriminator.init_weights()

    def train_step(self, data_batch, optimizer, **kwargs):
        """Train step function.

        In this function, the detector will finish the train step following
        the pipeline:

            1. get fake and real SMPL parameters
            2. optimize discriminator (if have)
            3. optimize generator

        If `self.train_cfg.disc_step > 1`, the train step will contain multiple
        iterations for optimizing discriminator with different input data and
        only one iteration for optimizing generator after `disc_step`
        iterations for discriminator.

        Args:
            data_batch (torch.Tensor): Batch of data as input.
            optimizer (dict[torch.optim.Optimizer]): Dict with optimizers for
                generator and discriminator (if have).

        Returns:
            outputs (dict): Dict with loss, information for logger,
            the number of samples.
        """

        img = data_batch['img']
        pred_smpl = self.generator(img)
        pred_pose, pred_beta, pred_camera = pred_smpl

        # optimize discriminator (if have)
        if self.train_cfg['disc_step'] > 0 and self.with_gan:
            set_requires_grad(self.discriminator, True)
            fake_data = (pred_camera.detach(), pred_pose.detach(),
                         pred_beta.detach())
            mosh_theta = data_batch['mosh_theta']
            real_data = (mosh_theta[:, :3], mosh_theta[:,
                                                       3:75], mosh_theta[:,
                                                                         75:])
            fake_score = self.discriminator(fake_data)
            real_score = self.discriminator(real_data)

            disc_losses = {}
            disc_losses['real_loss'] = self.loss_gan(
                real_score, target_is_real=True, is_disc=True)
            disc_losses['fake_loss'] = self.loss_gan(
                fake_score, target_is_real=False, is_disc=True)
            loss_disc, log_vars_d = self._parse_losses(disc_losses)

            optimizer['discriminator'].zero_grad()
            loss_disc.backward()
            optimizer['discriminator'].step()
            self.disc_step_count = \
                (self.disc_step_count + 1) % self.train_cfg['disc_step']

            if self.disc_step_count != 0:
                outputs = dict(
                    loss=loss_disc,
                    log_vars=log_vars_d,
                    num_samples=len(next(iter(data_batch.values()))))
                return outputs

        # optimize generator
        pred_out = self.smpl(
            betas=pred_beta,
            body_pose=pred_pose[:, 1:],
            global_orient=pred_pose[:, :1])
        pred_vertices, pred_joints_3d = pred_out['vertices'], pred_out[
            'joints']

        gt_beta = data_batch['beta']
        gt_pose = data_batch['pose']
        gt_vertices = self.smpl(
            betas=gt_beta,
            body_pose=gt_pose[:, 3:],
            global_orient=gt_pose[:, :3])['vertices']

        pred = dict(
            pose=pred_pose,
            beta=pred_beta,
            camera=pred_camera,
            vertices=pred_vertices,
            joints_3d=pred_joints_3d)

        target = {
            key: data_batch[key]
            for key in [
                'pose', 'beta', 'has_smpl', 'joints_3d', 'joints_2d',
                'joints_3d_visible', 'joints_2d_visible'
            ]
        }
        target['vertices'] = gt_vertices

        losses = self.loss_mesh(pred, target)

        if self.with_gan:
            set_requires_grad(self.discriminator, False)
            pred_theta = (pred_camera, pred_pose, pred_beta)
            pred_score = self.discriminator(pred_theta)
            loss_adv = self.loss_gan(
                pred_score, target_is_real=True, is_disc=False)
            losses['adv_loss'] = loss_adv

        loss, log_vars = self._parse_losses(losses)
        optimizer['generator'].zero_grad()
        loss.backward()
        optimizer['generator'].step()

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))

        return outputs

    def forward_train(self, *args, **kwargs):
        """Forward function for training.

        For ParametricMesh, we do not use this interface.
        """
        raise NotImplementedError('This interface should not be used in '
                                  'current training schedule. Please use '
                                  '`train_step` for training.')

    def val_step(self, data_batch, **kwargs):
        """Forward function for evaluation.

        Args:
            data_batch (dict): Contain data for forward.

        Returns:
            dict: Contain the results from model.
        """
        output = self.forward_test(**data_batch, **kwargs)
        return output

    def forward_dummy(self, img):
        """Used for computing network FLOPs.

        See ``tools/get_flops.py``.

        Args:
            img (torch.Tensor): Input image.

        Returns:
            Tensor: Outputs.
        """
        output = self.generator(img)
        return output

    def forward_test(self,
                     img,
                     img_metas,
                     return_vertices=False,
                     return_faces=False,
                     **kwargs):
        """Defines the computation performed at every call when testing."""

        pred_smpl = self.generator(img)
        pred_pose, pred_beta, pred_camera = pred_smpl
        pred_out = self.smpl(
            betas=pred_beta,
            body_pose=pred_pose[:, 1:],
            global_orient=pred_pose[:, :1])
        pred_vertices, pred_joints_3d = pred_out['vertices'], pred_out[
            'joints']

        all_preds = {}
        all_preds['keypoints_3d'] = pred_joints_3d.detach().cpu().numpy()
        all_preds['smpl_pose'] = pred_pose.detach().cpu().numpy()
        all_preds['smpl_beta'] = pred_beta.detach().cpu().numpy()
        all_preds['camera'] = pred_camera.detach().cpu().numpy()

        if return_vertices:
            all_preds['vertices'] = pred_vertices.detach().cpu().numpy()
        if return_faces:
            all_preds['faces'] = self.smpl.get_faces()

        all_boxes = []
        image_path = []
        for img_meta in img_metas:
            box = np.zeros(6, dtype=np.float32)
            c = img_meta['center']
            s = img_meta['scale']
            if 'bbox_score' in img_metas:
                score = np.array(img_metas['bbox_score']).reshape(-1)
            else:
                score = 1.0
            box[0:2] = c
            box[2:4] = s
            box[4] = np.prod(s * 200.0, axis=0)
            box[5] = score
            all_boxes.append(box)
            image_path.append(img_meta['image_file'])

        all_preds['bboxes'] = np.stack(all_boxes, axis=0)
        all_preds['image_path'] = image_path
        return all_preds

    def get_3d_joints_from_mesh(self, vertices):
        """Get 3D joints from 3D mesh using predefined joints regressor."""
        return torch.matmul(
            self.joints_regressor.to(vertices.device), vertices)

    def forward(self, img, img_metas=None, return_loss=False, **kwargs):
        """Forward function.

        Calls either forward_train or forward_test depending on whether
        return_loss=True.

        Note:
            - batch_size: N
            - num_img_channel: C (Default: 3)
            - img height: imgH
            - img width: imgW

        Args:
            img (torch.Tensor[N x C x imgH x imgW]): Input images.
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
            Return predicted 3D joints, SMPL parameters, boxes and image paths.
        """

        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        return self.forward_test(img, img_metas, **kwargs)

    def show_result(self,
                    result,
                    img,
                    show=False,
                    out_file=None,
                    win_name='',
                    wait_time=0,
                    bbox_color='green',
                    mesh_color=(76, 76, 204),
                    **kwargs):
        """Visualize 3D mesh estimation results.

        Args:
            result (list[dict]): The mesh estimation results containing:

               - "bbox" (ndarray[4]): instance bounding bbox
               - "center" (ndarray[2]): bbox center
               - "scale" (ndarray[2]): bbox scale
               - "keypoints_3d" (ndarray[K,3]): predicted 3D keypoints
               - "camera" (ndarray[3]): camera parameters
               - "vertices" (ndarray[V, 3]): predicted 3D vertices
               - "faces" (ndarray[F, 3]): mesh faces
            img (str or Tensor): Optional. The image to visualize 2D inputs on.
            win_name (str): The window name.
            show (bool): Whether to show the image. Default: False.
            wait_time (int): Value of waitKey param. Default: 0.
            out_file (str or None): The filename to write the image.
                Default: None.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            mesh_color (str or tuple or :obj:`Color`): Color of mesh surface.

        Returns:
            ndarray: Visualized img, only if not `show` or `out_file`.
        """

        if img is not None:
            img = mmcv.imread(img)

        focal_length = self.loss_mesh.focal_length
        H, W, C = img.shape
        img_center = np.array([[0.5 * W], [0.5 * H]])

        # show bounding boxes
        bboxes = [res['bbox'] for res in result]
        bboxes = np.vstack(bboxes)
        mmcv.imshow_bboxes(
            img, bboxes, colors=bbox_color, top_k=-1, thickness=2, show=False)

        vertex_list = []
        face_list = []
        for res in result:
            vertices = res['vertices']
            faces = res['faces']
            camera = res['camera']
            camera_center = res['center']
            scale = res['scale']

            # predicted vertices are in root-relative space,
            # we need to translate them to camera space.
            translation = np.array([
                camera[1], camera[2],
                2 * focal_length / (scale[0] * 200.0 * camera[0] + 1e-9)
            ])
            mean_depth = vertices[:, -1].mean() + translation[-1]
            translation[:2] += (camera_center -
                                img_center[:, 0]) / focal_length * mean_depth
            vertices += translation[None, :]

            vertex_list.append(vertices)
            face_list.append(faces)

        # render from front view
        img_vis = imshow_mesh_3d(
            img,
            vertex_list,
            face_list,
            img_center, [focal_length, focal_length],
            colors=mesh_color)

        # render from side view
        # rotate mesh vertices
        R = cv2.Rodrigues(np.array([0, np.radians(90.), 0]))[0]
        rot_vertex_list = [np.dot(vert, R) for vert in vertex_list]

        # get the 3D bbox containing all meshes
        rot_vertices = np.concatenate(rot_vertex_list, axis=0)
        min_corner = rot_vertices.min(0)
        max_corner = rot_vertices.max(0)

        center_3d = 0.5 * (min_corner + max_corner)
        ratio = 0.8
        bbox3d_size = max_corner - min_corner

        # set appropriate translation to make all meshes appear in the image
        z_x = bbox3d_size[0] * focal_length / (ratio * W) - min_corner[2]
        z_y = bbox3d_size[1] * focal_length / (ratio * H) - min_corner[2]
        z = max(z_x, z_y)
        translation = -center_3d
        translation[2] = z
        translation = translation[None, :]
        rot_vertex_list = [
            rot_vert + translation for rot_vert in rot_vertex_list
        ]

        # render from side view
        img_side = imshow_mesh_3d(
            np.ones_like(img) * 255, rot_vertex_list, face_list, img_center,
            [focal_length, focal_length])

        # merger images from front view and side view
        img_vis = np.concatenate([img_vis, img_side], axis=1)

        if show:
            mmcv.visualization.imshow(img_vis, win_name, wait_time)

        if out_file is not None:
            mmcv.imwrite(img_vis, out_file)

        return img_vis
