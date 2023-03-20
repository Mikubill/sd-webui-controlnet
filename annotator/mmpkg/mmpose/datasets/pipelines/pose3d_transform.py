# Copyright (c) OpenMMLab. All rights reserved.
import copy
import random

import annotator.mmpkg.mmcv as mmcv
import numpy as np
import torch
from annotator.mmpkg.mmcv.utils import build_from_cfg

from annotator.mmpkg.mmpose.core.camera import CAMERAS
from annotator.mmpkg.mmpose.core.post_processing import (affine_transform, fliplr_regression,
                                         get_affine_transform)
from annotator.mmpkg.mmpose.datasets.builder import PIPELINES


@PIPELINES.register_module()
class GetRootCenteredPose:
    """Zero-center the pose around a given root joint. Optionally, the root
    joint can be removed from the original pose and stored as a separate item.

    Note that the root-centered joints may no longer align with some annotation
    information (e.g. flip_pairs, num_joints, inference_channel, etc.) due to
    the removal of the root joint.

    Args:
        item (str): The name of the pose to apply root-centering.
        root_index (int): Root joint index in the pose.
        visible_item (str): The name of the visibility item.
        remove_root (bool): If true, remove the root joint from the pose
        root_name (str): Optional. If not none, it will be used as the key to
            store the root position separated from the original pose.

    Required keys:
        item

    Modified keys:
        item, visible_item, root_name
    """

    def __init__(self,
                 item,
                 root_index,
                 visible_item=None,
                 remove_root=False,
                 root_name=None):
        self.item = item
        self.root_index = root_index
        self.remove_root = remove_root
        self.root_name = root_name
        self.visible_item = visible_item

    def __call__(self, results):
        assert self.item in results
        joints = results[self.item]
        root_idx = self.root_index

        assert joints.ndim >= 2 and joints.shape[-2] > root_idx,\
            f'Got invalid joint shape {joints.shape}'

        root = joints[..., root_idx:root_idx + 1, :]
        joints = joints - root

        results[self.item] = joints
        if self.root_name is not None:
            results[self.root_name] = root

        if self.remove_root:
            results[self.item] = np.delete(
                results[self.item], root_idx, axis=-2)
            if self.visible_item is not None:
                assert self.visible_item in results
                results[self.visible_item] = np.delete(
                    results[self.visible_item], root_idx, axis=-2)
            # Add a flag to avoid latter transforms that rely on the root
            # joint or the original joint index
            results[f'{self.item}_root_removed'] = True

            # Save the root index which is necessary to restore the global pose
            if self.root_name is not None:
                results[f'{self.root_name}_index'] = self.root_index

        return results


@PIPELINES.register_module()
class NormalizeJointCoordinate:
    """Normalize the joint coordinate with given mean and std.

    Args:
        item (str): The name of the pose to normalize.
        mean (array): Mean values of joint coordinates in shape [K, C].
        std (array): Std values of joint coordinates in shape [K, C].
        norm_param_file (str): Optionally load a dict containing `mean` and
            `std` from a file using `mmcv.load`.

    Required keys:
        item

    Modified keys:
        item
    """

    def __init__(self, item, mean=None, std=None, norm_param_file=None):
        self.item = item
        self.norm_param_file = norm_param_file
        if norm_param_file is not None:
            norm_param = mmcv.load(norm_param_file)
            assert 'mean' in norm_param and 'std' in norm_param
            mean = norm_param['mean']
            std = norm_param['std']
        else:
            assert mean is not None
            assert std is not None

        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, results):
        assert self.item in results
        results[self.item] = (results[self.item] - self.mean) / self.std
        results[f'{self.item}_mean'] = self.mean.copy()
        results[f'{self.item}_std'] = self.std.copy()
        return results


@PIPELINES.register_module()
class ImageCoordinateNormalization:
    """Normalize the 2D joint coordinate with image width and height. Range [0,
    w] is mapped to [-1, 1], while preserving the aspect ratio.

    Args:
        item (str|list[str]): The name of the pose to normalize.
        norm_camera (bool): Whether to normalize camera intrinsics.
            Default: False.
        camera_param (dict|None): The camera parameter dict. See the camera
            class definition for more details. If None is given, the camera
            parameter will be obtained during processing of each data sample
            with the key "camera_param".

    Required keys:
        item

    Modified keys:
        item (, camera_param)
    """

    def __init__(self, item, norm_camera=False, camera_param=None):
        self.item = item
        if isinstance(self.item, str):
            self.item = [self.item]

        self.norm_camera = norm_camera

        if camera_param is None:
            self.static_camera = False
        else:
            self.static_camera = True
            self.camera_param = camera_param

    def __call__(self, results):
        center = np.array(
            [0.5 * results['image_width'], 0.5 * results['image_height']],
            dtype=np.float32)
        scale = np.array(0.5 * results['image_width'], dtype=np.float32)

        for item in self.item:
            results[item] = (results[item] - center) / scale

        if self.norm_camera:
            if self.static_camera:
                camera_param = copy.deepcopy(self.camera_param)
            else:
                assert 'camera_param' in results, \
                    'Camera parameters are missing.'
                camera_param = results['camera_param']
            assert 'f' in camera_param and 'c' in camera_param
            camera_param['f'] = camera_param['f'] / scale
            camera_param['c'] = (camera_param['c'] - center[:, None]) / scale
            if 'camera_param' not in results:
                results['camera_param'] = dict()
            results['camera_param'].update(camera_param)

        return results


@PIPELINES.register_module()
class CollectCameraIntrinsics:
    """Store camera intrinsics in a 1-dim array, including f, c, k, p.

    Args:
        camera_param (dict|None): The camera parameter dict. See the camera
            class definition for more details. If None is given, the camera
            parameter will be obtained during processing of each data sample
            with the key "camera_param".
        need_distortion (bool): Whether need distortion parameters k and p.
            Default: True.

    Required keys:
        camera_param (if camera parameters are not given in initialization)

    Modified keys:
        intrinsics
    """

    def __init__(self, camera_param=None, need_distortion=True):
        if camera_param is None:
            self.static_camera = False
        else:
            self.static_camera = True
            self.camera_param = camera_param
        self.need_distortion = need_distortion

    def __call__(self, results):
        if self.static_camera:
            camera_param = copy.deepcopy(self.camera_param)
        else:
            assert 'camera_param' in results, 'Camera parameters are missing.'
            camera_param = results['camera_param']
        assert 'f' in camera_param and 'c' in camera_param
        intrinsics = np.concatenate(
            [camera_param['f'].reshape(2), camera_param['c'].reshape(2)])
        if self.need_distortion:
            assert 'k' in camera_param and 'p' in camera_param
            intrinsics = np.concatenate([
                intrinsics, camera_param['k'].reshape(3),
                camera_param['p'].reshape(2)
            ])
        results['intrinsics'] = intrinsics

        return results


@PIPELINES.register_module()
class CameraProjection:
    """Apply camera projection to joint coordinates.

    Args:
        item (str): The name of the pose to apply camera projection.
        mode (str): The type of camera projection, supported options are

            - world_to_camera
            - world_to_pixel
            - camera_to_world
            - camera_to_pixel
        output_name (str|None): The name of the projected pose. If None
            (default) is given, the projected pose will be stored in place.
        camera_type (str): The camera class name (should be registered in
            CAMERA).
        camera_param (dict|None): The camera parameter dict. See the camera
            class definition for more details. If None is given, the camera
            parameter will be obtained during processing of each data sample
            with the key "camera_param".

    Required keys:

        - item
        - camera_param (if camera parameters are not given in initialization)

    Modified keys:
        output_name
    """

    def __init__(self,
                 item,
                 mode,
                 output_name=None,
                 camera_type='SimpleCamera',
                 camera_param=None):
        self.item = item
        self.mode = mode
        self.output_name = output_name
        self.camera_type = camera_type
        allowed_mode = {
            'world_to_camera',
            'world_to_pixel',
            'camera_to_world',
            'camera_to_pixel',
        }
        if mode not in allowed_mode:
            raise ValueError(
                f'Got invalid mode: {mode}, allowed modes are {allowed_mode}')

        if camera_param is None:
            self.static_camera = False
        else:
            self.static_camera = True
            self.camera = self._build_camera(camera_param)

    def _build_camera(self, param):
        cfgs = dict(type=self.camera_type, param=param)
        return build_from_cfg(cfgs, CAMERAS)

    def __call__(self, results):
        assert self.item in results
        joints = results[self.item]

        if self.static_camera:
            camera = self.camera
        else:
            assert 'camera_param' in results, 'Camera parameters are missing.'
            camera = self._build_camera(results['camera_param'])

        if self.mode == 'world_to_camera':
            output = camera.world_to_camera(joints)
        elif self.mode == 'world_to_pixel':
            output = camera.world_to_pixel(joints)
        elif self.mode == 'camera_to_world':
            output = camera.camera_to_world(joints)
        elif self.mode == 'camera_to_pixel':
            output = camera.camera_to_pixel(joints)
        else:
            raise NotImplementedError

        output_name = self.output_name
        if output_name is None:
            output_name = self.item

        results[output_name] = output
        return results


@PIPELINES.register_module()
class RelativeJointRandomFlip:
    """Data augmentation with random horizontal joint flip around a root joint.

    Args:
        item (str|list[str]): The name of the pose to flip.
        flip_cfg (dict|list[dict]): Configurations of the fliplr_regression
            function. It should contain the following arguments:

            - ``center_mode``: The mode to set the center location on the \
                x-axis to flip around.
            - ``center_x`` or ``center_index``: Set the x-axis location or \
                the root joint's index to define the flip center.

            Please refer to the docstring of the fliplr_regression function for
            more details.
        visible_item (str|list[str]): The name of the visibility item which
            will be flipped accordingly along with the pose.
        flip_prob (float): Probability of flip.
        flip_camera (bool): Whether to flip horizontal distortion coefficients.
        camera_param (dict|None): The camera parameter dict. See the camera
            class definition for more details. If None is given, the camera
            parameter will be obtained during processing of each data sample
            with the key "camera_param".

    Required keys:
        item

    Modified keys:
        item (, camera_param)
    """

    def __init__(self,
                 item,
                 flip_cfg,
                 visible_item=None,
                 flip_prob=0.5,
                 flip_camera=False,
                 camera_param=None):
        self.item = item
        self.flip_cfg = flip_cfg
        self.vis_item = visible_item
        self.flip_prob = flip_prob
        self.flip_camera = flip_camera
        if camera_param is None:
            self.static_camera = False
        else:
            self.static_camera = True
            self.camera_param = camera_param

        if isinstance(self.item, str):
            self.item = [self.item]
        if isinstance(self.flip_cfg, dict):
            self.flip_cfg = [self.flip_cfg] * len(self.item)
        assert len(self.item) == len(self.flip_cfg)
        if isinstance(self.vis_item, str):
            self.vis_item = [self.vis_item]

    def __call__(self, results):

        if results.get(f'{self.item}_root_removed', False):
            raise RuntimeError('The transform RelativeJointRandomFlip should '
                               f'not be applied to {self.item} whose root '
                               'joint has been removed and joint indices have '
                               'been changed')

        if np.random.rand() <= self.flip_prob:

            flip_pairs = results['ann_info']['flip_pairs']

            # flip joint coordinates
            for i, item in enumerate(self.item):
                assert item in results
                joints = results[item]

                joints_flipped = fliplr_regression(joints, flip_pairs,
                                                   **self.flip_cfg[i])

                results[item] = joints_flipped

            # flip joint visibility
            for vis_item in self.vis_item:
                assert vis_item in results
                visible = results[vis_item]
                visible_flipped = visible.copy()
                for left, right in flip_pairs:
                    visible_flipped[..., left, :] = visible[..., right, :]
                    visible_flipped[..., right, :] = visible[..., left, :]
                results[vis_item] = visible_flipped

            # flip horizontal distortion coefficients
            if self.flip_camera:
                if self.static_camera:
                    camera_param = copy.deepcopy(self.camera_param)
                else:
                    assert 'camera_param' in results, \
                        'Camera parameters are missing.'
                    camera_param = results['camera_param']
                assert 'c' in camera_param
                camera_param['c'][0] *= -1

                if 'p' in camera_param:
                    camera_param['p'][0] *= -1

                if 'camera_param' not in results:
                    results['camera_param'] = dict()
                results['camera_param'].update(camera_param)

        return results


@PIPELINES.register_module()
class PoseSequenceToTensor:
    """Convert pose sequence from numpy array to Tensor.

    The original pose sequence should have a shape of [T,K,C] or [K,C], where
    T is the sequence length, K and C are keypoint number and dimension. The
    converted pose sequence will have a shape of [KxC, T].

    Args:
        item (str): The name of the pose sequence

    Required keys:
        item

    Modified keys:
        item
    """

    def __init__(self, item):
        self.item = item

    def __call__(self, results):
        assert self.item in results
        seq = results[self.item]

        assert isinstance(seq, np.ndarray)
        assert seq.ndim in {2, 3}

        if seq.ndim == 2:
            seq = seq[None, ...]

        T = seq.shape[0]
        seq = seq.transpose(1, 2, 0).reshape(-1, T)
        results[self.item] = torch.from_numpy(seq)

        return results


@PIPELINES.register_module()
class Generate3DHeatmapTarget:
    """Generate the target 3d heatmap.

    Required keys: 'joints_3d', 'joints_3d_visible', 'ann_info'.
    Modified keys: 'target', and 'target_weight'.

    Args:
        sigma: Sigma of heatmap gaussian.
        joint_indices (list): Indices of joints used for heatmap generation.
            If None (default) is given, all joints will be used.
        max_bound (float): The maximal value of heatmap.
    """

    def __init__(self, sigma=2, joint_indices=None, max_bound=1.0):
        self.sigma = sigma
        self.joint_indices = joint_indices
        self.max_bound = max_bound

    def __call__(self, results):
        """Generate the target heatmap."""
        joints_3d = results['joints_3d']
        joints_3d_visible = results['joints_3d_visible']
        cfg = results['ann_info']
        image_size = cfg['image_size']
        W, H, D = cfg['heatmap_size']
        heatmap3d_depth_bound = cfg['heatmap3d_depth_bound']
        joint_weights = cfg['joint_weights']
        use_different_joint_weights = cfg['use_different_joint_weights']

        # select the joints used for target generation
        if self.joint_indices is not None:
            joints_3d = joints_3d[self.joint_indices, ...]
            joints_3d_visible = joints_3d_visible[self.joint_indices, ...]
            joint_weights = joint_weights[self.joint_indices, ...]
        num_joints = joints_3d.shape[0]

        # get the joint location in heatmap coordinates
        mu_x = joints_3d[:, 0] * W / image_size[0]
        mu_y = joints_3d[:, 1] * H / image_size[1]
        mu_z = (joints_3d[:, 2] / heatmap3d_depth_bound + 0.5) * D

        target = np.zeros([num_joints, D, H, W], dtype=np.float32)

        target_weight = joints_3d_visible[:, 0].astype(np.float32)
        target_weight = target_weight * (mu_z >= 0) * (mu_z < D)
        if use_different_joint_weights:
            target_weight = target_weight * joint_weights
        target_weight = target_weight[:, None]

        # only compute the voxel value near the joints location
        tmp_size = 3 * self.sigma

        # get neighboring voxels coordinates
        x = y = z = np.arange(2 * tmp_size + 1, dtype=np.float32) - tmp_size
        zz, yy, xx = np.meshgrid(z, y, x)
        xx = xx[None, ...].astype(np.float32)
        yy = yy[None, ...].astype(np.float32)
        zz = zz[None, ...].astype(np.float32)
        mu_x = mu_x[..., None, None, None]
        mu_y = mu_y[..., None, None, None]
        mu_z = mu_z[..., None, None, None]
        xx, yy, zz = xx + mu_x, yy + mu_y, zz + mu_z

        # round the coordinates
        xx = xx.round().clip(0, W - 1)
        yy = yy.round().clip(0, H - 1)
        zz = zz.round().clip(0, D - 1)

        # compute the target value near joints
        local_target = \
            np.exp(-((xx - mu_x)**2 + (yy - mu_y)**2 + (zz - mu_z)**2) /
                   (2 * self.sigma**2))

        # put the local target value to the full target heatmap
        local_size = xx.shape[1]
        idx_joints = np.tile(
            np.arange(num_joints)[:, None, None, None],
            [1, local_size, local_size, local_size])
        idx = np.stack([idx_joints, zz, yy, xx],
                       axis=-1).astype(int).reshape(-1, 4)
        target[idx[:, 0], idx[:, 1], idx[:, 2],
               idx[:, 3]] = local_target.reshape(-1)
        target = target * self.max_bound
        results['target'] = target
        results['target_weight'] = target_weight
        return results


@PIPELINES.register_module()
class GenerateVoxel3DHeatmapTarget:
    """Generate the target 3d heatmap.

    Required keys: 'joints_3d', 'joints_3d_visible', 'ann_info_3d'.
    Modified keys: 'target', and 'target_weight'.

    Args:
        sigma: Sigma of heatmap gaussian (mm).
        joint_indices (list): Indices of joints used for heatmap generation.
            If None (default) is given, all joints will be used.
    """

    def __init__(self, sigma=200.0, joint_indices=None):
        self.sigma = sigma  # mm
        self.joint_indices = joint_indices

    def __call__(self, results):
        """Generate the target heatmap."""
        joints_3d = results['joints_3d']
        joints_3d_visible = results['joints_3d_visible']
        cfg = results['ann_info']

        num_people = len(joints_3d)
        num_joints = joints_3d[0].shape[0]

        if self.joint_indices is not None:
            num_joints = len(self.joint_indices)
            joint_indices = self.joint_indices
        else:
            joint_indices = list(range(num_joints))

        space_size = cfg['space_size']
        space_center = cfg['space_center']
        cube_size = cfg['cube_size']
        grids_x = np.linspace(-space_size[0] / 2, space_size[0] / 2,
                              cube_size[0]) + space_center[0]
        grids_y = np.linspace(-space_size[1] / 2, space_size[1] / 2,
                              cube_size[1]) + space_center[1]
        grids_z = np.linspace(-space_size[2] / 2, space_size[2] / 2,
                              cube_size[2]) + space_center[2]

        target = np.zeros(
            (num_joints, cube_size[0], cube_size[1], cube_size[2]),
            dtype=np.float32)

        for n in range(num_people):
            for idx, joint_id in enumerate(joint_indices):
                assert joints_3d.shape[2] == 3

                mu_x = np.mean(joints_3d[n][joint_id, 0])
                mu_y = np.mean(joints_3d[n][joint_id, 1])
                mu_z = np.mean(joints_3d[n][joint_id, 2])
                vis = np.mean(joints_3d_visible[n][joint_id, 0])
                if vis < 1:
                    continue
                i_x = [
                    np.searchsorted(grids_x, mu_x - 3 * self.sigma),
                    np.searchsorted(grids_x, mu_x + 3 * self.sigma, 'right')
                ]
                i_y = [
                    np.searchsorted(grids_y, mu_y - 3 * self.sigma),
                    np.searchsorted(grids_y, mu_y + 3 * self.sigma, 'right')
                ]
                i_z = [
                    np.searchsorted(grids_z, mu_z - 3 * self.sigma),
                    np.searchsorted(grids_z, mu_z + 3 * self.sigma, 'right')
                ]
                if i_x[0] >= i_x[1] or i_y[0] >= i_y[1] or i_z[0] >= i_z[1]:
                    continue
                kernel_xs, kernel_ys, kernel_zs = np.meshgrid(
                    grids_x[i_x[0]:i_x[1]],
                    grids_y[i_y[0]:i_y[1]],
                    grids_z[i_z[0]:i_z[1]],
                    indexing='ij')
                g = np.exp(-((kernel_xs - mu_x)**2 + (kernel_ys - mu_y)**2 +
                             (kernel_zs - mu_z)**2) / (2 * self.sigma**2))
                target[idx, i_x[0]:i_x[1], i_y[0]:i_y[1], i_z[0]:i_z[1]] \
                    = np.maximum(target[idx, i_x[0]:i_x[1],
                                 i_y[0]:i_y[1], i_z[0]:i_z[1]], g)

        target = np.clip(target, 0, 1)
        if target.shape[0] == 1:
            target = target[0]

        results['targets_3d'] = target

        return results


@PIPELINES.register_module()
class AffineJoints:
    """Apply affine transformation to joints coordinates.

    Args:
        item (str): The name of the joints to apply affine.
        visible_item (str): The name of the visibility item.

    Required keys:
        item, visible_item(optional)

    Modified keys:
        item, visible_item(optional)
    """

    def __init__(self, item='joints', visible_item=None):
        self.item = item
        self.visible_item = visible_item

    def __call__(self, results):
        """Perform random affine transformation to joints coordinates."""

        c = results['center']
        s = results['scale'] / 200.0
        r = results['rotation']
        image_size = results['ann_info']['image_size']

        assert self.item in results
        joints = results[self.item]

        if self.visible_item is not None:
            assert self.visible_item in results
            joints_vis = results[self.visible_item]
        else:
            joints_vis = [np.ones_like(joints[0]) for _ in range(len(joints))]

        trans = get_affine_transform(c, s, r, image_size)
        nposes = len(joints)
        for n in range(nposes):
            for i in range(len(joints[0])):
                if joints_vis[n][i, 0] > 0.0:
                    joints[n][i,
                              0:2] = affine_transform(joints[n][i, 0:2], trans)
                    if (np.min(joints[n][i, :2]) < 0
                            or joints[n][i, 0] >= image_size[0]
                            or joints[n][i, 1] >= image_size[1]):
                        joints_vis[n][i, :] = 0

        results[self.item] = joints
        if self.visible_item is not None:
            results[self.visible_item] = joints_vis

        return results


@PIPELINES.register_module()
class GenerateInputHeatmaps:
    """Generate 2D input heatmaps for multi-camera heatmaps when the 2D model
    is not available.

    Required keys: 'joints'
    Modified keys: 'input_heatmaps'

    Args:
        sigma (int): Sigma of heatmap gaussian (mm).
        base_size (int): the base size of human
        target_type (str): type of target heatmap, only support 'gaussian' now
    """

    def __init__(self,
                 item='joints',
                 visible_item=None,
                 obscured=0.0,
                 from_pred=True,
                 sigma=3,
                 scale=None,
                 base_size=96,
                 target_type='gaussian',
                 heatmap_cfg=None):
        self.item = item
        self.visible_item = visible_item
        self.obscured = obscured
        self.from_pred = from_pred
        self.sigma = sigma
        self.scale = scale
        self.base_size = base_size
        self.target_type = target_type
        self.heatmap_cfg = heatmap_cfg

    def _compute_human_scale(self, pose, joints_vis):
        idx = joints_vis[:, 0] == 1
        if np.sum(idx) == 0:
            return 0
        minx, maxx = np.min(pose[idx, 0]), np.max(pose[idx, 0])
        miny, maxy = np.min(pose[idx, 1]), np.max(pose[idx, 1])

        return np.clip(
            np.maximum(maxy - miny, maxx - minx)**2, (self.base_size / 2)**2,
            (self.base_size * 2)**2)

    def __call__(self, results):
        assert self.target_type == 'gaussian', 'Only support gaussian map now'
        assert results['ann_info'][
            'num_scales'] == 1, 'Only support one scale now'
        heatmap_size = results['ann_info']['heatmap_size'][0]

        num_joints = results['ann_info']['num_joints']
        image_size = results['ann_info']['image_size']

        joints = results[self.item]

        if self.visible_item is not None:
            assert self.visible_item in results
            joints_vis = results[self.visible_item]
        else:
            joints_vis = [np.ones_like(joints[0]) for _ in range(len(joints))]

        nposes = len(joints)
        target = np.zeros((num_joints, heatmap_size[1], heatmap_size[0]),
                          dtype=np.float32)
        feat_stride = image_size / heatmap_size

        for n in range(nposes):
            if random.random() < self.obscured:
                continue

            human_scale = 2 * self._compute_human_scale(
                joints[n][:, 0:2] / feat_stride, joints_vis[n])
            if human_scale == 0:
                continue
            cur_sigma = self.sigma * np.sqrt(
                (human_scale / (self.base_size**2)))
            tmp_size = cur_sigma * 3
            for joint_id in range(num_joints):
                feat_stride = image_size / heatmap_size
                mu_x = int(joints[n][joint_id][0] / feat_stride[0])
                mu_y = int(joints[n][joint_id][1] / feat_stride[1])

                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= heatmap_size[0] or \
                        ul[1] >= heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    continue

                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2

                # determine the value of scale
                if self.from_pred:
                    if self.scale is None:
                        scale = joints[n][joint_id][2] if len(
                            joints[n][joint_id]) == 3 else 1.0
                    else:
                        scale = self.scale
                else:
                    if self.heatmap_cfg is None:
                        scale = self.scale
                    else:
                        base_scale = self.heatmap_cfg['base_scale']
                        offset = self.heatmap_cfg['offset']
                        thr = self.heatmap_cfg['threshold']
                        scale = (base_scale + np.random.randn(1) * offset
                                 ) if random.random() < thr else self.scale

                        for cfg in self.heatmap_cfg['extra']:
                            if joint_id in cfg['joint_ids']:
                                scale = scale * cfg[
                                    'scale_factor'] if random.random(
                                    ) < cfg['threshold'] else scale

                g = np.exp(-((x - x0)**2 + (y - y0)**2) /
                           (2 * cur_sigma**2)) * scale

                # usable gaussian range
                g_x = max(0, 0 - ul[0]), min(br[0], heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]

                # Image range
                img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

                target[joint_id][img_y[0]:img_y[1],
                                 img_x[0]:img_x[1]] = np.maximum(
                                     target[joint_id][img_y[0]:img_y[1],
                                                      img_x[0]:img_x[1]],
                                     g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
            target = np.clip(target, 0, 1)

        # target can be extended to multi-scale,
        # if results['ann_info']['num_scales'] > 1
        results['input_heatmaps'] = [target]
        return results
