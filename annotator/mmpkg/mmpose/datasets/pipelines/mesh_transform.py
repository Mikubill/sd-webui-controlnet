# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import annotator.mmpkg.mmcv as mmcv
import numpy as np
import torch

from annotator.mmpkg.mmpose.core.post_processing import (affine_transform, fliplr_joints,
                                         get_affine_transform)
from annotator.mmpkg.mmpose.datasets.builder import PIPELINES


def _flip_smpl_pose(pose):
    """Flip SMPL pose parameters horizontally.

    Args:
        pose (np.ndarray([72])): SMPL pose parameters

    Returns:
        pose_flipped
    """

    flippedParts = [
        0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11, 15, 16, 17, 12, 13, 14, 18, 19,
        20, 24, 25, 26, 21, 22, 23, 27, 28, 29, 33, 34, 35, 30, 31, 32, 36, 37,
        38, 42, 43, 44, 39, 40, 41, 45, 46, 47, 51, 52, 53, 48, 49, 50, 57, 58,
        59, 54, 55, 56, 63, 64, 65, 60, 61, 62, 69, 70, 71, 66, 67, 68
    ]
    pose_flipped = pose[flippedParts]
    # Negate the second and the third dimension of the axis-angle
    pose_flipped[1::3] = -pose_flipped[1::3]
    pose_flipped[2::3] = -pose_flipped[2::3]
    return pose_flipped


def _flip_iuv(iuv, uv_type='BF'):
    """Flip IUV image horizontally.

    Note:
        IUV image height: H
        IUV image width: W

    Args:
        iuv np.ndarray([H, W, 3]): IUV image
        uv_type (str): The type of the UV map.
            Candidate values:
                'DP': The UV map used in DensePose project.
                'SMPL': The default UV map of SMPL model.
                'BF': The UV map used in DecoMR project.
            Default: 'BF'

    Returns:
        iuv_flipped np.ndarray([H, W, 3]): Flipped IUV image
    """
    assert uv_type in ['DP', 'SMPL', 'BF']
    if uv_type == 'BF':
        iuv_flipped = iuv[:, ::-1, :]
        iuv_flipped[:, :, 1] = 255 - iuv_flipped[:, :, 1]
    else:
        # The flip of other UV map is complex, not finished yet.
        raise NotImplementedError(
            f'The flip of {uv_type} UV map is not implemented yet.')

    return iuv_flipped


def _construct_rotation_matrix(rot, size=3):
    """Construct the in-plane rotation matrix.

    Args:
        rot (float): Rotation angle (degree).
        size (int): The size of the rotation matrix.
            Candidate Values: 2, 3. Defaults to 3.

    Returns:
        rot_mat (np.ndarray([size, size]): Rotation matrix.
    """
    rot_mat = np.eye(size, dtype=np.float32)
    if rot != 0:
        rot_rad = np.deg2rad(rot)
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]

    return rot_mat


def _rotate_joints_3d(joints_3d, rot):
    """Rotate the 3D joints in the local coordinates.

    Note:
        Joints number: K

    Args:
        joints_3d (np.ndarray([K, 3])): Coordinates of keypoints.
        rot (float): Rotation angle (degree).

    Returns:
        joints_3d_rotated
    """
    # in-plane rotation
    # 3D joints are rotated counterclockwise,
    # so the rot angle is inversed.
    rot_mat = _construct_rotation_matrix(-rot, 3)

    joints_3d_rotated = np.einsum('ij,kj->ki', rot_mat, joints_3d)
    joints_3d_rotated = joints_3d_rotated.astype('float32')
    return joints_3d_rotated


def _rotate_smpl_pose(pose, rot):
    """Rotate SMPL pose parameters. SMPL (https://smpl.is.tue.mpg.de/) is a 3D
    human model.

    Args:
        pose (np.ndarray([72])): SMPL pose parameters
        rot (float): Rotation angle (degree).

    Returns:
        pose_rotated
    """
    pose_rotated = pose.copy()
    if rot != 0:
        rot_mat = _construct_rotation_matrix(-rot)
        orient = pose[:3]
        # find the rotation of the body in camera frame
        per_rdg, _ = cv2.Rodrigues(orient)
        # apply the global rotation to the global orientation
        res_rot, _ = cv2.Rodrigues(np.dot(rot_mat, per_rdg))
        pose_rotated[:3] = (res_rot.T)[0]

    return pose_rotated


def _flip_joints_3d(joints_3d, joints_3d_visible, flip_pairs):
    """Flip human joints in 3D space horizontally.

    Note:
        num_keypoints: K

    Args:
        joints_3d (np.ndarray([K, 3])): Coordinates of keypoints.
        joints_3d_visible (np.ndarray([K, 1])): Visibility of keypoints.
        flip_pairs (list[tuple()]): Pairs of keypoints which are mirrored
            (for example, left ear -- right ear).

    Returns:
        joints_3d_flipped, joints_3d_visible_flipped
    """

    assert len(joints_3d) == len(joints_3d_visible)

    joints_3d_flipped = joints_3d.copy()
    joints_3d_visible_flipped = joints_3d_visible.copy()

    # Swap left-right parts
    for left, right in flip_pairs:
        joints_3d_flipped[left, :] = joints_3d[right, :]
        joints_3d_flipped[right, :] = joints_3d[left, :]

        joints_3d_visible_flipped[left, :] = joints_3d_visible[right, :]
        joints_3d_visible_flipped[right, :] = joints_3d_visible[left, :]

    # Flip horizontally
    joints_3d_flipped[:, 0] = -joints_3d_flipped[:, 0]
    joints_3d_flipped = joints_3d_flipped * joints_3d_visible_flipped

    return joints_3d_flipped, joints_3d_visible_flipped


@PIPELINES.register_module()
class LoadIUVFromFile:
    """Loading IUV image from file."""

    def __init__(self, to_float32=False):
        self.to_float32 = to_float32
        self.color_type = 'color'
        # channel relations: iuv->bgr
        self.channel_order = 'bgr'

    def __call__(self, results):
        """Loading image from file."""
        has_iuv = results['has_iuv']
        use_iuv = results['ann_info']['use_IUV']
        if has_iuv and use_iuv:
            iuv_file = results['iuv_file']
            iuv = mmcv.imread(iuv_file, self.color_type, self.channel_order)
            if iuv is None:
                raise ValueError(f'Fail to read {iuv_file}')
        else:
            has_iuv = 0
            iuv = None

        results['has_iuv'] = has_iuv
        results['iuv'] = iuv
        return results


@PIPELINES.register_module()
class IUVToTensor:
    """Transform IUV image to part index mask and uv coordinates image. The 3
    channels of IUV image means: part index, u coordinates, v coordinates.

    Required key: 'iuv', 'ann_info'.
    Modifies key: 'part_index', 'uv_coordinates'.

    Args:
        results (dict): contain all information about training.
    """

    def __call__(self, results):
        iuv = results['iuv']
        if iuv is None:
            H, W = results['ann_info']['iuv_size']
            part_index = torch.zeros([1, H, W], dtype=torch.long)
            uv_coordinates = torch.zeros([2, H, W], dtype=torch.float32)
        else:
            part_index = torch.LongTensor(iuv[:, :, 0])[None, :, :]
            uv_coordinates = torch.FloatTensor(iuv[:, :, 1:]) / 255
            uv_coordinates = uv_coordinates.permute(2, 0, 1)
        results['part_index'] = part_index
        results['uv_coordinates'] = uv_coordinates
        return results


@PIPELINES.register_module()
class MeshRandomChannelNoise:
    """Data augmentation with random channel noise.

    Required keys: 'img'
    Modifies key: 'img'

    Args:
        noise_factor (float): Multiply each channel with
         a factor between``[1-scale_factor, 1+scale_factor]``
    """

    def __init__(self, noise_factor=0.4):
        self.noise_factor = noise_factor

    def __call__(self, results):
        """Perform data augmentation with random channel noise."""
        img = results['img']

        # Each channel is multiplied with a number
        # in the area [1-self.noise_factor, 1+self.noise_factor]
        pn = np.random.uniform(1 - self.noise_factor, 1 + self.noise_factor,
                               (1, 3))
        img = cv2.multiply(img, pn)

        results['img'] = img
        return results


@PIPELINES.register_module()
class MeshRandomFlip:
    """Data augmentation with random image flip.

    Required keys: 'img', 'joints_2d','joints_2d_visible', 'joints_3d',
    'joints_3d_visible', 'center', 'pose', 'iuv' and 'ann_info'.
    Modifies key: 'img', 'joints_2d','joints_2d_visible', 'joints_3d',
    'joints_3d_visible', 'center', 'pose', 'iuv'.

    Args:
        flip_prob (float): Probability of flip.
    """

    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, results):
        """Perform data augmentation with random image flip."""
        if np.random.rand() > self.flip_prob:
            return results

        img = results['img']
        joints_2d = results['joints_2d']
        joints_2d_visible = results['joints_2d_visible']
        joints_3d = results['joints_3d']
        joints_3d_visible = results['joints_3d_visible']
        pose = results['pose']
        center = results['center']

        img = img[:, ::-1, :]
        pose = _flip_smpl_pose(pose)

        joints_2d, joints_2d_visible = fliplr_joints(
            joints_2d, joints_2d_visible, img.shape[1],
            results['ann_info']['flip_pairs'])

        joints_3d, joints_3d_visible = _flip_joints_3d(
            joints_3d, joints_3d_visible, results['ann_info']['flip_pairs'])
        center[0] = img.shape[1] - center[0] - 1

        if 'iuv' in results.keys():
            iuv = results['iuv']
            if iuv is not None:
                iuv = _flip_iuv(iuv, results['ann_info']['uv_type'])
            results['iuv'] = iuv

        results['img'] = img
        results['joints_2d'] = joints_2d
        results['joints_2d_visible'] = joints_2d_visible
        results['joints_3d'] = joints_3d
        results['joints_3d_visible'] = joints_3d_visible
        results['pose'] = pose
        results['center'] = center
        return results


@PIPELINES.register_module()
class MeshGetRandomScaleRotation:
    """Data augmentation with random scaling & rotating.

    Required key: 'scale'. Modifies key: 'scale' and 'rotation'.

    Args:
        rot_factor (int): Rotating to ``[-2*rot_factor, 2*rot_factor]``.
        scale_factor (float): Scaling to ``[1-scale_factor, 1+scale_factor]``.
        rot_prob (float): Probability of random rotation.
    """

    def __init__(self, rot_factor=30, scale_factor=0.25, rot_prob=0.6):
        self.rot_factor = rot_factor
        self.scale_factor = scale_factor
        self.rot_prob = rot_prob

    def __call__(self, results):
        """Perform data augmentation with random scaling & rotating."""
        s = results['scale']

        sf = self.scale_factor
        rf = self.rot_factor

        s_factor = np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
        s = s * s_factor

        r_factor = np.clip(np.random.randn() * rf, -rf * 2, rf * 2)
        r = r_factor if np.random.rand() <= self.rot_prob else 0

        results['scale'] = s
        results['rotation'] = r

        return results


@PIPELINES.register_module()
class MeshAffine:
    """Affine transform the image to get input image. Affine transform the 2D
    keypoints, 3D kepoints and IUV image too.

    Required keys: 'img', 'joints_2d','joints_2d_visible', 'joints_3d',
    'joints_3d_visible', 'pose', 'iuv', 'ann_info','scale',  'rotation' and
    'center'. Modifies key: 'img', 'joints_2d','joints_2d_visible',
    'joints_3d',  'pose', 'iuv'.
    """

    def __call__(self, results):
        image_size = results['ann_info']['image_size']

        img = results['img']
        joints_2d = results['joints_2d']
        joints_2d_visible = results['joints_2d_visible']
        joints_3d = results['joints_3d']
        pose = results['pose']

        c = results['center']
        s = results['scale']
        r = results['rotation']
        trans = get_affine_transform(c, s, r, image_size)

        img = cv2.warpAffine(
            img,
            trans, (int(image_size[0]), int(image_size[1])),
            flags=cv2.INTER_LINEAR)

        for i in range(results['ann_info']['num_joints']):
            if joints_2d_visible[i, 0] > 0.0:
                joints_2d[i] = affine_transform(joints_2d[i], trans)

        joints_3d = _rotate_joints_3d(joints_3d, r)
        pose = _rotate_smpl_pose(pose, r)

        results['img'] = img
        results['joints_2d'] = joints_2d
        results['joints_2d_visible'] = joints_2d_visible
        results['joints_3d'] = joints_3d
        results['pose'] = pose

        if 'iuv' in results.keys():
            iuv = results['iuv']
            if iuv is not None:
                iuv_size = results['ann_info']['iuv_size']
                iuv = cv2.warpAffine(
                    iuv,
                    trans, (int(iuv_size[0]), int(iuv_size[1])),
                    flags=cv2.INTER_NEAREST)
            results['iuv'] = iuv

        return results
