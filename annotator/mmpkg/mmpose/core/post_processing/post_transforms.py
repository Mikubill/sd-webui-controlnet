# ------------------------------------------------------------------------------
# Adapted from https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
# Original licence: Copyright (c) Microsoft, under the MIT License.
# ------------------------------------------------------------------------------

import math

import cv2
import numpy as np
import torch


def fliplr_joints(joints_3d, joints_3d_visible, img_width, flip_pairs):
    """Flip human joints horizontally.

    Note:
        - num_keypoints: K

    Args:
        joints_3d (np.ndarray([K, 3])): Coordinates of keypoints.
        joints_3d_visible (np.ndarray([K, 1])): Visibility of keypoints.
        img_width (int): Image width.
        flip_pairs (list[tuple]): Pairs of keypoints which are mirrored
            (for example, left ear and right ear).

    Returns:
        tuple: Flipped human joints.

        - joints_3d_flipped (np.ndarray([K, 3])): Flipped joints.
        - joints_3d_visible_flipped (np.ndarray([K, 1])): Joint visibility.
    """

    assert len(joints_3d) == len(joints_3d_visible)
    assert img_width > 0

    joints_3d_flipped = joints_3d.copy()
    joints_3d_visible_flipped = joints_3d_visible.copy()

    # Swap left-right parts
    for left, right in flip_pairs:
        joints_3d_flipped[left, :] = joints_3d[right, :]
        joints_3d_flipped[right, :] = joints_3d[left, :]

        joints_3d_visible_flipped[left, :] = joints_3d_visible[right, :]
        joints_3d_visible_flipped[right, :] = joints_3d_visible[left, :]

    # Flip horizontally
    joints_3d_flipped[:, 0] = img_width - 1 - joints_3d_flipped[:, 0]
    joints_3d_flipped = joints_3d_flipped * (joints_3d_visible_flipped > 0)

    return joints_3d_flipped, joints_3d_visible_flipped


def fliplr_regression(regression,
                      flip_pairs,
                      center_mode='static',
                      center_x=0.5,
                      center_index=0):
    """Flip human joints horizontally.

    Note:
        - batch_size: N
        - num_keypoint: K

    Args:
        regression (np.ndarray([..., K, C])): Coordinates of keypoints, where K
            is the joint number and C is the dimension. Example shapes are:

            - [N, K, C]: a batch of keypoints where N is the batch size.
            - [N, T, K, C]: a batch of pose sequences, where T is the frame
                number.
        flip_pairs (list[tuple()]): Pairs of keypoints which are mirrored
            (for example, left ear -- right ear).
        center_mode (str): The mode to set the center location on the x-axis
            to flip around. Options are:

            - static: use a static x value (see center_x also)
            - root: use a root joint (see center_index also)
        center_x (float): Set the x-axis location of the flip center. Only used
            when center_mode=static.
        center_index (int): Set the index of the root joint, whose x location
            will be used as the flip center. Only used when center_mode=root.

    Returns:
        np.ndarray([..., K, C]): Flipped joints.
    """
    assert regression.ndim >= 2, f'Invalid pose shape {regression.shape}'

    allowed_center_mode = {'static', 'root'}
    assert center_mode in allowed_center_mode, 'Get invalid center_mode ' \
        f'{center_mode}, allowed choices are {allowed_center_mode}'

    if center_mode == 'static':
        x_c = center_x
    elif center_mode == 'root':
        assert regression.shape[-2] > center_index
        x_c = regression[..., center_index:center_index + 1, 0]

    regression_flipped = regression.copy()
    # Swap left-right parts
    for left, right in flip_pairs:
        regression_flipped[..., left, :] = regression[..., right, :]
        regression_flipped[..., right, :] = regression[..., left, :]

    # Flip horizontally
    regression_flipped[..., 0] = x_c * 2 - regression_flipped[..., 0]
    return regression_flipped


def flip_back(output_flipped, flip_pairs, target_type='GaussianHeatmap'):
    """Flip the flipped heatmaps back to the original form.

    Note:
        - batch_size: N
        - num_keypoints: K
        - heatmap height: H
        - heatmap width: W

    Args:
        output_flipped (np.ndarray[N, K, H, W]): The output heatmaps obtained
            from the flipped images.
        flip_pairs (list[tuple()): Pairs of keypoints which are mirrored
            (for example, left ear -- right ear).
        target_type (str): GaussianHeatmap or CombinedTarget

    Returns:
        np.ndarray: heatmaps that flipped back to the original image
    """
    assert output_flipped.ndim == 4, \
        'output_flipped should be [batch_size, num_keypoints, height, width]'
    shape_ori = output_flipped.shape
    channels = 1
    if target_type.lower() == 'CombinedTarget'.lower():
        channels = 3
        output_flipped[:, 1::3, ...] = -output_flipped[:, 1::3, ...]
    output_flipped = output_flipped.reshape(shape_ori[0], -1, channels,
                                            shape_ori[2], shape_ori[3])
    output_flipped_back = output_flipped.copy()

    # Swap left-right parts
    for left, right in flip_pairs:
        output_flipped_back[:, left, ...] = output_flipped[:, right, ...]
        output_flipped_back[:, right, ...] = output_flipped[:, left, ...]
    output_flipped_back = output_flipped_back.reshape(shape_ori)
    # Flip horizontally
    output_flipped_back = output_flipped_back[..., ::-1]
    return output_flipped_back


def transform_preds(coords, center, scale, output_size, use_udp=False):
    """Get final keypoint predictions from heatmaps and apply scaling and
    translation to map them back to the image.

    Note:
        num_keypoints: K

    Args:
        coords (np.ndarray[K, ndims]):

            * If ndims=2, corrds are predicted keypoint location.
            * If ndims=4, corrds are composed of (x, y, scores, tags)
            * If ndims=5, corrds are composed of (x, y, scores, tags,
              flipped_tags)

        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        use_udp (bool): Use unbiased data processing

    Returns:
        np.ndarray: Predicted coordinates in the images.
    """
    assert coords.shape[1] in (2, 4, 5)
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2

    # Recover the scale which is normalized by a factor of 200.
    scale = scale * 200.0

    if use_udp:
        scale_x = scale[0] / (output_size[0] - 1.0)
        scale_y = scale[1] / (output_size[1] - 1.0)
    else:
        scale_x = scale[0] / output_size[0]
        scale_y = scale[1] / output_size[1]

    target_coords = coords.copy()
    target_coords[:, 0] = coords[:, 0] * scale_x + center[0] - scale[0] * 0.5
    target_coords[:, 1] = coords[:, 1] * scale_y + center[1] - scale[1] * 0.5

    return target_coords


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=(0., 0.),
                         inv=False):
    """Get the affine transform matrix, given the center/scale/rot/output_size.

    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        shift (0-100%): Shift translation ratio wrt the width/height.
            Default (0., 0.).
        inv (bool): Option to inverse the affine transform direction.
            (inv=False: src->dst or inv=True: dst->src)

    Returns:
        np.ndarray: The transform matrix.
    """
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2
    assert len(shift) == 2

    # pixel_std is 200.
    scale_tmp = scale * 200.0

    shift = np.array(shift)
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = rotate_point([0., src_w * -0.5], rot_rad)
    dst_dir = np.array([0., dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, trans_mat):
    """Apply an affine transformation to the points.

    Args:
        pt (np.ndarray): a 2 dimensional point to be transformed
        trans_mat (np.ndarray): 2x3 matrix of an affine transform

    Returns:
        np.ndarray: Transformed points.
    """
    assert len(pt) == 2
    new_pt = np.array(trans_mat) @ np.array([pt[0], pt[1], 1.])

    return new_pt


def _get_3rd_point(a, b):
    """To calculate the affine matrix, three pairs of points are required. This
    function is used to get the 3rd point, given 2D points a & b.

    The 3rd point is defined by rotating vector `a - b` by 90 degrees
    anticlockwise, using b as the rotation center.

    Args:
        a (np.ndarray): point(x,y)
        b (np.ndarray): point(x,y)

    Returns:
        np.ndarray: The 3rd point.
    """
    assert len(a) == 2
    assert len(b) == 2
    direction = a - b
    third_pt = b + np.array([-direction[1], direction[0]], dtype=np.float32)

    return third_pt


def rotate_point(pt, angle_rad):
    """Rotate a point by an angle.

    Args:
        pt (list[float]): 2 dimensional point to be rotated
        angle_rad (float): rotation angle by radian

    Returns:
        list[float]: Rotated point.
    """
    assert len(pt) == 2
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    new_x = pt[0] * cs - pt[1] * sn
    new_y = pt[0] * sn + pt[1] * cs
    rotated_pt = [new_x, new_y]

    return rotated_pt


def get_warp_matrix(theta, size_input, size_dst, size_target):
    """Calculate the transformation matrix under the constraint of unbiased.
    Paper ref: Huang et al. The Devil is in the Details: Delving into Unbiased
    Data Processing for Human Pose Estimation (CVPR 2020).

    Args:
        theta (float): Rotation angle in degrees.
        size_input (np.ndarray): Size of input image [w, h].
        size_dst (np.ndarray): Size of output image [w, h].
        size_target (np.ndarray): Size of ROI in input plane [w, h].

    Returns:
        np.ndarray: A matrix for transformation.
    """
    theta = np.deg2rad(theta)
    matrix = np.zeros((2, 3), dtype=np.float32)
    scale_x = size_dst[0] / size_target[0]
    scale_y = size_dst[1] / size_target[1]
    matrix[0, 0] = math.cos(theta) * scale_x
    matrix[0, 1] = -math.sin(theta) * scale_x
    matrix[0, 2] = scale_x * (-0.5 * size_input[0] * math.cos(theta) +
                              0.5 * size_input[1] * math.sin(theta) +
                              0.5 * size_target[0])
    matrix[1, 0] = math.sin(theta) * scale_y
    matrix[1, 1] = math.cos(theta) * scale_y
    matrix[1, 2] = scale_y * (-0.5 * size_input[0] * math.sin(theta) -
                              0.5 * size_input[1] * math.cos(theta) +
                              0.5 * size_target[1])
    return matrix


def warp_affine_joints(joints, mat):
    """Apply affine transformation defined by the transform matrix on the
    joints.

    Args:
        joints (np.ndarray[..., 2]): Origin coordinate of joints.
        mat (np.ndarray[3, 2]): The affine matrix.

    Returns:
        np.ndarray[..., 2]: Result coordinate of joints.
    """
    joints = np.array(joints)
    shape = joints.shape
    joints = joints.reshape(-1, 2)
    return np.dot(
        np.concatenate((joints, joints[:, 0:1] * 0 + 1), axis=1),
        mat.T).reshape(shape)


def affine_transform_torch(pts, t):
    npts = pts.shape[0]
    pts_homo = torch.cat([pts, torch.ones(npts, 1, device=pts.device)], dim=1)
    out = torch.mm(t, torch.t(pts_homo))
    return torch.t(out[:2, :])
