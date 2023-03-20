# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from .mesh_eval import compute_similarity_transform


def keypoint_mpjpe(pred, gt, mask, alignment='none'):
    """Calculate the mean per-joint position error (MPJPE) and the error after
    rigid alignment with the ground truth (P-MPJPE).

    Note:
        - batch_size: N
        - num_keypoints: K
        - keypoint_dims: C

    Args:
        pred (np.ndarray): Predicted keypoint location with shape [N, K, C].
        gt (np.ndarray): Groundtruth keypoint location with shape [N, K, C].
        mask (np.ndarray): Visibility of the target with shape [N, K].
            False for invisible joints, and True for visible.
            Invisible joints will be ignored for accuracy calculation.
        alignment (str, optional): method to align the prediction with the
            groundtruth. Supported options are:

                - ``'none'``: no alignment will be applied
                - ``'scale'``: align in the least-square sense in scale
                - ``'procrustes'``: align in the least-square sense in
                    scale, rotation and translation.
    Returns:
        tuple: A tuple containing joint position errors

        - (float | np.ndarray): mean per-joint position error (mpjpe).
        - (float | np.ndarray): mpjpe after rigid alignment with the
            ground truth (p-mpjpe).
    """
    assert mask.any()

    if alignment == 'none':
        pass
    elif alignment == 'procrustes':
        pred = np.stack([
            compute_similarity_transform(pred_i, gt_i)
            for pred_i, gt_i in zip(pred, gt)
        ])
    elif alignment == 'scale':
        pred_dot_pred = np.einsum('nkc,nkc->n', pred, pred)
        pred_dot_gt = np.einsum('nkc,nkc->n', pred, gt)
        scale_factor = pred_dot_gt / pred_dot_pred
        pred = pred * scale_factor[:, None, None]
    else:
        raise ValueError(f'Invalid value for alignment: {alignment}')

    error = np.linalg.norm(pred - gt, ord=2, axis=-1)[mask].mean()

    return error


def keypoint_3d_pck(pred, gt, mask, alignment='none', threshold=0.15):
    """Calculate the Percentage of Correct Keypoints (3DPCK) w. or w/o rigid
    alignment.

    Paper ref: `Monocular 3D Human Pose Estimation In The Wild Using Improved
    CNN Supervision' 3DV'2017. <https://arxiv.org/pdf/1611.09813>`__ .

    Note:
        - batch_size: N
        - num_keypoints: K
        - keypoint_dims: C

    Args:
        pred (np.ndarray[N, K, C]): Predicted keypoint location.
        gt (np.ndarray[N, K, C]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        alignment (str, optional): method to align the prediction with the
            groundtruth. Supported options are:

            - ``'none'``: no alignment will be applied
            - ``'scale'``: align in the least-square sense in scale
            - ``'procrustes'``: align in the least-square sense in scale,
                rotation and translation.

        threshold:  If L2 distance between the prediction and the groundtruth
            is less then threshold, the predicted result is considered as
            correct. Default: 0.15 (m).

    Returns:
        pck: percentage of correct keypoints.
    """
    assert mask.any()

    if alignment == 'none':
        pass
    elif alignment == 'procrustes':
        pred = np.stack([
            compute_similarity_transform(pred_i, gt_i)
            for pred_i, gt_i in zip(pred, gt)
        ])
    elif alignment == 'scale':
        pred_dot_pred = np.einsum('nkc,nkc->n', pred, pred)
        pred_dot_gt = np.einsum('nkc,nkc->n', pred, gt)
        scale_factor = pred_dot_gt / pred_dot_pred
        pred = pred * scale_factor[:, None, None]
    else:
        raise ValueError(f'Invalid value for alignment: {alignment}')

    error = np.linalg.norm(pred - gt, ord=2, axis=-1)
    pck = (error < threshold).astype(np.float32)[mask].mean() * 100

    return pck


def keypoint_3d_auc(pred, gt, mask, alignment='none'):
    """Calculate the Area Under the Curve (3DAUC) computed for a range of 3DPCK
    thresholds.

    Paper ref: `Monocular 3D Human Pose Estimation In The Wild Using Improved
    CNN Supervision' 3DV'2017. <https://arxiv.org/pdf/1611.09813>`__ .
    This implementation is derived from mpii_compute_3d_pck.m, which is
    provided as part of the MPI-INF-3DHP test data release.

    Note:
        batch_size: N
        num_keypoints: K
        keypoint_dims: C

    Args:
        pred (np.ndarray[N, K, C]): Predicted keypoint location.
        gt (np.ndarray[N, K, C]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        alignment (str, optional): method to align the prediction with the
            groundtruth. Supported options are:

            - ``'none'``: no alignment will be applied
            - ``'scale'``: align in the least-square sense in scale
            - ``'procrustes'``: align in the least-square sense in scale,
                rotation and translation.

    Returns:
        auc: AUC computed for a range of 3DPCK thresholds.
    """
    assert mask.any()

    if alignment == 'none':
        pass
    elif alignment == 'procrustes':
        pred = np.stack([
            compute_similarity_transform(pred_i, gt_i)
            for pred_i, gt_i in zip(pred, gt)
        ])
    elif alignment == 'scale':
        pred_dot_pred = np.einsum('nkc,nkc->n', pred, pred)
        pred_dot_gt = np.einsum('nkc,nkc->n', pred, gt)
        scale_factor = pred_dot_gt / pred_dot_pred
        pred = pred * scale_factor[:, None, None]
    else:
        raise ValueError(f'Invalid value for alignment: {alignment}')

    error = np.linalg.norm(pred - gt, ord=2, axis=-1)

    thresholds = np.linspace(0., 0.15, 31)
    pck_values = np.zeros(len(thresholds))
    for i in range(len(thresholds)):
        pck_values[i] = (error < thresholds[i]).astype(np.float32)[mask].mean()

    auc = pck_values.mean() * 100

    return auc
