# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

from annotator.mmpkg.mmcv import Config

from annotator.mmpkg.mmpose.datasets.dataset_info import DatasetInfo


def get_eye_keypoint_ids(model_cfg: Config) -> Tuple[int, int]:
    """A helper function to get the keypoint indices of left and right eyes
    from the model config.

    Args:
        model_cfg (Config): mmpose model config

    Returns:
        tuple[int, int]: The keypoint indices of left eye and right eye.
    """
    left_eye_idx = None
    right_eye_idx = None

    # try obtaining eye point ids from dataset_info
    try:
        dataset_info = DatasetInfo(model_cfg.data.test.dataset_info)
        left_eye_idx = dataset_info.keypoint_name2id.get('left_eye', None)
        right_eye_idx = dataset_info.keypoint_name2id.get('right_eye', None)
    except AttributeError:
        left_eye_idx = None
        right_eye_idx = None

    if left_eye_idx is None or right_eye_idx is None:
        # Fall back to hard coded keypoint id
        dataset_name = model_cfg.data.test.type
        if dataset_name in {
                'TopDownCocoDataset', 'TopDownCocoWholeBodyDataset'
        }:
            left_eye_idx = 1
            right_eye_idx = 2
        elif dataset_name in {'AnimalPoseDataset', 'AnimalAP10KDataset'}:
            left_eye_idx = 0
            right_eye_idx = 1
        else:
            raise ValueError('Can not determine the eye keypoint id of '
                             f'{dataset_name}')

    return left_eye_idx, right_eye_idx


def get_face_keypoint_ids(model_cfg: Config) -> List:
    """A helper function to get the keypoint indices of the face from the model
    config.

    Args:
        model_cfg (Config): pose model config.

    Returns:
        list[int]: face keypoint indices. The length depends on the dataset.
    """
    face_indices = []

    # try obtaining nose point ids from dataset_info
    try:
        dataset_info = DatasetInfo(model_cfg.data.test.dataset_info)
        for id in range(68):
            face_indices.append(
                dataset_info.keypoint_name2id.get(f'face_{id}', None))
    except AttributeError:
        face_indices = []

    if not face_indices:
        # Fall back to hard coded keypoint id
        dataset_name = model_cfg.data.test.type
        if dataset_name in {'TopDownCocoWholeBodyDataset'}:
            face_indices = list(range(23, 91))
        else:
            raise ValueError('Can not determine the face id of '
                             f'{dataset_name}')

    return face_indices


def get_wrist_keypoint_ids(model_cfg: Config) -> Tuple[int, int]:
    """A helper function to get the keypoint indices of left and right wrists
    from the model config.

    Args:
        model_cfg (Config): pose model config.
    Returns:
        tuple[int, int]: The keypoint indices of left and right wrists.
    """

    # try obtaining eye point ids from dataset_info
    try:
        dataset_info = DatasetInfo(model_cfg.data.test.dataset_info)
        left_wrist_idx = dataset_info.keypoint_name2id.get('left_wrist', None)
        right_wrist_idx = dataset_info.keypoint_name2id.get(
            'right_wrist', None)
    except AttributeError:
        left_wrist_idx = None
        right_wrist_idx = None

    if left_wrist_idx is None or right_wrist_idx is None:
        # Fall back to hard coded keypoint id
        dataset_name = model_cfg.data.test.type
        if dataset_name in {
                'TopDownCocoDataset', 'TopDownCocoWholeBodyDataset'
        }:
            left_wrist_idx = 9
            right_wrist_idx = 10
        elif dataset_name == 'AnimalPoseDataset':
            left_wrist_idx = 16
            right_wrist_idx = 17
        elif dataset_name == 'AnimalAP10KDataset':
            left_wrist_idx = 7
            right_wrist_idx = 10
        else:
            raise ValueError('Can not determine the eye keypoint id of '
                             f'{dataset_name}')

    return left_wrist_idx, right_wrist_idx


def get_mouth_keypoint_ids(model_cfg: Config) -> int:
    """A helper function to get the mouth keypoint index from the model config.

    Args:
        model_cfg (Config): pose model config.
    Returns:
        int: The mouth keypoint index
    """
    # try obtaining mouth point ids from dataset_info
    try:
        dataset_info = DatasetInfo(model_cfg.data.test.dataset_info)
        mouth_index = dataset_info.keypoint_name2id.get('face-62', None)
    except AttributeError:
        mouth_index = None

    if mouth_index is None:
        # Fall back to hard coded keypoint id
        dataset_name = model_cfg.data.test.type
        if dataset_name == 'TopDownCocoWholeBodyDataset':
            mouth_index = 85
        else:
            raise ValueError('Can not determine the eye keypoint id of '
                             f'{dataset_name}')

    return mouth_index


def get_hand_keypoint_ids(model_cfg: Config) -> List[int]:
    """A helper function to get the keypoint indices of left and right hand
    from the model config.

    Args:
        model_cfg (Config): pose model config.
    Returns:
        list[int]: hand keypoint indices. The length depends on the dataset.
    """
    # try obtaining hand keypoint ids from dataset_info
    try:
        hand_indices = []
        dataset_info = DatasetInfo(model_cfg.data.test.dataset_info)

        hand_indices.append(
            dataset_info.keypoint_name2id.get('left_hand_root', None))

        for id in range(1, 5):
            hand_indices.append(
                dataset_info.keypoint_name2id.get(f'left_thumb{id}', None))
        for id in range(1, 5):
            hand_indices.append(
                dataset_info.keypoint_name2id.get(f'left_forefinger{id}',
                                                  None))
        for id in range(1, 5):
            hand_indices.append(
                dataset_info.keypoint_name2id.get(f'left_middle_finger{id}',
                                                  None))
        for id in range(1, 5):
            hand_indices.append(
                dataset_info.keypoint_name2id.get(f'left_ring_finger{id}',
                                                  None))
        for id in range(1, 5):
            hand_indices.append(
                dataset_info.keypoint_name2id.get(f'left_pinky_finger{id}',
                                                  None))

        hand_indices.append(
            dataset_info.keypoint_name2id.get('right_hand_root', None))

        for id in range(1, 5):
            hand_indices.append(
                dataset_info.keypoint_name2id.get(f'right_thumb{id}', None))
        for id in range(1, 5):
            hand_indices.append(
                dataset_info.keypoint_name2id.get(f'right_forefinger{id}',
                                                  None))
        for id in range(1, 5):
            hand_indices.append(
                dataset_info.keypoint_name2id.get(f'right_middle_finger{id}',
                                                  None))
        for id in range(1, 5):
            hand_indices.append(
                dataset_info.keypoint_name2id.get(f'right_ring_finger{id}',
                                                  None))
        for id in range(1, 5):
            hand_indices.append(
                dataset_info.keypoint_name2id.get(f'right_pinky_finger{id}',
                                                  None))

    except AttributeError:
        hand_indices = None

    if hand_indices is None:
        # Fall back to hard coded keypoint id
        dataset_name = model_cfg.data.test.type
        if dataset_name in {'TopDownCocoWholeBodyDataset'}:
            hand_indices = list(range(91, 133))
        else:
            raise ValueError('Can not determine the hand id of '
                             f'{dataset_name}')

    return hand_indices
