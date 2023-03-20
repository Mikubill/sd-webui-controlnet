# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import numpy as np
import torch
from annotator.mmpkg.mmcv.parallel import collate, scatter

from annotator.mmpkg.mmpose.core.bbox import bbox_xywh2cs, bbox_xywh2xyxy, bbox_xyxy2xywh
from annotator.mmpkg.mmpose.datasets.pipelines import Compose


def extract_pose_sequence(pose_results, frame_idx, causal, seq_len, step=1):
    """Extract the target frame from 2D pose results, and pad the sequence to a
    fixed length.

    Args:
        pose_results (list[list[dict]]): Multi-frame pose detection results
            stored in a nested list. Each element of the outer list is the
            pose detection results of a single frame, and each element of the
            inner list is the pose information of one person, which contains:

                - keypoints (ndarray[K, 2 or 3]): x, y, [score]
                - track_id (int): unique id of each person, required \
                    when ``with_track_id==True``.
                - bbox ((4, ) or (5, )): left, right, top, bottom, [score]

        frame_idx (int): The index of the frame in the original video.
        causal (bool): If True, the target frame is the last frame in
            a sequence. Otherwise, the target frame is in the middle of
            a sequence.
        seq_len (int): The number of frames in the input sequence.
        step (int): Step size to extract frames from the video.

    Returns:
        list[list[dict]]: Multi-frame pose detection results stored \
            in a nested list with a length of seq_len.
    """

    if causal:
        frames_left = seq_len - 1
        frames_right = 0
    else:
        frames_left = (seq_len - 1) // 2
        frames_right = frames_left
    num_frames = len(pose_results)

    # get the padded sequence
    pad_left = max(0, frames_left - frame_idx // step)
    pad_right = max(0, frames_right - (num_frames - 1 - frame_idx) // step)
    start = max(frame_idx % step, frame_idx - frames_left * step)
    end = min(num_frames - (num_frames - 1 - frame_idx) % step,
              frame_idx + frames_right * step + 1)
    pose_results_seq = [pose_results[0]] * pad_left + \
        pose_results[start:end:step] + [pose_results[-1]] * pad_right
    return pose_results_seq


def _gather_pose_lifter_inputs(pose_results,
                               bbox_center,
                               bbox_scale,
                               norm_pose_2d=False):
    """Gather input data (keypoints and track_id) for pose lifter model.

    Note:
        - The temporal length of the pose detection results: T
        - The number of the person instances: N
        - The number of the keypoints: K
        - The channel number of each keypoint: C

    Args:
        pose_results (List[List[Dict]]): Multi-frame pose detection results
            stored in a nested list. Each element of the outer list is the
            pose detection results of a single frame, and each element of the
            inner list is the pose information of one person, which contains:

                - keypoints (ndarray[K, 2 or 3]): x, y, [score]
                - track_id (int): unique id of each person, required when
                    ``with_track_id==True```
                - bbox ((4, ) or (5, )): left, right, top, bottom, [score]

        bbox_center (ndarray[1, 2], optional): x, y. The average center
            coordinate of the bboxes in the dataset. `bbox_center` will be
            used only when `norm_pose_2d` is `True`.
        bbox_scale (int|float, optional): The average scale of the bboxes
            in the dataset.
            `bbox_scale` will be used only when `norm_pose_2d` is `True`.
        norm_pose_2d (bool): If True, scale the bbox (along with the 2D
            pose) to bbox_scale, and move the bbox (along with the 2D pose) to
            bbox_center. Default: False.

    Returns:
        list[list[dict]]: Multi-frame pose detection results
            stored in a nested list. Each element of the outer list is the
            pose detection results of a single frame, and each element of the
            inner list is the pose information of one person, which contains:

                - keypoints (ndarray[K, 2 or 3]): x, y, [score]
                - track_id (int): unique id of each person, required when
                    ``with_track_id==True``
    """
    sequence_inputs = []
    for frame in pose_results:
        frame_inputs = []
        for res in frame:
            inputs = dict()

            if norm_pose_2d:
                bbox = res['bbox']
                center = np.array([[(bbox[0] + bbox[2]) / 2,
                                    (bbox[1] + bbox[3]) / 2]])
                scale = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
                inputs['keypoints'] = (res['keypoints'][:, :2] - center) \
                    / scale * bbox_scale + bbox_center
            else:
                inputs['keypoints'] = res['keypoints'][:, :2]

            if res['keypoints'].shape[1] == 3:
                inputs['keypoints'] = np.concatenate(
                    [inputs['keypoints'], res['keypoints'][:, 2:]], axis=1)

            if 'track_id' in res:
                inputs['track_id'] = res['track_id']
            frame_inputs.append(inputs)
        sequence_inputs.append(frame_inputs)
    return sequence_inputs


def _collate_pose_sequence(pose_results, with_track_id=True, target_frame=-1):
    """Reorganize multi-frame pose detection results into individual pose
    sequences.

    Note:
        - The temporal length of the pose detection results: T
        - The number of the person instances: N
        - The number of the keypoints: K
        - The channel number of each keypoint: C

    Args:
        pose_results (List[List[Dict]]): Multi-frame pose detection results
            stored in a nested list. Each element of the outer list is the
            pose detection results of a single frame, and each element of the
            inner list is the pose information of one person, which contains:

                - keypoints (ndarray[K, 2 or 3]): x, y, [score]
                - track_id (int): unique id of each person, required when
                    ``with_track_id==True```

        with_track_id (bool): If True, the element in pose_results is expected
            to contain "track_id", which will be used to gather the pose
            sequence of a person from multiple frames. Otherwise, the pose
            results in each frame are expected to have a consistent number and
            order of identities. Default is True.
        target_frame (int): The index of the target frame. Default: -1.
    """
    T = len(pose_results)
    assert T > 0

    target_frame = (T + target_frame) % T  # convert negative index to positive

    N = len(pose_results[target_frame])  # use identities in the target frame
    if N == 0:
        return []

    K, C = pose_results[target_frame][0]['keypoints'].shape

    track_ids = None
    if with_track_id:
        track_ids = [res['track_id'] for res in pose_results[target_frame]]

    pose_sequences = []
    for idx in range(N):
        pose_seq = dict()
        # gather static information
        for k, v in pose_results[target_frame][idx].items():
            if k != 'keypoints':
                pose_seq[k] = v
        # gather keypoints
        if not with_track_id:
            pose_seq['keypoints'] = np.stack(
                [frame[idx]['keypoints'] for frame in pose_results])
        else:
            keypoints = np.zeros((T, K, C), dtype=np.float32)
            keypoints[target_frame] = pose_results[target_frame][idx][
                'keypoints']
            # find the left most frame containing track_ids[idx]
            for frame_idx in range(target_frame - 1, -1, -1):
                contains_idx = False
                for res in pose_results[frame_idx]:
                    if res['track_id'] == track_ids[idx]:
                        keypoints[frame_idx] = res['keypoints']
                        contains_idx = True
                        break
                if not contains_idx:
                    # replicate the left most frame
                    keypoints[:frame_idx + 1] = keypoints[frame_idx + 1]
                    break
            # find the right most frame containing track_idx[idx]
            for frame_idx in range(target_frame + 1, T):
                contains_idx = False
                for res in pose_results[frame_idx]:
                    if res['track_id'] == track_ids[idx]:
                        keypoints[frame_idx] = res['keypoints']
                        contains_idx = True
                        break
                if not contains_idx:
                    # replicate the right most frame
                    keypoints[frame_idx + 1:] = keypoints[frame_idx]
                    break
            pose_seq['keypoints'] = keypoints
        pose_sequences.append(pose_seq)

    return pose_sequences


def inference_pose_lifter_model(model,
                                pose_results_2d,
                                dataset=None,
                                dataset_info=None,
                                with_track_id=True,
                                image_size=None,
                                norm_pose_2d=False):
    """Inference 3D pose from 2D pose sequences using a pose lifter model.

    Args:
        model (nn.Module): The loaded pose lifter model
        pose_results_2d (list[list[dict]]): The 2D pose sequences stored in a
            nested list. Each element of the outer list is the 2D pose results
            of a single frame, and each element of the inner list is the 2D
            pose of one person, which contains:

            - "keypoints" (ndarray[K, 2 or 3]): x, y, [score]
            - "track_id" (int)
        dataset (str): Dataset name, e.g. 'Body3DH36MDataset'
        with_track_id: If True, the element in pose_results_2d is expected to
            contain "track_id", which will be used to gather the pose sequence
            of a person from multiple frames. Otherwise, the pose results in
            each frame are expected to have a consistent number and order of
            identities. Default is True.
        image_size (tuple|list): image width, image height. If None, image size
            will not be contained in dict ``data``.
        norm_pose_2d (bool): If True, scale the bbox (along with the 2D
            pose) to the average bbox scale of the dataset, and move the bbox
            (along with the 2D pose) to the average bbox center of the dataset.

    Returns:
        list[dict]: 3D pose inference results. Each element is the result of \
            an instance, which contains:

            - "keypoints_3d" (ndarray[K, 3]): predicted 3D keypoints
            - "keypoints" (ndarray[K, 2 or 3]): from the last frame in \
                ``pose_results_2d``.
            - "track_id" (int): from the last frame in ``pose_results_2d``. \
                If there is no valid instance, an empty list will be \
                returned.
    """
    cfg = model.cfg
    test_pipeline = Compose(cfg.test_pipeline)

    device = next(model.parameters()).device
    if device.type == 'cpu':
        device = -1

    if dataset_info is not None:
        flip_pairs = dataset_info.flip_pairs
        if 'stats_info' in dataset_info._dataset_info:
            bbox_center = dataset_info._dataset_info['stats_info'][
                'bbox_center']
            bbox_scale = dataset_info._dataset_info['stats_info']['bbox_scale']
        else:
            bbox_center = None
            bbox_scale = None
    else:
        warnings.warn(
            'dataset is deprecated.'
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
        # TODO: These will be removed in the later versions.
        if dataset == 'Body3DH36MDataset':
            flip_pairs = [[1, 4], [2, 5], [3, 6], [11, 14], [12, 15], [13, 16]]
            bbox_center = np.array([[528, 427]], dtype=np.float32)
            bbox_scale = 400
        else:
            raise NotImplementedError()

    target_idx = -1 if model.causal else len(pose_results_2d) // 2
    pose_lifter_inputs = _gather_pose_lifter_inputs(pose_results_2d,
                                                    bbox_center, bbox_scale,
                                                    norm_pose_2d)
    pose_sequences_2d = _collate_pose_sequence(pose_lifter_inputs,
                                               with_track_id, target_idx)

    if not pose_sequences_2d:
        return []

    batch_data = []
    for seq in pose_sequences_2d:
        pose_2d = seq['keypoints'].astype(np.float32)
        T, K, C = pose_2d.shape

        input_2d = pose_2d[..., :2]
        input_2d_visible = pose_2d[..., 2:3]
        if C > 2:
            input_2d_visible = pose_2d[..., 2:3]
        else:
            input_2d_visible = np.ones((T, K, 1), dtype=np.float32)

        # TODO: Will be removed in the later versions
        # Dummy 3D input
        # This is for compatibility with configs in mmpose<=v0.14.0, where a
        # 3D input is required to generate denormalization parameters. This
        # part will be removed in the future.
        target = np.zeros((K, 3), dtype=np.float32)
        target_visible = np.ones((K, 1), dtype=np.float32)

        # Dummy image path
        # This is for compatibility with configs in mmpose<=v0.14.0, where
        # target_image_path is required. This part will be removed in the
        # future.
        target_image_path = None

        data = {
            'input_2d': input_2d,
            'input_2d_visible': input_2d_visible,
            'target': target,
            'target_visible': target_visible,
            'target_image_path': target_image_path,
            'ann_info': {
                'num_joints': K,
                'flip_pairs': flip_pairs
            }
        }

        if image_size is not None:
            assert len(image_size) == 2
            data['image_width'] = image_size[0]
            data['image_height'] = image_size[1]

        data = test_pipeline(data)
        batch_data.append(data)

    batch_data = collate(batch_data, samples_per_gpu=len(batch_data))
    batch_data = scatter(batch_data, target_gpus=[device])[0]

    with torch.no_grad():
        result = model(
            input=batch_data['input'],
            metas=batch_data['metas'],
            return_loss=False)

    poses_3d = result['preds']
    if poses_3d.shape[-1] != 4:
        assert poses_3d.shape[-1] == 3
        dummy_score = np.ones(
            poses_3d.shape[:-1] + (1, ), dtype=poses_3d.dtype)
        poses_3d = np.concatenate((poses_3d, dummy_score), axis=-1)
    pose_results = []
    for pose_2d, pose_3d in zip(pose_sequences_2d, poses_3d):
        pose_result = pose_2d.copy()
        pose_result['keypoints_3d'] = pose_3d
        pose_results.append(pose_result)

    return pose_results


def vis_3d_pose_result(model,
                       result,
                       img=None,
                       dataset='Body3DH36MDataset',
                       dataset_info=None,
                       kpt_score_thr=0.3,
                       radius=8,
                       thickness=2,
                       vis_height=400,
                       num_instances=-1,
                       axis_azimuth=70,
                       show=False,
                       out_file=None):
    """Visualize the 3D pose estimation results.

    Args:
        model (nn.Module): The loaded model.
        result (list[dict])
    """

    if dataset_info is not None:
        skeleton = dataset_info.skeleton
        pose_kpt_color = dataset_info.pose_kpt_color
        pose_link_color = dataset_info.pose_link_color
    else:
        warnings.warn(
            'dataset is deprecated.'
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
        # TODO: These will be removed in the later versions.
        palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                            [230, 230, 0], [255, 153, 255], [153, 204, 255],
                            [255, 102, 255], [255, 51, 255], [102, 178, 255],
                            [51, 153, 255], [255, 153, 153], [255, 102, 102],
                            [255, 51, 51], [153, 255, 153], [102, 255, 102],
                            [51, 255, 51], [0, 255, 0], [0, 0, 255],
                            [255, 0, 0], [255, 255, 255]])

        if dataset == 'Body3DH36MDataset':
            skeleton = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7],
                        [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13],
                        [8, 14], [14, 15], [15, 16]]

            pose_kpt_color = palette[[
                9, 0, 0, 0, 16, 16, 16, 9, 9, 9, 9, 16, 16, 16, 0, 0, 0
            ]]
            pose_link_color = palette[[
                0, 0, 0, 16, 16, 16, 9, 9, 9, 9, 16, 16, 16, 0, 0, 0
            ]]

        elif dataset == 'InterHand3DDataset':
            skeleton = [[0, 1], [1, 2], [2, 3], [3, 20], [4, 5], [5, 6],
                        [6, 7], [7, 20], [8, 9], [9, 10], [10, 11], [11, 20],
                        [12, 13], [13, 14], [14, 15], [15, 20], [16, 17],
                        [17, 18], [18, 19], [19, 20], [21, 22], [22, 23],
                        [23, 24], [24, 41], [25, 26], [26, 27], [27, 28],
                        [28, 41], [29, 30], [30, 31], [31, 32], [32, 41],
                        [33, 34], [34, 35], [35, 36], [36, 41], [37, 38],
                        [38, 39], [39, 40], [40, 41]]

            pose_kpt_color = [[14, 128, 250], [14, 128, 250], [14, 128, 250],
                              [14, 128, 250], [80, 127, 255], [80, 127, 255],
                              [80, 127, 255], [80, 127, 255], [71, 99, 255],
                              [71, 99, 255], [71, 99, 255], [71, 99, 255],
                              [0, 36, 255], [0, 36, 255], [0, 36, 255],
                              [0, 36, 255], [0, 0, 230], [0, 0, 230],
                              [0, 0, 230], [0, 0, 230], [0, 0, 139],
                              [237, 149, 100], [237, 149, 100],
                              [237, 149, 100], [237, 149, 100], [230, 128, 77],
                              [230, 128, 77], [230, 128, 77], [230, 128, 77],
                              [255, 144, 30], [255, 144, 30], [255, 144, 30],
                              [255, 144, 30], [153, 51, 0], [153, 51, 0],
                              [153, 51, 0], [153, 51, 0], [255, 51, 13],
                              [255, 51, 13], [255, 51, 13], [255, 51, 13],
                              [103, 37, 8]]

            pose_link_color = [[14, 128, 250], [14, 128, 250], [14, 128, 250],
                               [14, 128, 250], [80, 127, 255], [80, 127, 255],
                               [80, 127, 255], [80, 127, 255], [71, 99, 255],
                               [71, 99, 255], [71, 99, 255], [71, 99, 255],
                               [0, 36, 255], [0, 36, 255], [0, 36, 255],
                               [0, 36, 255], [0, 0, 230], [0, 0, 230],
                               [0, 0, 230], [0, 0, 230], [237, 149, 100],
                               [237, 149, 100], [237, 149, 100],
                               [237, 149, 100], [230, 128, 77], [230, 128, 77],
                               [230, 128, 77], [230, 128, 77], [255, 144, 30],
                               [255, 144, 30], [255, 144, 30], [255, 144, 30],
                               [153, 51, 0], [153, 51, 0], [153, 51, 0],
                               [153, 51, 0], [255, 51, 13], [255, 51, 13],
                               [255, 51, 13], [255, 51, 13]]
        else:
            raise NotImplementedError

    if hasattr(model, 'module'):
        model = model.module

    img = model.show_result(
        result,
        img,
        skeleton,
        radius=radius,
        thickness=thickness,
        pose_kpt_color=pose_kpt_color,
        pose_link_color=pose_link_color,
        vis_height=vis_height,
        num_instances=num_instances,
        axis_azimuth=axis_azimuth,
        show=show,
        out_file=out_file)

    return img


def inference_interhand_3d_model(model,
                                 img_or_path,
                                 det_results,
                                 bbox_thr=None,
                                 format='xywh',
                                 dataset='InterHand3DDataset'):
    """Inference a single image with a list of hand bounding boxes.

    Note:
        - num_bboxes: N
        - num_keypoints: K

    Args:
        model (nn.Module): The loaded pose model.
        img_or_path (str | np.ndarray): Image filename or loaded image.
        det_results (list[dict]): The 2D bbox sequences stored in a list.
            Each each element of the list is the bbox of one person, whose
            shape is (ndarray[4 or 5]), containing 4 box coordinates
            (and score).
        dataset (str): Dataset name.
        format: bbox format ('xyxy' | 'xywh'). Default: 'xywh'.
            'xyxy' means (left, top, right, bottom),
            'xywh' means (left, top, width, height).

    Returns:
        list[dict]: 3D pose inference results. Each element is the result \
            of an instance, which contains the predicted 3D keypoints with \
            shape (ndarray[K,3]). If there is no valid instance, an \
            empty list will be returned.
    """

    assert format in ['xyxy', 'xywh']

    pose_results = []

    if len(det_results) == 0:
        return pose_results

    # Change for-loop preprocess each bbox to preprocess all bboxes at once.
    bboxes = np.array([box['bbox'] for box in det_results])

    # Select bboxes by score threshold
    if bbox_thr is not None:
        assert bboxes.shape[1] == 5
        valid_idx = np.where(bboxes[:, 4] > bbox_thr)[0]
        bboxes = bboxes[valid_idx]
        det_results = [det_results[i] for i in valid_idx]

    if format == 'xyxy':
        bboxes_xyxy = bboxes
        bboxes_xywh = bbox_xyxy2xywh(bboxes)
    else:
        # format is already 'xywh'
        bboxes_xywh = bboxes
        bboxes_xyxy = bbox_xywh2xyxy(bboxes)

    # if bbox_thr remove all bounding box
    if len(bboxes_xywh) == 0:
        return []

    cfg = model.cfg
    device = next(model.parameters()).device
    if device.type == 'cpu':
        device = -1

    # build the data pipeline
    test_pipeline = Compose(cfg.test_pipeline)

    assert len(bboxes[0]) in [4, 5]

    if dataset == 'InterHand3DDataset':
        flip_pairs = [[i, 21 + i] for i in range(21)]
    else:
        raise NotImplementedError()

    batch_data = []
    for bbox in bboxes:
        image_size = cfg.data_cfg.image_size
        aspect_ratio = image_size[0] / image_size[1]  # w over h
        center, scale = bbox_xywh2cs(bbox, aspect_ratio, padding=1.25)

        # prepare data
        data = {
            'center':
            center,
            'scale':
            scale,
            'bbox_score':
            bbox[4] if len(bbox) == 5 else 1,
            'bbox_id':
            0,  # need to be assigned if batch_size > 1
            'dataset':
            dataset,
            'joints_3d':
            np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
            'joints_3d_visible':
            np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
            'rotation':
            0,
            'ann_info': {
                'image_size': np.array(cfg.data_cfg['image_size']),
                'num_joints': cfg.data_cfg['num_joints'],
                'flip_pairs': flip_pairs,
                'heatmap3d_depth_bound': cfg.data_cfg['heatmap3d_depth_bound'],
                'heatmap_size_root': cfg.data_cfg['heatmap_size_root'],
                'root_depth_bound': cfg.data_cfg['root_depth_bound']
            }
        }

        if isinstance(img_or_path, np.ndarray):
            data['img'] = img_or_path
        else:
            data['image_file'] = img_or_path

        data = test_pipeline(data)
        batch_data.append(data)

    batch_data = collate(batch_data, samples_per_gpu=len(batch_data))
    batch_data = scatter(batch_data, [device])[0]

    # forward the model
    with torch.no_grad():
        result = model(
            img=batch_data['img'],
            img_metas=batch_data['img_metas'],
            return_loss=False)

    poses_3d = result['preds']
    rel_root_depth = result['rel_root_depth']
    hand_type = result['hand_type']
    if poses_3d.shape[-1] != 4:
        assert poses_3d.shape[-1] == 3
        dummy_score = np.ones(
            poses_3d.shape[:-1] + (1, ), dtype=poses_3d.dtype)
        poses_3d = np.concatenate((poses_3d, dummy_score), axis=-1)

    # add relative root depth to left hand joints
    poses_3d[:, 21:, 2] += rel_root_depth

    # set joint scores according to hand type
    poses_3d[:, :21, 3] *= hand_type[:, [0]]
    poses_3d[:, 21:, 3] *= hand_type[:, [1]]

    pose_results = []
    for pose_3d, person_res, bbox_xyxy in zip(poses_3d, det_results,
                                              bboxes_xyxy):
        pose_res = person_res.copy()
        pose_res['keypoints_3d'] = pose_3d
        pose_res['bbox'] = bbox_xyxy
        pose_results.append(pose_res)

    return pose_results


def inference_mesh_model(model,
                         img_or_path,
                         det_results,
                         bbox_thr=None,
                         format='xywh',
                         dataset='MeshH36MDataset'):
    """Inference a single image with a list of bounding boxes.

    Note:
        - num_bboxes: N
        - num_keypoints: K
        - num_vertices: V
        - num_faces: F

    Args:
        model (nn.Module): The loaded pose model.
        img_or_path (str | np.ndarray): Image filename or loaded image.
        det_results (list[dict]): The 2D bbox sequences stored in a list.
            Each element of the list is the bbox of one person.
            "bbox" (ndarray[4 or 5]): The person bounding box,
            which contains 4 box coordinates (and score).
        bbox_thr (float | None): Threshold for bounding boxes.
            Only bboxes with higher scores will be fed into the pose
            detector. If bbox_thr is None, all boxes will be used.
        format (str): bbox format ('xyxy' | 'xywh'). Default: 'xywh'.

            - 'xyxy' means (left, top, right, bottom),
            - 'xywh' means (left, top, width, height).
        dataset (str): Dataset name.

    Returns:
        list[dict]: 3D pose inference results. Each element \
            is the result of an instance, which contains:

            - 'bbox' (ndarray[4]): instance bounding bbox
            - 'center' (ndarray[2]): bbox center
            - 'scale' (ndarray[2]): bbox scale
            - 'keypoints_3d' (ndarray[K,3]): predicted 3D keypoints
            - 'camera' (ndarray[3]): camera parameters
            - 'vertices' (ndarray[V, 3]): predicted 3D vertices
            - 'faces' (ndarray[F, 3]): mesh faces

            If there is no valid instance, an empty list
            will be returned.
    """

    assert format in ['xyxy', 'xywh']

    pose_results = []

    if len(det_results) == 0:
        return pose_results

    # Change for-loop preprocess each bbox to preprocess all bboxes at once.
    bboxes = np.array([box['bbox'] for box in det_results])

    # Select bboxes by score threshold
    if bbox_thr is not None:
        assert bboxes.shape[1] == 5
        valid_idx = np.where(bboxes[:, 4] > bbox_thr)[0]
        bboxes = bboxes[valid_idx]
        det_results = [det_results[i] for i in valid_idx]

    if format == 'xyxy':
        bboxes_xyxy = bboxes
        bboxes_xywh = bbox_xyxy2xywh(bboxes)
    else:
        # format is already 'xywh'
        bboxes_xywh = bboxes
        bboxes_xyxy = bbox_xywh2xyxy(bboxes)

    # if bbox_thr remove all bounding box
    if len(bboxes_xywh) == 0:
        return []

    cfg = model.cfg
    device = next(model.parameters()).device
    if device.type == 'cpu':
        device = -1

    # build the data pipeline
    test_pipeline = Compose(cfg.test_pipeline)

    assert len(bboxes[0]) in [4, 5]

    if dataset == 'MeshH36MDataset':
        flip_pairs = [[0, 5], [1, 4], [2, 3], [6, 11], [7, 10], [8, 9],
                      [20, 21], [22, 23]]
    else:
        raise NotImplementedError()

    batch_data = []
    for bbox in bboxes_xywh:
        image_size = cfg.data_cfg.image_size
        aspect_ratio = image_size[0] / image_size[1]  # w over h
        center, scale = bbox_xywh2cs(bbox, aspect_ratio, padding=1.25)

        # prepare data
        data = {
            'image_file':
            img_or_path,
            'center':
            center,
            'scale':
            scale,
            'rotation':
            0,
            'bbox_score':
            bbox[4] if len(bbox) == 5 else 1,
            'dataset':
            dataset,
            'joints_2d':
            np.zeros((cfg.data_cfg.num_joints, 2), dtype=np.float32),
            'joints_2d_visible':
            np.zeros((cfg.data_cfg.num_joints, 1), dtype=np.float32),
            'joints_3d':
            np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
            'joints_3d_visible':
            np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
            'pose':
            np.zeros(72, dtype=np.float32),
            'beta':
            np.zeros(10, dtype=np.float32),
            'has_smpl':
            0,
            'ann_info': {
                'image_size': np.array(cfg.data_cfg['image_size']),
                'num_joints': cfg.data_cfg['num_joints'],
                'flip_pairs': flip_pairs,
            }
        }

        data = test_pipeline(data)
        batch_data.append(data)

    batch_data = collate(batch_data, samples_per_gpu=len(batch_data))
    batch_data = scatter(batch_data, target_gpus=[device])[0]

    # forward the model
    with torch.no_grad():
        preds = model(
            img=batch_data['img'],
            img_metas=batch_data['img_metas'],
            return_loss=False,
            return_vertices=True,
            return_faces=True)

    for idx in range(len(det_results)):
        pose_res = det_results[idx].copy()
        pose_res['bbox'] = bboxes_xyxy[idx]
        pose_res['center'] = batch_data['img_metas'][idx]['center']
        pose_res['scale'] = batch_data['img_metas'][idx]['scale']
        pose_res['keypoints_3d'] = preds['keypoints_3d'][idx]
        pose_res['camera'] = preds['camera'][idx]
        pose_res['vertices'] = preds['vertices'][idx]
        pose_res['faces'] = preds['faces']
        pose_results.append(pose_res)
    return pose_results


def vis_3d_mesh_result(model, result, img=None, show=False, out_file=None):
    """Visualize the 3D mesh estimation results.

    Args:
        model (nn.Module): The loaded model.
        result (list[dict]): 3D mesh estimation results.
    """
    if hasattr(model, 'module'):
        model = model.module

    img = model.show_result(result, img, show=show, out_file=out_file)

    return img
