# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from functools import partial

import numpy as np

from annotator.mmpkg.mmpose.core import OneEuroFilter, oks_iou


def _compute_iou(bboxA, bboxB):
    """Compute the Intersection over Union (IoU) between two boxes .

    Args:
        bboxA (list): The first bbox info (left, top, right, bottom, score).
        bboxB (list): The second bbox info (left, top, right, bottom, score).

    Returns:
        float: The IoU value.
    """

    x1 = max(bboxA[0], bboxB[0])
    y1 = max(bboxA[1], bboxB[1])
    x2 = min(bboxA[2], bboxB[2])
    y2 = min(bboxA[3], bboxB[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    bboxA_area = (bboxA[2] - bboxA[0]) * (bboxA[3] - bboxA[1])
    bboxB_area = (bboxB[2] - bboxB[0]) * (bboxB[3] - bboxB[1])
    union_area = float(bboxA_area + bboxB_area - inter_area)
    if union_area == 0:
        union_area = 1e-5
        warnings.warn('union_area=0 is unexpected')

    iou = inter_area / union_area

    return iou


def _track_by_iou(res, results_last, thr):
    """Get track id using IoU tracking greedily.

    Args:
        res (dict): The bbox & pose results of the person instance.
        results_last (list[dict]): The bbox & pose & track_id info of the
            last frame (bbox_result, pose_result, track_id).
        thr (float): The threshold for iou tracking.

    Returns:
        int: The track id for the new person instance.
        list[dict]: The bbox & pose & track_id info of the persons
            that have not been matched on the last frame.
        dict: The matched person instance on the last frame.
    """

    bbox = list(res['bbox'])

    max_iou_score = -1
    max_index = -1
    match_result = {}
    for index, res_last in enumerate(results_last):
        bbox_last = list(res_last['bbox'])

        iou_score = _compute_iou(bbox, bbox_last)
        if iou_score > max_iou_score:
            max_iou_score = iou_score
            max_index = index

    if max_iou_score > thr:
        track_id = results_last[max_index]['track_id']
        match_result = results_last[max_index]
        del results_last[max_index]
    else:
        track_id = -1

    return track_id, results_last, match_result


def _track_by_oks(res, results_last, thr, sigmas):
    """Get track id using OKS tracking greedily.

    Args:
        res (dict): The pose results of the person instance.
        results_last (list[dict]): The pose & track_id info of the
            last frame (pose_result, track_id).
        thr (float): The threshold for oks tracking.
        sigmas (np.ndarray): standard deviation of keypoint labelling.

    Returns:
        int: The track id for the new person instance.
        list[dict]: The pose & track_id info of the persons
            that have not been matched on the last frame.
        dict: The matched person instance on the last frame.
    """
    pose = res['keypoints'].reshape((-1))
    area = res['area']
    max_index = -1
    match_result = {}

    if len(results_last) == 0:
        return -1, results_last, match_result

    pose_last = np.array(
        [res_last['keypoints'].reshape((-1)) for res_last in results_last])
    area_last = np.array([res_last['area'] for res_last in results_last])

    oks_score = oks_iou(pose, pose_last, area, area_last, sigmas=sigmas)

    max_index = np.argmax(oks_score)

    if oks_score[max_index] > thr:
        track_id = results_last[max_index]['track_id']
        match_result = results_last[max_index]
        del results_last[max_index]
    else:
        track_id = -1

    return track_id, results_last, match_result


def _get_area(results):
    """Get bbox for each person instance on the current frame.

    Args:
        results (list[dict]): The pose results of the current frame
            (pose_result).
    Returns:
        list[dict]: The bbox & pose info of the current frame
            (bbox_result, pose_result, area).
    """
    for result in results:
        if 'bbox' in result:
            result['area'] = ((result['bbox'][2] - result['bbox'][0]) *
                              (result['bbox'][3] - result['bbox'][1]))
        else:
            xmin = np.min(
                result['keypoints'][:, 0][result['keypoints'][:, 0] > 0],
                initial=1e10)
            xmax = np.max(result['keypoints'][:, 0])
            ymin = np.min(
                result['keypoints'][:, 1][result['keypoints'][:, 1] > 0],
                initial=1e10)
            ymax = np.max(result['keypoints'][:, 1])
            result['area'] = (xmax - xmin) * (ymax - ymin)
            result['bbox'] = np.array([xmin, ymin, xmax, ymax])
    return results


def _temporal_refine(result, match_result, fps=None):
    """Refine koypoints using tracked person instance on last frame.

    Args:
        results (dict): The pose results of the current frame
                (pose_result).
        match_result (dict): The pose results of the last frame
                (match_result)
    Returns:
        (array): The person keypoints after refine.
    """
    if 'one_euro' in match_result:
        result['keypoints'][:, :2] = match_result['one_euro'](
            result['keypoints'][:, :2])
        result['one_euro'] = match_result['one_euro']
    else:
        result['one_euro'] = OneEuroFilter(result['keypoints'][:, :2], fps=fps)
    return result['keypoints']


def get_track_id(results,
                 results_last,
                 next_id,
                 min_keypoints=3,
                 use_oks=False,
                 tracking_thr=0.3,
                 use_one_euro=False,
                 fps=None,
                 sigmas=None):
    """Get track id for each person instance on the current frame.

    Args:
        results (list[dict]): The bbox & pose results of the current frame
            (bbox_result, pose_result).
        results_last (list[dict], optional): The bbox & pose & track_id info
            of the last frame (bbox_result, pose_result, track_id). None is
            equivalent to an empty result list. Default: None
        next_id (int): The track id for the new person instance.
        min_keypoints (int): Minimum number of keypoints recognized as person.
            0 means no minimum threshold required. Default: 3.
        use_oks (bool): Flag to using oks tracking. default: False.
        tracking_thr (float): The threshold for tracking.
        use_one_euro (bool): Option to use one-euro-filter. default: False.
        fps (optional): Parameters that d_cutoff
            when one-euro-filter is used as a video input
        sigmas (np.ndarray): Standard deviation of keypoint labelling. It is
            necessary for oks_iou tracking (`use_oks==True`). It will be use
            sigmas of COCO as default if it is set to None. Default is None.

    Returns:
        tuple:
        - results (list[dict]): The bbox & pose & track_id info of the \
            current frame (bbox_result, pose_result, track_id).
        - next_id (int): The track id for the new person instance.
    """
    if use_one_euro:
        warnings.warn(
            'In the future, get_track_id() will no longer perform '
            'temporal refinement and the arguments `use_one_euro` and '
            '`fps` will be deprecated. This part of function has been '
            'migrated to Smoother (mmpose.core.Smoother). See '
            'demo/top_down_pose_trackign_demo_with_mmdet.py for an '
            'example.', DeprecationWarning)

    if results_last is None:
        results_last = []

    results = _get_area(results)

    if use_oks:
        _track = partial(_track_by_oks, sigmas=sigmas)
    else:
        _track = _track_by_iou

    for result in results:
        track_id, results_last, match_result = _track(result, results_last,
                                                      tracking_thr)
        if track_id == -1:
            if np.count_nonzero(result['keypoints'][:, 1]) >= min_keypoints:
                result['track_id'] = next_id
                next_id += 1
            else:
                # If the number of keypoints detected is small,
                # delete that person instance.
                result['keypoints'][:, 1] = -10
                result['bbox'] *= 0
                result['track_id'] = -1
        else:
            result['track_id'] = track_id

        if use_one_euro:
            result['keypoints'] = _temporal_refine(
                result, match_result, fps=fps)
        del match_result

    return results, next_id


def vis_pose_tracking_result(model,
                             img,
                             result,
                             radius=4,
                             thickness=1,
                             kpt_score_thr=0.3,
                             dataset='TopDownCocoDataset',
                             dataset_info=None,
                             show=False,
                             out_file=None):
    """Visualize the pose tracking results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str | np.ndarray): Image filename or loaded image.
        result (list[dict]): The results to draw over `img`
            (bbox_result, pose_result).
        radius (int): Radius of circles.
        thickness (int): Thickness of lines.
        kpt_score_thr (float): The threshold to visualize the keypoints.
        skeleton (list[tuple]): Default None.
        show (bool):  Whether to show the image. Default True.
        out_file (str|None): The filename of the output visualization image.
    """
    if hasattr(model, 'module'):
        model = model.module

    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])

    if dataset_info is None and dataset is not None:
        warnings.warn(
            'dataset is deprecated.'
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
        # TODO: These will be removed in the later versions.
        if dataset in ('TopDownCocoDataset', 'BottomUpCocoDataset',
                       'TopDownOCHumanDataset'):
            kpt_num = 17
            skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                        [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                        [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
                        [3, 5], [4, 6]]

        elif dataset == 'TopDownCocoWholeBodyDataset':
            kpt_num = 133
            skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                        [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                        [8, 10], [1, 2], [0, 1], [0, 2],
                        [1, 3], [2, 4], [3, 5], [4, 6], [15, 17], [15, 18],
                        [15, 19], [16, 20], [16, 21], [16, 22], [91, 92],
                        [92, 93], [93, 94], [94, 95], [91, 96], [96, 97],
                        [97, 98], [98, 99], [91, 100], [100, 101], [101, 102],
                        [102, 103], [91, 104], [104, 105], [105, 106],
                        [106, 107], [91, 108], [108, 109], [109, 110],
                        [110, 111], [112, 113], [113, 114], [114, 115],
                        [115, 116], [112, 117], [117, 118], [118, 119],
                        [119, 120], [112, 121], [121, 122], [122, 123],
                        [123, 124], [112, 125], [125, 126], [126, 127],
                        [127, 128], [112, 129], [129, 130], [130, 131],
                        [131, 132]]
            radius = 1

        elif dataset == 'TopDownAicDataset':
            kpt_num = 14
            skeleton = [[2, 1], [1, 0], [0, 13], [13, 3], [3, 4], [4, 5],
                        [8, 7], [7, 6], [6, 9], [9, 10], [10, 11], [12, 13],
                        [0, 6], [3, 9]]

        elif dataset == 'TopDownMpiiDataset':
            kpt_num = 16
            skeleton = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5], [6, 7],
                        [7, 8], [8, 9], [8, 12], [12, 11], [11, 10], [8, 13],
                        [13, 14], [14, 15]]

        elif dataset in ('OneHand10KDataset', 'FreiHandDataset',
                         'PanopticDataset'):
            kpt_num = 21
            skeleton = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7],
                        [7, 8], [0, 9], [9, 10], [10, 11], [11, 12], [0, 13],
                        [13, 14], [14, 15], [15, 16], [0, 17], [17, 18],
                        [18, 19], [19, 20]]

        elif dataset == 'InterHand2DDataset':
            kpt_num = 21
            skeleton = [[0, 1], [1, 2], [2, 3], [4, 5], [5, 6], [6, 7], [8, 9],
                        [9, 10], [10, 11], [12, 13], [13, 14], [14, 15],
                        [16, 17], [17, 18], [18, 19], [3, 20], [7, 20],
                        [11, 20], [15, 20], [19, 20]]

        else:
            raise NotImplementedError()

    elif dataset_info is not None:
        kpt_num = dataset_info.keypoint_num
        skeleton = dataset_info.skeleton

    for res in result:
        track_id = res['track_id']
        bbox_color = palette[track_id % len(palette)]
        pose_kpt_color = palette[[track_id % len(palette)] * kpt_num]
        pose_link_color = palette[[track_id % len(palette)] * len(skeleton)]
        img = model.show_result(
            img, [res],
            skeleton,
            radius=radius,
            thickness=thickness,
            pose_kpt_color=pose_kpt_color,
            pose_link_color=pose_link_color,
            bbox_color=tuple(bbox_color.tolist()),
            kpt_score_thr=kpt_score_thr,
            show=show,
            out_file=out_file)

    return img
