# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import warnings
from collections import defaultdict

import annotator.mmpkg.mmcv as mmcv
import numpy as np
import torch
from annotator.mmpkg.mmcv.parallel import collate, scatter
from annotator.mmpkg.mmcv.runner import load_checkpoint
from annotator.mmpkg.mmcv.utils.misc import deprecated_api_warning
from PIL import Image

from annotator.mmpkg.mmpose.core.bbox import bbox_xywh2xyxy, bbox_xyxy2xywh
from annotator.mmpkg.mmpose.core.post_processing import oks_nms
from annotator.mmpkg.mmpose.datasets.dataset_info import DatasetInfo
from annotator.mmpkg.mmpose.datasets.pipelines import Compose, ToTensor
from annotator.mmpkg.mmpose.models import build_posenet
from annotator.mmpkg.mmpose.utils.hooks import OutputHook

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def init_pose_model(config, checkpoint=None, device='cuda:0'):
    """Initialize a pose model from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    model = build_posenet(config.model)
    if checkpoint is not None:
        # load model checkpoint
        load_checkpoint(model, checkpoint, map_location='cpu')
    # save the config in the model for convenience
    model.cfg = config
    model.to(device)
    model.eval()
    return model


def _pipeline_gpu_speedup(pipeline, device):
    """Load images to GPU and speed up the data transforms in pipelines.

    Args:
        pipeline: A instance of `Compose`.
        device: A string or torch.device.

    Examples:
        _pipeline_gpu_speedup(test_pipeline, 'cuda:0')
    """

    for t in pipeline.transforms:
        if isinstance(t, ToTensor):
            t.device = device


def _inference_single_pose_model(model,
                                 imgs_or_paths,
                                 bboxes,
                                 dataset='TopDownCocoDataset',
                                 dataset_info=None,
                                 return_heatmap=False,
                                 use_multi_frames=False):
    """Inference human bounding boxes.

    Note:
        - num_frames: F
        - num_bboxes: N
        - num_keypoints: K

    Args:
        model (nn.Module): The loaded pose model.
        imgs_or_paths (list(str) | list(np.ndarray)): Image filename(s) or
            loaded image(s)
        bboxes (list | np.ndarray): All bounding boxes (with scores),
            shaped (N, 4) or (N, 5). (left, top, width, height, [score])
            where N is number of bounding boxes.
        dataset (str): Dataset name. Deprecated.
        dataset_info (DatasetInfo): A class containing all dataset info.
        return_heatmap (bool): Flag to return heatmap, default: False
        use_multi_frames (bool): Flag to use multi frames for inference

    Returns:
        ndarray[NxKx3]: Predicted pose x, y, score.
        heatmap[N, K, H, W]: Model output heatmap.
    """

    cfg = model.cfg
    device = next(model.parameters()).device
    if device.type == 'cpu':
        device = -1

    if use_multi_frames:
        assert 'frame_weight_test' in cfg.data.test.data_cfg
        # use multi frames for inference
        # the number of input frames must equal to frame weight in the config
        assert len(imgs_or_paths) == len(
            cfg.data.test.data_cfg.frame_weight_test)

    # build the data pipeline
    _test_pipeline = copy.deepcopy(cfg.test_pipeline)

    has_bbox_xywh2cs = False
    for transform in _test_pipeline:
        if transform['type'] == 'TopDownGetBboxCenterScale':
            has_bbox_xywh2cs = True
            break
    if not has_bbox_xywh2cs:
        _test_pipeline.insert(
            0, dict(type='TopDownGetBboxCenterScale', padding=1.25))
    test_pipeline = Compose(_test_pipeline)
    _pipeline_gpu_speedup(test_pipeline, next(model.parameters()).device)

    assert len(bboxes[0]) in [4, 5]

    if dataset_info is not None:
        dataset_name = dataset_info.dataset_name
        flip_pairs = dataset_info.flip_pairs
    else:
        warnings.warn(
            'dataset is deprecated.'
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
        # TODO: These will be removed in the later versions.
        if dataset in ('TopDownCocoDataset', 'TopDownOCHumanDataset',
                       'AnimalMacaqueDataset'):
            flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12],
                          [13, 14], [15, 16]]
        elif dataset == 'TopDownCocoWholeBodyDataset':
            body = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12],
                    [13, 14], [15, 16]]
            foot = [[17, 20], [18, 21], [19, 22]]

            face = [[23, 39], [24, 38], [25, 37], [26, 36], [27, 35], [28, 34],
                    [29, 33], [30, 32], [40, 49], [41, 48], [42, 47], [43, 46],
                    [44, 45], [54, 58], [55, 57], [59, 68], [60, 67], [61, 66],
                    [62, 65], [63, 70], [64, 69], [71, 77], [72, 76], [73, 75],
                    [78, 82], [79, 81], [83, 87], [84, 86], [88, 90]]

            hand = [[91, 112], [92, 113], [93, 114], [94, 115], [95, 116],
                    [96, 117], [97, 118], [98, 119], [99, 120], [100, 121],
                    [101, 122], [102, 123], [103, 124], [104, 125], [105, 126],
                    [106, 127], [107, 128], [108, 129], [109, 130], [110, 131],
                    [111, 132]]
            flip_pairs = body + foot + face + hand
        elif dataset == 'TopDownAicDataset':
            flip_pairs = [[0, 3], [1, 4], [2, 5], [6, 9], [7, 10], [8, 11]]
        elif dataset == 'TopDownMpiiDataset':
            flip_pairs = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
        elif dataset == 'TopDownMpiiTrbDataset':
            flip_pairs = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11],
                          [14, 15], [16, 22], [28, 34], [17, 23], [29, 35],
                          [18, 24], [30, 36], [19, 25], [31, 37], [20, 26],
                          [32, 38], [21, 27], [33, 39]]
        elif dataset in ('OneHand10KDataset', 'FreiHandDataset',
                         'PanopticDataset', 'InterHand2DDataset'):
            flip_pairs = []
        elif dataset in 'Face300WDataset':
            flip_pairs = [[0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11],
                          [6, 10], [7, 9], [17, 26], [18, 25], [19, 24],
                          [20, 23], [21, 22], [31, 35], [32, 34], [36, 45],
                          [37, 44], [38, 43], [39, 42], [40, 47], [41, 46],
                          [48, 54], [49, 53], [50, 52], [61, 63], [60, 64],
                          [67, 65], [58, 56], [59, 55]]

        elif dataset in 'FaceAFLWDataset':
            flip_pairs = [[0, 5], [1, 4], [2, 3], [6, 11], [7, 10], [8, 9],
                          [12, 14], [15, 17]]

        elif dataset in 'FaceCOFWDataset':
            flip_pairs = [[0, 1], [4, 6], [2, 3], [5, 7], [8, 9], [10, 11],
                          [12, 14], [16, 17], [13, 15], [18, 19], [22, 23]]

        elif dataset in 'FaceWFLWDataset':
            flip_pairs = [[0, 32], [1, 31], [2, 30], [3, 29], [4, 28], [5, 27],
                          [6, 26], [7, 25], [8, 24], [9, 23], [10, 22],
                          [11, 21], [12, 20], [13, 19], [14, 18], [15, 17],
                          [33, 46], [34, 45], [35, 44], [36, 43], [37, 42],
                          [38, 50], [39, 49], [40, 48], [41, 47], [60, 72],
                          [61, 71], [62, 70], [63, 69], [64, 68], [65, 75],
                          [66, 74], [67, 73], [55, 59], [56, 58], [76, 82],
                          [77, 81], [78, 80], [87, 83], [86, 84], [88, 92],
                          [89, 91], [95, 93], [96, 97]]

        elif dataset in 'AnimalFlyDataset':
            flip_pairs = [[1, 2], [6, 18], [7, 19], [8, 20], [9, 21], [10, 22],
                          [11, 23], [12, 24], [13, 25], [14, 26], [15, 27],
                          [16, 28], [17, 29], [30, 31]]
        elif dataset in 'AnimalHorse10Dataset':
            flip_pairs = []

        elif dataset in 'AnimalLocustDataset':
            flip_pairs = [[5, 20], [6, 21], [7, 22], [8, 23], [9, 24],
                          [10, 25], [11, 26], [12, 27], [13, 28], [14, 29],
                          [15, 30], [16, 31], [17, 32], [18, 33], [19, 34]]

        elif dataset in 'AnimalZebraDataset':
            flip_pairs = [[3, 4], [5, 6]]

        elif dataset in 'AnimalPoseDataset':
            flip_pairs = [[0, 1], [2, 3], [8, 9], [10, 11], [12, 13], [14, 15],
                          [16, 17], [18, 19]]
        else:
            raise NotImplementedError()
        dataset_name = dataset

    batch_data = []
    for bbox in bboxes:
        # prepare data
        data = {
            'bbox':
            bbox,
            'bbox_score':
            bbox[4] if len(bbox) == 5 else 1,
            'bbox_id':
            0,  # need to be assigned if batch_size > 1
            'dataset':
            dataset_name,
            'joints_3d':
            np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
            'joints_3d_visible':
            np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
            'rotation':
            0,
            'ann_info': {
                'image_size': np.array(cfg.data_cfg['image_size']),
                'num_joints': cfg.data_cfg['num_joints'],
                'flip_pairs': flip_pairs
            }
        }

        if use_multi_frames:
            # weight for different frames in multi-frame inference setting
            data['frame_weight'] = cfg.data.test.data_cfg.frame_weight_test
            if isinstance(imgs_or_paths[0], np.ndarray):
                data['img'] = imgs_or_paths
            else:
                data['image_file'] = imgs_or_paths
        else:
            if isinstance(imgs_or_paths, np.ndarray):
                data['img'] = imgs_or_paths
            else:
                data['image_file'] = imgs_or_paths

        data = test_pipeline(data)
        batch_data.append(data)

    batch_data = collate(batch_data, samples_per_gpu=len(batch_data))
    batch_data = scatter(batch_data, [device])[0]

    # forward the model
    with torch.no_grad():
        result = model(
            img=batch_data['img'],
            img_metas=batch_data['img_metas'],
            return_loss=False,
            return_heatmap=return_heatmap)

    return result['preds'], result['output_heatmap']


@deprecated_api_warning(name_dict=dict(img_or_path='imgs_or_paths'))
def inference_top_down_pose_model(model,
                                  imgs_or_paths,
                                  person_results=None,
                                  bbox_thr=None,
                                  format='xywh',
                                  dataset='TopDownCocoDataset',
                                  dataset_info=None,
                                  return_heatmap=False,
                                  outputs=None):
    """Inference a single image with a list of person bounding boxes. Support
    single-frame and multi-frame inference setting.

    Note:
        - num_frames: F
        - num_people: P
        - num_keypoints: K
        - bbox height: H
        - bbox width: W

    Args:
        model (nn.Module): The loaded pose model.
        imgs_or_paths (str | np.ndarray | list(str) | list(np.ndarray)):
            Image filename(s) or loaded image(s).
        person_results (list(dict), optional): a list of detected persons that
            contains ``bbox`` and/or ``track_id``:

            - ``bbox`` (4, ) or (5, ): The person bounding box, which contains
                4 box coordinates (and score).
            - ``track_id`` (int): The unique id for each human instance. If
                not provided, a dummy person result with a bbox covering
                the entire image will be used. Default: None.
        bbox_thr (float | None): Threshold for bounding boxes. Only bboxes
            with higher scores will be fed into the pose detector.
            If bbox_thr is None, all boxes will be used.
        format (str): bbox format ('xyxy' | 'xywh'). Default: 'xywh'.

            - `xyxy` means (left, top, right, bottom),
            - `xywh` means (left, top, width, height).
        dataset (str): Dataset name, e.g. 'TopDownCocoDataset'.
            It is deprecated. Please use dataset_info instead.
        dataset_info (DatasetInfo): A class containing all dataset info.
        return_heatmap (bool) : Flag to return heatmap, default: False
        outputs (list(str) | tuple(str)) : Names of layers whose outputs
            need to be returned. Default: None.

    Returns:
        tuple:
        - pose_results (list[dict]): The bbox & pose info. \
            Each item in the list is a dictionary, \
            containing the bbox: (left, top, right, bottom, [score]) \
            and the pose (ndarray[Kx3]): x, y, score.
        - returned_outputs (list[dict[np.ndarray[N, K, H, W] | \
            torch.Tensor[N, K, H, W]]]): \
            Output feature maps from layers specified in `outputs`. \
            Includes 'heatmap' if `return_heatmap` is True.
    """
    # decide whether to use multi frames for inference
    if isinstance(imgs_or_paths, (list, tuple)):
        use_multi_frames = True
    else:
        assert isinstance(imgs_or_paths, (str, np.ndarray))
        use_multi_frames = False
    # get dataset info
    if (dataset_info is None and hasattr(model, 'cfg')
            and 'dataset_info' in model.cfg):
        dataset_info = DatasetInfo(model.cfg.dataset_info)
    if dataset_info is None:
        warnings.warn(
            'dataset is deprecated.'
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663'
            ' for details.', DeprecationWarning)

    # only two kinds of bbox format is supported.
    assert format in ['xyxy', 'xywh']

    pose_results = []
    returned_outputs = []

    if person_results is None:
        # create dummy person results
        sample = imgs_or_paths[0] if use_multi_frames else imgs_or_paths
        if isinstance(sample, str):
            width, height = Image.open(sample).size
        else:
            height, width = sample.shape[:2]
        person_results = [{'bbox': np.array([0, 0, width, height])}]

    if len(person_results) == 0:
        return pose_results, returned_outputs

    # Change for-loop preprocess each bbox to preprocess all bboxes at once.
    bboxes = np.array([box['bbox'] for box in person_results])

    # Select bboxes by score threshold
    if bbox_thr is not None:
        assert bboxes.shape[1] == 5
        valid_idx = np.where(bboxes[:, 4] > bbox_thr)[0]
        bboxes = bboxes[valid_idx]
        person_results = [person_results[i] for i in valid_idx]

    if format == 'xyxy':
        bboxes_xyxy = bboxes
        bboxes_xywh = bbox_xyxy2xywh(bboxes)
    else:
        # format is already 'xywh'
        bboxes_xywh = bboxes
        bboxes_xyxy = bbox_xywh2xyxy(bboxes)

    # if bbox_thr remove all bounding box
    if len(bboxes_xywh) == 0:
        return [], []

    with OutputHook(model, outputs=outputs, as_tensor=False) as h:
        # poses is results['pred'] # N x 17x 3
        poses, heatmap = _inference_single_pose_model(
            model,
            imgs_or_paths,
            bboxes_xywh,
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            use_multi_frames=use_multi_frames)

        if return_heatmap:
            h.layer_outputs['heatmap'] = heatmap

        returned_outputs.append(h.layer_outputs)

    assert len(poses) == len(person_results), print(
        len(poses), len(person_results), len(bboxes_xyxy))
    for pose, person_result, bbox_xyxy in zip(poses, person_results,
                                              bboxes_xyxy):
        pose_result = person_result.copy()
        pose_result['keypoints'] = pose
        pose_result['bbox'] = bbox_xyxy
        pose_results.append(pose_result)

    return pose_results, returned_outputs


def inference_bottom_up_pose_model(model,
                                   img_or_path,
                                   dataset='BottomUpCocoDataset',
                                   dataset_info=None,
                                   pose_nms_thr=0.9,
                                   return_heatmap=False,
                                   outputs=None):
    """Inference a single image with a bottom-up pose model.

    Note:
        - num_people: P
        - num_keypoints: K
        - bbox height: H
        - bbox width: W

    Args:
        model (nn.Module): The loaded pose model.
        img_or_path (str| np.ndarray): Image filename or loaded image.
        dataset (str): Dataset name, e.g. 'BottomUpCocoDataset'.
            It is deprecated. Please use dataset_info instead.
        dataset_info (DatasetInfo): A class containing all dataset info.
        pose_nms_thr (float): retain oks overlap < pose_nms_thr, default: 0.9.
        return_heatmap (bool) : Flag to return heatmap, default: False.
        outputs (list(str) | tuple(str)) : Names of layers whose outputs
            need to be returned, default: None.

    Returns:
        tuple:
        - pose_results (list[np.ndarray]): The predicted pose info. \
            The length of the list is the number of people (P). \
            Each item in the list is a ndarray, containing each \
            person's pose (np.ndarray[Kx3]): x, y, score.
        - returned_outputs (list[dict[np.ndarray[N, K, H, W] | \
            torch.Tensor[N, K, H, W]]]): \
            Output feature maps from layers specified in `outputs`. \
            Includes 'heatmap' if `return_heatmap` is True.
    """
    # get dataset info
    if (dataset_info is None and hasattr(model, 'cfg')
            and 'dataset_info' in model.cfg):
        dataset_info = DatasetInfo(model.cfg.dataset_info)

    if dataset_info is not None:
        dataset_name = dataset_info.dataset_name
        flip_index = dataset_info.flip_index
        sigmas = getattr(dataset_info, 'sigmas', None)
        skeleton = getattr(dataset_info, 'skeleton', None)
    else:
        warnings.warn(
            'dataset is deprecated.'
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
        assert (dataset == 'BottomUpCocoDataset')
        dataset_name = dataset
        flip_index = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
        sigmas = None
        skeleton = None

    pose_results = []
    returned_outputs = []

    cfg = model.cfg
    device = next(model.parameters()).device
    if device.type == 'cpu':
        device = -1

    # build the data pipeline
    test_pipeline = Compose(cfg.test_pipeline)
    _pipeline_gpu_speedup(test_pipeline, next(model.parameters()).device)

    # prepare data
    data = {
        'dataset': dataset_name,
        'ann_info': {
            'image_size': np.array(cfg.data_cfg['image_size']),
            'heatmap_size': cfg.data_cfg.get('heatmap_size', None),
            'num_joints': cfg.data_cfg['num_joints'],
            'flip_index': flip_index,
            'skeleton': skeleton,
        }
    }
    if isinstance(img_or_path, np.ndarray):
        data['img'] = img_or_path
    else:
        data['image_file'] = img_or_path

    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    data = scatter(data, [device])[0]

    with OutputHook(model, outputs=outputs, as_tensor=False) as h:
        # forward the model
        with torch.no_grad():
            result = model(
                img=data['img'],
                img_metas=data['img_metas'],
                return_loss=False,
                return_heatmap=return_heatmap)

        if return_heatmap:
            h.layer_outputs['heatmap'] = result['output_heatmap']

        returned_outputs.append(h.layer_outputs)

        for idx, pred in enumerate(result['preds']):
            area = (np.max(pred[:, 0]) - np.min(pred[:, 0])) * (
                np.max(pred[:, 1]) - np.min(pred[:, 1]))
            pose_results.append({
                'keypoints': pred[:, :3],
                'score': result['scores'][idx],
                'area': area,
            })

        # pose nms
        score_per_joint = cfg.model.test_cfg.get('score_per_joint', False)
        keep = oks_nms(
            pose_results,
            pose_nms_thr,
            sigmas,
            score_per_joint=score_per_joint)
        pose_results = [pose_results[_keep] for _keep in keep]

    return pose_results, returned_outputs


def vis_pose_result(model,
                    img,
                    result,
                    radius=4,
                    thickness=1,
                    kpt_score_thr=0.3,
                    bbox_color='green',
                    dataset='TopDownCocoDataset',
                    dataset_info=None,
                    show=False,
                    out_file=None):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str | np.ndarray): Image filename or loaded image.
        result (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
        radius (int): Radius of circles.
        thickness (int): Thickness of lines.
        kpt_score_thr (float): The threshold to visualize the keypoints.
        skeleton (list[tuple()]): Default None.
        show (bool):  Whether to show the image. Default True.
        out_file (str|None): The filename of the output visualization image.
    """

    # get dataset info
    if (dataset_info is None and hasattr(model, 'cfg')
            and 'dataset_info' in model.cfg):
        dataset_info = DatasetInfo(model.cfg.dataset_info)

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

        if dataset in ('TopDownCocoDataset', 'BottomUpCocoDataset',
                       'TopDownOCHumanDataset', 'AnimalMacaqueDataset'):
            # show the results
            skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                        [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                        [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
                        [3, 5], [4, 6]]

            pose_link_color = palette[[
                0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16
            ]]
            pose_kpt_color = palette[[
                16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0
            ]]

        elif dataset == 'TopDownCocoWholeBodyDataset':
            # show the results
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

            pose_link_color = palette[[
                0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16
            ] + [16, 16, 16, 16, 16, 16] + [
                0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16,
                16
            ] + [
                0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16,
                16
            ]]
            pose_kpt_color = palette[
                [16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0] +
                [0, 0, 0, 0, 0, 0] + [19] * (68 + 42)]

        elif dataset == 'TopDownAicDataset':
            skeleton = [[2, 1], [1, 0], [0, 13], [13, 3], [3, 4], [4, 5],
                        [8, 7], [7, 6], [6, 9], [9, 10], [10, 11], [12, 13],
                        [0, 6], [3, 9]]

            pose_link_color = palette[[
                9, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 0, 7, 7
            ]]
            pose_kpt_color = palette[[
                9, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 0, 0
            ]]

        elif dataset == 'TopDownMpiiDataset':
            skeleton = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5], [6, 7],
                        [7, 8], [8, 9], [8, 12], [12, 11], [11, 10], [8, 13],
                        [13, 14], [14, 15]]

            pose_link_color = palette[[
                16, 16, 16, 16, 16, 16, 7, 7, 0, 9, 9, 9, 9, 9, 9
            ]]
            pose_kpt_color = palette[[
                16, 16, 16, 16, 16, 16, 7, 7, 0, 0, 9, 9, 9, 9, 9, 9
            ]]

        elif dataset == 'TopDownMpiiTrbDataset':
            skeleton = [[12, 13], [13, 0], [13, 1], [0, 2], [1, 3], [2, 4],
                        [3, 5], [0, 6], [1, 7], [6, 7], [6, 8], [7,
                                                                 9], [8, 10],
                        [9, 11], [14, 15], [16, 17], [18, 19], [20, 21],
                        [22, 23], [24, 25], [26, 27], [28, 29], [30, 31],
                        [32, 33], [34, 35], [36, 37], [38, 39]]

            pose_link_color = palette[[16] * 14 + [19] * 13]
            pose_kpt_color = palette[[16] * 14 + [0] * 26]

        elif dataset in ('OneHand10KDataset', 'FreiHandDataset',
                         'PanopticDataset'):
            skeleton = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7],
                        [7, 8], [0, 9], [9, 10], [10, 11], [11, 12], [0, 13],
                        [13, 14], [14, 15], [15, 16], [0, 17], [17, 18],
                        [18, 19], [19, 20]]

            pose_link_color = palette[[
                0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16,
                16
            ]]
            pose_kpt_color = palette[[
                0, 0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16,
                16, 16
            ]]

        elif dataset == 'InterHand2DDataset':
            skeleton = [[0, 1], [1, 2], [2, 3], [4, 5], [5, 6], [6, 7], [8, 9],
                        [9, 10], [10, 11], [12, 13], [13, 14], [14, 15],
                        [16, 17], [17, 18], [18, 19], [3, 20], [7, 20],
                        [11, 20], [15, 20], [19, 20]]

            pose_link_color = palette[[
                0, 0, 0, 4, 4, 4, 8, 8, 8, 12, 12, 12, 16, 16, 16, 0, 4, 8, 12,
                16
            ]]
            pose_kpt_color = palette[[
                0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16,
                16, 0
            ]]

        elif dataset == 'Face300WDataset':
            # show the results
            skeleton = []

            pose_link_color = palette[[]]
            pose_kpt_color = palette[[19] * 68]
            kpt_score_thr = 0

        elif dataset == 'FaceAFLWDataset':
            # show the results
            skeleton = []

            pose_link_color = palette[[]]
            pose_kpt_color = palette[[19] * 19]
            kpt_score_thr = 0

        elif dataset == 'FaceCOFWDataset':
            # show the results
            skeleton = []

            pose_link_color = palette[[]]
            pose_kpt_color = palette[[19] * 29]
            kpt_score_thr = 0

        elif dataset == 'FaceWFLWDataset':
            # show the results
            skeleton = []

            pose_link_color = palette[[]]
            pose_kpt_color = palette[[19] * 98]
            kpt_score_thr = 0

        elif dataset == 'AnimalHorse10Dataset':
            skeleton = [[0, 1], [1, 12], [12, 16], [16, 21], [21, 17],
                        [17, 11], [11, 10], [10, 8], [8, 9], [9, 12], [2, 3],
                        [3, 4], [5, 6], [6, 7], [13, 14], [14, 15], [18, 19],
                        [19, 20]]

            pose_link_color = palette[[4] * 10 + [6] * 2 + [6] * 2 + [7] * 2 +
                                      [7] * 2]
            pose_kpt_color = palette[[
                4, 4, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 7, 7, 7, 4, 4, 7, 7, 7,
                4
            ]]

        elif dataset == 'AnimalFlyDataset':
            skeleton = [[1, 0], [2, 0], [3, 0], [4, 3], [5, 4], [7, 6], [8, 7],
                        [9, 8], [11, 10], [12, 11], [13, 12], [15, 14],
                        [16, 15], [17, 16], [19, 18], [20, 19], [21, 20],
                        [23, 22], [24, 23], [25, 24], [27, 26], [28, 27],
                        [29, 28], [30, 3], [31, 3]]

            pose_link_color = palette[[0] * 25]
            pose_kpt_color = palette[[0] * 32]

        elif dataset == 'AnimalLocustDataset':
            skeleton = [[1, 0], [2, 1], [3, 2], [4, 3], [6, 5], [7, 6], [9, 8],
                        [10, 9], [11, 10], [13, 12], [14, 13], [15, 14],
                        [17, 16], [18, 17], [19, 18], [21, 20], [22, 21],
                        [24, 23], [25, 24], [26, 25], [28, 27], [29, 28],
                        [30, 29], [32, 31], [33, 32], [34, 33]]

            pose_link_color = palette[[0] * 26]
            pose_kpt_color = palette[[0] * 35]

        elif dataset == 'AnimalZebraDataset':
            skeleton = [[1, 0], [2, 1], [3, 2], [4, 2], [5, 7], [6, 7], [7, 2],
                        [8, 7]]

            pose_link_color = palette[[0] * 8]
            pose_kpt_color = palette[[0] * 9]

        elif dataset in 'AnimalPoseDataset':
            skeleton = [[0, 1], [0, 2], [1, 3], [0, 4], [1, 4], [4, 5], [5, 7],
                        [6, 7], [5, 8], [8, 12], [12, 16], [5, 9], [9, 13],
                        [13, 17], [6, 10], [10, 14], [14, 18], [6, 11],
                        [11, 15], [15, 19]]

            pose_link_color = palette[[0] * 20]
            pose_kpt_color = palette[[0] * 20]
        else:
            NotImplementedError()

    if hasattr(model, 'module'):
        model = model.module

    img = model.show_result(
        img,
        result,
        skeleton,
        radius=radius,
        thickness=thickness,
        pose_kpt_color=pose_kpt_color,
        pose_link_color=pose_link_color,
        kpt_score_thr=kpt_score_thr,
        bbox_color=bbox_color,
        show=show,
        out_file=out_file)

    return img


def inference_gesture_model(
    model,
    videos_or_paths,
    bboxes=None,
    dataset_info=None,
):

    cfg = model.cfg
    device = next(model.parameters()).device
    if device.type == 'cpu':
        device = -1

    # build the data pipeline
    test_pipeline = Compose(cfg.test_pipeline)
    _pipeline_gpu_speedup(test_pipeline, next(model.parameters()).device)

    # data preprocessing
    data = defaultdict(list)
    data['label'] = -1

    if not isinstance(videos_or_paths, (tuple, list)):
        videos_or_paths = [videos_or_paths]
    if isinstance(videos_or_paths[0], str):
        data['video_file'] = videos_or_paths
    else:
        data['video'] = videos_or_paths

    if bboxes is not None:
        data['bbox'] = bboxes

    if isinstance(dataset_info, dict):
        data['modality'] = dataset_info.get('modality', ['rgb'])
        data['fps'] = dataset_info.get('fps', None)
        if not isinstance(data['fps'], (tuple, list)):
            data['fps'] = [data['fps']]

    data = test_pipeline(data)
    batch_data = collate([data], samples_per_gpu=1)
    batch_data = scatter(batch_data, [device])[0]

    # inference
    with torch.no_grad():
        output = model.forward(return_loss=False, **batch_data)
        scores = []
        for modal, logit in output['logits'].items():
            while logit.ndim > 2:
                logit = logit.mean(dim=2)
            score = torch.softmax(logit, dim=1)
            scores.append(score)
        score = torch.stack(scores, dim=2).mean(dim=2)
        pred_score, pred_label = torch.max(score, dim=1)

    return pred_label, pred_score


def process_mmdet_results(mmdet_results, cat_id=1):
    """Process mmdet results, and return a list of bboxes.

    Args:
        mmdet_results (list|tuple): mmdet results.
        cat_id (int): category id (default: 1 for human)

    Returns:
        person_results (list): a list of detected bounding boxes
    """
    if isinstance(mmdet_results, tuple):
        det_results = mmdet_results[0]
    else:
        det_results = mmdet_results

    bboxes = det_results[cat_id - 1]

    person_results = []
    for bbox in bboxes:
        person = {}
        person['bbox'] = bbox
        person_results.append(person)

    return person_results


def collect_multi_frames(video, frame_id, indices, online=False):
    """Collect multi frames from the video.

    Args:
        video (mmcv.VideoReader): A VideoReader of the input video file.
        frame_id (int): index of the current frame
        indices (list(int)): index offsets of the frames to collect
        online (bool): inference mode, if set to True, can not use future
            frame information.

    Returns:
        list(ndarray): multi frames collected from the input video file.
    """
    num_frames = len(video)
    frames = []
    # put the current frame at first
    frames.append(video[frame_id])
    # use multi frames for inference
    for idx in indices:
        # skip current frame
        if idx == 0:
            continue
        support_idx = frame_id + idx
        # online mode, can not use future frame information
        if online:
            support_idx = np.clip(support_idx, 0, frame_id)
        else:
            support_idx = np.clip(support_idx, 0, num_frames - 1)
        frames.append(video[support_idx])

    return frames
