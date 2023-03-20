# Copyright (c) OpenMMLab. All rights reserved.
import copy
from dataclasses import dataclass
from itertools import zip_longest
from typing import Dict, List, Optional, Union

from ...utils import get_config_path
from ..node import Node
from ..registry import NODES

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

import numpy as np

from annotator.mmpkg.mmpose.apis import (get_track_id, inference_top_down_pose_model,
                         init_pose_model)
from annotator.mmpkg.mmpose.core import Smoother


@dataclass
class TrackInfo:
    next_id: int = 0
    last_objects: List = None


@NODES.register_module()
class PoseTrackerNode(Node):
    """Perform object detection and top-down pose estimation. Only detect
    objects every few frames, and use the pose estimation results to track the
    object at interval.

    Note that MMDetection is required for this node. Please refer to
    `MMDetection documentation <https://mmdetection.readthedocs.io/en
    /latest/get_started.html>`_ for the installation guide.

    Parameters:
        name (str): The node name (also thread name)
        det_model_cfg (str): The config file of the detection model
        det_model_checkpoint (str): The checkpoint file of the detection model
        pose_model_cfg (str): The config file of the pose estimation model
        pose_model_checkpoint (str): The checkpoint file of the pose
            estimation model
        input_buffer (str): The name of the input buffer
        output_buffer (str|list): The name(s) of the output buffer(s)
        enable_key (str|int, optional): Set a hot-key to toggle enable/disable
            of the node. If an int value is given, it will be treated as an
            ascii code of a key. Please note: (1) If ``enable_key`` is set,
            the ``bypass()`` method need to be overridden to define the node
            behavior when disabled; (2) Some hot-keys are reserved for
            particular use. For example: 'q', 'Q' and 27 are used for exiting.
            Default: ``None``
        enable (bool): Default enable/disable status. Default: ``True``
        device (str): Specify the device to hold model weights and inference
            the model. Default: ``'cuda:0'``
        det_interval (int): Set the detection interval in frames. For example,
            ``det_interval==10`` means inference the detection model every
            10 frames. Default: 1
        class_ids (list[int], optional): Specify the object category indices
            to apply pose estimation. If both ``class_ids`` and ``labels``
            are given, ``labels`` will be ignored. If neither is given, pose
            estimation will be applied for all objects. Default: ``None``
        labels (list[str], optional): Specify the object category names to
            apply pose estimation. See also ``class_ids``. Default: ``None``
        bbox_thr (float): Set a threshold to filter out objects with low bbox
            scores. Default: 0.5
        kpt2bbox_cfg (dict, optional): Configure the process to get object
            bbox from its keypoints during tracking. Specifically, the bbox
            is obtained from the minimal outer rectangle of the keyponits with
            following configurable arguments: ``'scale'``, the coefficient to
            expand the keypoint outer rectangle, defaults to 1.5;
            ``'kpt_thr'``: a threshold to filter out low-scored keypoint,
            defaults to 0.3. See ``self.default_kpt2bbox_cfg`` for details
        smooth (bool): If set to ``True``, a :class:`Smoother` will be used to
            refine the pose estimation result. Default: ``True``
        smooth_filter_cfg (str): The filter config path to build the smoother.
            Only valid when ``smooth==True``. Default to use an OneEuro filter

    Example::
        >>> cfg = dict(
        ...    type='PoseTrackerNode',
        ...    name='pose tracker',
        ...    det_model_config='demo/mmdetection_cfg/'
        ...    'ssdlite_mobilenetv2_scratch_600e_coco.py',
        ...    det_model_checkpoint='https://download.openmmlab.com'
        ...    '/mmdetection/v2.0/ssd/'
        ...    'ssdlite_mobilenetv2_scratch_600e_coco/ssdlite_mobilenetv2_'
        ...    'scratch_600e_coco_20210629_110627-974d9307.pth',
        ...    pose_model_config='configs/wholebody/2d_kpt_sview_rgb_img/'
        ...    'topdown_heatmap/coco-wholebody/'
        ...    'vipnas_mbv3_coco_wholebody_256x192_dark.py',
        ...    pose_model_checkpoint='https://download.openmmlab.com/mmpose/'
        ...    'top_down/vipnas/vipnas_mbv3_coco_wholebody_256x192_dark'
        ...    '-e2158108_20211205.pth',
        ...    det_interval=10,
        ...    labels=['person'],
        ...    smooth=True,
        ...    device='cuda:0',
        ...    # `_input_` is an executor-reserved buffer
        ...    input_buffer='_input_',
        ...    output_buffer='human_pose')

        >>> from annotator.mmpkg.mmpose.apis.webcam.nodes import NODES
        >>> node = NODES.build(cfg)
    """

    default_kpt2bbox_cfg: Dict = dict(scale=1.5, kpt_thr=0.3)

    def __init__(
            self,
            name: str,
            det_model_config: str,
            det_model_checkpoint: str,
            pose_model_config: str,
            pose_model_checkpoint: str,
            input_buffer: str,
            output_buffer: Union[str, List[str]],
            enable_key: Optional[Union[str, int]] = None,
            enable: bool = True,
            device: str = 'cuda:0',
            det_interval: int = 1,
            class_ids: Optional[List] = None,
            labels: Optional[List] = None,
            bbox_thr: float = 0.5,
            kpt2bbox_cfg: Optional[dict] = None,
            smooth: bool = False,
            smooth_filter_cfg: str = 'configs/_base_/filters/one_euro.py'):

        assert has_mmdet, \
            f'MMDetection is required for {self.__class__.__name__}.'

        super().__init__(name=name, enable_key=enable_key, enable=enable)

        self.det_model_config = get_config_path(det_model_config, 'mmdet')
        self.det_model_checkpoint = det_model_checkpoint
        self.pose_model_config = get_config_path(pose_model_config, 'mmpose')
        self.pose_model_checkpoint = pose_model_checkpoint
        self.device = device.lower()
        self.class_ids = class_ids
        self.labels = labels
        self.bbox_thr = bbox_thr
        self.det_interval = det_interval

        if not kpt2bbox_cfg:
            kpt2bbox_cfg = self.default_kpt2bbox_cfg
        self.kpt2bbox_cfg = copy.deepcopy(kpt2bbox_cfg)

        self.det_countdown = 0
        self.track_info = TrackInfo()

        if smooth:
            smooth_filter_cfg = get_config_path(smooth_filter_cfg, 'mmpose')
            self.smoother = Smoother(smooth_filter_cfg, keypoint_dim=2)
        else:
            self.smoother = None

        # init models
        self.det_model = init_detector(
            self.det_model_config,
            self.det_model_checkpoint,
            device=self.device)

        self.pose_model = init_pose_model(
            self.pose_model_config,
            self.pose_model_checkpoint,
            device=self.device)

        # register buffers
        self.register_input_buffer(input_buffer, 'input', trigger=True)
        self.register_output_buffer(output_buffer)

    def bypass(self, input_msgs):
        return input_msgs['input']

    def process(self, input_msgs):
        input_msg = input_msgs['input']
        img = input_msg.get_image()

        if self.det_countdown == 0:
            # get objects by detection model
            self.det_countdown = self.det_interval
            preds = inference_detector(self.det_model, img)
            objects_det = self._post_process_det(preds)
        else:
            # get object by pose tracking
            objects_det = self._get_objects_by_tracking(img.shape)

        self.det_countdown -= 1

        objects_pose, _ = inference_top_down_pose_model(
            self.pose_model,
            img,
            objects_det,
            bbox_thr=self.bbox_thr,
            format='xyxy')

        objects, next_id = get_track_id(
            objects_pose,
            self.track_info.last_objects,
            self.track_info.next_id,
            use_oks=False,
            tracking_thr=0.3)

        self.track_info.next_id = next_id
        self.track_info.last_objects = objects.copy()

        # Pose smoothing
        if self.smoother:
            objects = self.smoother.smooth(objects)

        for obj in objects:
            obj['det_model_cfg'] = self.det_model.cfg
            obj['pose_model_cfg'] = self.pose_model.cfg

        input_msg.update_objects(objects)

        return input_msg

    def _get_objects_by_tracking(self, img_shape):
        objects = []
        for obj in self.track_info.last_objects:
            obj = copy.deepcopy(obj)
            kpts = obj.pop('keypoints')
            bbox = self._keypoints_to_bbox(kpts, img_shape)
            if bbox is not None:
                obj['bbox'][:4] = bbox
            objects.append(obj)

        return objects

    def _keypoints_to_bbox(self, keypoints, img_shape):
        scale = self.kpt2bbox_cfg.get('scale', 1.5)
        kpt_thr = self.kpt2bbox_cfg.get('kpt_thr', 0.3)
        valid = keypoints[:, 2] > kpt_thr

        if not valid.any():
            return None

        x1 = np.min(keypoints[valid, 0])
        y1 = np.min(keypoints[valid, 1])
        x2 = np.max(keypoints[valid, 0])
        y2 = np.max(keypoints[valid, 1])

        xc = 0.5 * (x1 + x2)
        yc = 0.5 * (y1 + y2)
        w = (x2 - x1) * scale
        h = (y2 - y1) * scale

        img_h, img_w = img_shape[:2]

        bbox = np.array([
            np.clip(0, img_w, xc - 0.5 * w),
            np.clip(0, img_h, yc - 0.5 * h),
            np.clip(0, img_w, xc + 0.5 * w),
            np.clip(0, img_h, yc + 0.5 * h)
        ]).astype(np.float32)
        return bbox

    def _post_process_det(self, preds):
        """Post-process the predictions of MMDetection model."""
        if isinstance(preds, tuple):
            dets = preds[0]
            segms = preds[1]
        else:
            dets = preds
            segms = [[]] * len(dets)

        classes = self.det_model.CLASSES
        if isinstance(classes, str):
            classes = (classes, )

        assert len(dets) == len(classes)
        assert len(segms) == len(classes)

        objects = []

        for i, (label, bboxes, masks) in enumerate(zip(classes, dets, segms)):

            for bbox, mask in zip_longest(bboxes, masks):
                if bbox[4] < self.bbox_thr:
                    continue
                obj = {
                    'class_id': i,
                    'label': label,
                    'bbox': bbox,
                    'mask': mask,
                }
                objects.append(obj)

        return objects
