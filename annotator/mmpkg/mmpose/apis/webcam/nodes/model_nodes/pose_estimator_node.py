# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass
from typing import List, Optional, Union

from annotator.mmpkg.mmpose.apis import (get_track_id, inference_top_down_pose_model,
                         init_pose_model)
from annotator.mmpkg.mmpose.core import Smoother
from ...utils import get_config_path
from ..node import Node
from ..registry import NODES


@dataclass
class TrackInfo:
    """Dataclass for object tracking information."""
    next_id: int = 0
    last_objects: List = None


@NODES.register_module()
class TopDownPoseEstimatorNode(Node):
    """Perform top-down pose estimation using MMPose model.

    The node should be placed after an object detection node.

    Parameters:
        name (str): The node name (also thread name)
        model_cfg (str): The model config file
        model_checkpoint (str): The model checkpoint file
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
        class_ids (list[int], optional): Specify the object category indices
            to apply pose estimation. If both ``class_ids`` and ``labels``
            are given, ``labels`` will be ignored. If neither is given, pose
            estimation will be applied for all objects. Default: ``None``
        labels (list[str], optional): Specify the object category names to
            apply pose estimation. See also ``class_ids``. Default: ``None``
        bbox_thr (float): Set a threshold to filter out objects with low bbox
            scores. Default: 0.5
        smooth (bool): If set to ``True``, a :class:`Smoother` will be used to
            refine the pose estimation result. Default: ``True``
        smooth_filter_cfg (str): The filter config path to build the smoother.
            Only valid when ``smooth==True``. By default, OneEuro filter will
            be used.

    Example::
        >>> cfg = dict(
        ...     type='TopDownPoseEstimatorNode',
        ...     name='human pose estimator',
        ...     model_config='configs/wholebody/2d_kpt_sview_rgb_img/'
        ...     'topdown_heatmap/coco-wholebody/'
        ...     'vipnas_mbv3_coco_wholebody_256x192_dark.py',
        ...     model_checkpoint='https://download.openmmlab.com/mmpose/'
        ...     'top_down/vipnas/vipnas_mbv3_coco_wholebody_256x192_dark'
        ...     '-e2158108_20211205.pth',
        ...     labels=['person'],
        ...     smooth=True,
        ...     input_buffer='det_result',
        ...     output_buffer='human_pose')

        >>> from annotator.mmpkg.mmpose.apis.webcam.nodes import NODES
        >>> node = NODES.build(cfg)
    """

    def __init__(
            self,
            name: str,
            model_config: str,
            model_checkpoint: str,
            input_buffer: str,
            output_buffer: Union[str, List[str]],
            enable_key: Optional[Union[str, int]] = None,
            enable: bool = True,
            device: str = 'cuda:0',
            class_ids: Optional[List[int]] = None,
            labels: Optional[List[str]] = None,
            bbox_thr: float = 0.5,
            smooth: bool = False,
            smooth_filter_cfg: str = 'configs/_base_/filters/one_euro.py'):
        super().__init__(name=name, enable_key=enable_key, enable=enable)

        # Init model
        self.model_config = get_config_path(model_config, 'mmpose')
        self.model_checkpoint = model_checkpoint
        self.device = device.lower()

        self.class_ids = class_ids
        self.labels = labels
        self.bbox_thr = bbox_thr

        if smooth:
            smooth_filter_cfg = get_config_path(smooth_filter_cfg, 'mmpose')
            self.smoother = Smoother(smooth_filter_cfg, keypoint_dim=2)
        else:
            self.smoother = None
        # Init model
        self.model = init_pose_model(
            self.model_config, self.model_checkpoint, device=self.device)

        # Store history for pose tracking
        self.track_info = TrackInfo()

        # Register buffers
        self.register_input_buffer(input_buffer, 'input', trigger=True)
        self.register_output_buffer(output_buffer)

    def bypass(self, input_msgs):
        return input_msgs['input']

    def process(self, input_msgs):

        input_msg = input_msgs['input']
        img = input_msg.get_image()

        if self.class_ids:
            objects = input_msg.get_objects(
                lambda x: x.get('class_id') in self.class_ids)
        elif self.labels:
            objects = input_msg.get_objects(
                lambda x: x.get('label') in self.labels)
        else:
            objects = input_msg.get_objects()
        # Inference pose
        objects, _ = inference_top_down_pose_model(
            self.model, img, objects, bbox_thr=self.bbox_thr, format='xyxy')

        # Pose tracking
        objects, next_id = get_track_id(
            objects,
            self.track_info.last_objects,
            self.track_info.next_id,
            use_oks=False,
            tracking_thr=0.3)

        self.track_info.next_id = next_id
        # Copy the prediction to avoid track_info being affected by smoothing
        self.track_info.last_objects = [obj.copy() for obj in objects]

        # Pose smoothing
        if self.smoother:
            objects = self.smoother.smooth(objects)

        for obj in objects:
            obj['pose_model_cfg'] = self.model.cfg
        input_msg.update_objects(objects)

        return input_msg
