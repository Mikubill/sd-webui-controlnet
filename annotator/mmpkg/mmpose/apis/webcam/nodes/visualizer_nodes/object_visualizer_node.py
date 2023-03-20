# Copyright (c) OpenMMLab. All rights reserved.
from itertools import groupby
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from annotator.mmpkg.mmcv import color_val

from annotator.mmpkg.mmpose.core import imshow_bboxes, imshow_keypoints
from annotator.mmpkg.mmpose.datasets import DatasetInfo
from ...utils import FrameMessage
from ..base_visualizer_node import BaseVisualizerNode
from ..registry import NODES


@NODES.register_module()
class ObjectVisualizerNode(BaseVisualizerNode):
    """Visualize the bounding box and keypoints of objects.

    Args:
        name (str): The node name (also thread name)
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
        show_bbox (bool): Set ``True`` to show the bboxes of detection
            objects. Default: ``True``
        show_keypoint (bool): Set ``True`` to show the pose estimation
            results. Default: ``True``
        must_have_bbox (bool): Only show objects with keypoints.
            Default: ``False``
        kpt_thr (float): The threshold of keypoint score. Default: 0.3
        radius (int): The radius of keypoint. Default: 4
        thickness (int): The thickness of skeleton. Default: 2
        bbox_color (str|tuple|dict): The color of bboxes. If a single color is
            given (a str like 'green' or a BGR tuple like (0, 255, 0)), it
            will be used for all bboxes. If a dict is given, it will be used
            as a map from class labels to bbox colors. If not given, a default
            color map will be used. Default: ``None``

    Example::
        >>> cfg = dict(
        ...    type='ObjectVisualizerNode',
        ...    name='object visualizer',
        ...    enable_key='v',
        ...    enable=True,
        ...    show_bbox=True,
        ...    must_have_keypoint=False,
        ...    show_keypoint=True,
        ...    input_buffer='frame',
        ...    output_buffer='vis')

        >>> from annotator.mmpkg.mmpose.apis.webcam.nodes import NODES
        >>> node = NODES.build(cfg)
    """

    default_bbox_color = {
        'person': (148, 139, 255),
        'cat': (255, 255, 0),
        'dog': (255, 255, 0),
    }

    def __init__(self,
                 name: str,
                 input_buffer: str,
                 output_buffer: Union[str, List[str]],
                 enable_key: Optional[Union[str, int]] = None,
                 enable: bool = True,
                 show_bbox: bool = True,
                 show_keypoint: bool = True,
                 must_have_keypoint: bool = False,
                 kpt_thr: float = 0.3,
                 radius: int = 4,
                 thickness: int = 2,
                 bbox_color: Optional[Union[str, Tuple, Dict]] = None):

        super().__init__(
            name=name,
            input_buffer=input_buffer,
            output_buffer=output_buffer,
            enable_key=enable_key,
            enable=enable)

        self.kpt_thr = kpt_thr
        self.radius = radius
        self.thickness = thickness
        self.show_bbox = show_bbox
        self.show_keypoint = show_keypoint
        self.must_have_keypoint = must_have_keypoint

        if bbox_color is None:
            self.bbox_color = self.default_bbox_color
        elif isinstance(bbox_color, dict):
            self.bbox_color = {k: color_val(v) for k, v in bbox_color.items()}
        else:
            self.bbox_color = color_val(bbox_color)

    def _draw_bbox(self, canvas: np.ndarray, input_msg: FrameMessage):
        """Draw object bboxes."""

        if self.must_have_keypoint:
            objects = input_msg.get_objects(
                lambda x: 'bbox' in x and 'keypoints' in x)
        else:
            objects = input_msg.get_objects(lambda x: 'bbox' in x)
        # return if there is no detected objects
        if not objects:
            return canvas

        bboxes = [obj['bbox'] for obj in objects]
        labels = [obj.get('label', None) for obj in objects]
        default_color = (0, 255, 0)

        # Get bbox colors
        if isinstance(self.bbox_color, dict):
            colors = [
                self.bbox_color.get(label, default_color) for label in labels
            ]
        else:
            colors = self.bbox_color

        imshow_bboxes(
            canvas,
            np.vstack(bboxes),
            labels=labels,
            colors=colors,
            text_color='white',
            font_scale=0.5,
            show=False)

        return canvas

    def _draw_keypoint(self, canvas: np.ndarray, input_msg: FrameMessage):
        """Draw object keypoints."""
        objects = input_msg.get_objects(lambda x: 'pose_model_cfg' in x)

        # return if there is no object with keypoints
        if not objects:
            return canvas

        for model_cfg, group in groupby(objects,
                                        lambda x: x['pose_model_cfg']):
            dataset_info = DatasetInfo(model_cfg.dataset_info)
            keypoints = [obj['keypoints'] for obj in group]
            imshow_keypoints(
                canvas,
                keypoints,
                skeleton=dataset_info.skeleton,
                kpt_score_thr=0.3,
                pose_kpt_color=dataset_info.pose_kpt_color,
                pose_link_color=dataset_info.pose_link_color,
                radius=self.radius,
                thickness=self.thickness)

        return canvas

    def draw(self, input_msg: FrameMessage) -> np.ndarray:
        canvas = input_msg.get_image()

        if self.show_bbox:
            canvas = self._draw_bbox(canvas, input_msg)

        if self.show_keypoint:
            canvas = self._draw_keypoint(canvas, input_msg)

        return canvas
