# Copyright (c) OpenMMLab. All rights reserved.
from itertools import groupby
from typing import Dict, List, Optional, Union

import cv2
import numpy as np

from ...utils import get_eye_keypoint_ids
from ..base_visualizer_node import BaseVisualizerNode
from ..registry import NODES


@NODES.register_module()
class BigeyeEffectNode(BaseVisualizerNode):
    """Apply big-eye effect to the objects with eye keypoints in the frame.

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
        kpt_thr (float): The score threshold of valid keypoints. Default: 0.5

    Example::
        >>> cfg = dict(
        ...    type='SunglassesEffectNode',
        ...    name='sunglasses',
        ...    enable_key='s',
        ...    enable=False,
        ...    input_buffer='vis',
        ...    output_buffer='vis_sunglasses')

        >>> from annotator.mmpkg.mmpose.apis.webcam.nodes import NODES
        >>> node = NODES.build(cfg)
    """

    def __init__(self,
                 name: str,
                 input_buffer: str,
                 output_buffer: Union[str, List[str]],
                 enable_key: Optional[Union[str, int]] = None,
                 enable: bool = True,
                 kpt_thr: float = 0.5):

        super().__init__(
            name=name,
            input_buffer=input_buffer,
            output_buffer=output_buffer,
            enable_key=enable_key,
            enable=enable)
        self.kpt_thr = kpt_thr

    def draw(self, input_msg):
        canvas = input_msg.get_image()

        objects = input_msg.get_objects(lambda x:
                                        ('keypoints' in x and 'bbox' in x))

        for model_cfg, group in groupby(objects,
                                        lambda x: x['pose_model_cfg']):
            left_eye_index, right_eye_index = get_eye_keypoint_ids(model_cfg)
            canvas = self.apply_bigeye_effect(canvas, group, left_eye_index,
                                              right_eye_index)
        return canvas

    def apply_bigeye_effect(self, canvas: np.ndarray, objects: List[Dict],
                            left_eye_index: int,
                            right_eye_index: int) -> np.ndarray:
        """Apply big-eye effect.

        Args:
            canvas (np.ndarray): The image to apply the effect
            objects (list[dict]): The object list with bbox and keypoints
                - "bbox" ([K, 4(or 5)]): bbox in [x1, y1, x2, y2, (score)]
                - "keypoints" ([K,3]): keypoints in [x, y, score]
            left_eye_index (int): Keypoint index of left eye
            right_eye_index (int): Keypoint index of right eye

        Returns:
            np.ndarray: Processed image.
        """

        xx, yy = np.meshgrid(
            np.arange(canvas.shape[1]), np.arange(canvas.shape[0]))
        xx = xx.astype(np.float32)
        yy = yy.astype(np.float32)

        for obj in objects:
            bbox = obj['bbox']
            kpts = obj['keypoints']

            if kpts[left_eye_index,
                    2] < self.kpt_thr or kpts[right_eye_index,
                                              2] < self.kpt_thr:
                continue

            kpt_leye = kpts[left_eye_index, :2]
            kpt_reye = kpts[right_eye_index, :2]
            for xc, yc in [kpt_leye, kpt_reye]:

                # distortion parameters
                k1 = 0.001
                epe = 1e-5

                scale = (bbox[2] - bbox[0])**2 + (bbox[3] - bbox[1])**2
                r2 = ((xx - xc)**2 + (yy - yc)**2)
                r2 = (r2 + epe) / scale  # normalized by bbox scale

                xx = (xx - xc) / (1 + k1 / r2) + xc
                yy = (yy - yc) / (1 + k1 / r2) + yc

            canvas = cv2.remap(
                canvas,
                xx,
                yy,
                interpolation=cv2.INTER_AREA,
                borderMode=cv2.BORDER_REPLICATE)

        return canvas
