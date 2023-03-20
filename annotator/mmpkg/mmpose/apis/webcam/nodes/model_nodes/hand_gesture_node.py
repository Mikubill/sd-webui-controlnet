# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import numpy as np

from annotator.mmpkg.mmpose.apis import inference_gesture_model, init_pose_model
from ...utils import Message
from ..node import Node
from ..registry import NODES


def _compute_area(bbox):
    """Compute the area of bounding box in the format 'xxyy'."""
    area = abs(bbox['bbox'][2] - bbox['bbox'][0]) * abs(bbox['bbox'][3] -
                                                        bbox['bbox'][1])
    return area


def _merge_bbox(bboxes: List[Dict], ratio=0.5):
    """Merge bboxes in a video to create a new bbox that covers the region
    where hand moves in the video."""

    if len(bboxes) <= 1:
        return bboxes

    bboxes.sort(key=lambda b: _compute_area(b), reverse=True)
    merged = False
    for i in range(1, len(bboxes)):
        small_area = _compute_area(bboxes[i])
        x1 = max(bboxes[0]['bbox'][0], bboxes[i]['bbox'][0])
        y1 = max(bboxes[0]['bbox'][1], bboxes[i]['bbox'][1])
        x2 = min(bboxes[0]['bbox'][2], bboxes[i]['bbox'][2])
        y2 = min(bboxes[0]['bbox'][3], bboxes[i]['bbox'][3])
        area_ratio = (abs(x2 - x1) * abs(y2 - y1)) / small_area
        if area_ratio > ratio:
            bboxes[0]['bbox'][0] = min(bboxes[0]['bbox'][0],
                                       bboxes[i]['bbox'][0])
            bboxes[0]['bbox'][1] = min(bboxes[0]['bbox'][1],
                                       bboxes[i]['bbox'][1])
            bboxes[0]['bbox'][2] = max(bboxes[0]['bbox'][2],
                                       bboxes[i]['bbox'][2])
            bboxes[0]['bbox'][3] = max(bboxes[0]['bbox'][3],
                                       bboxes[i]['bbox'][3])
            merged = True
            break

    if merged:
        bboxes.pop(i)
        return _merge_bbox(bboxes, ratio)
    else:
        # return the largest bounding box
        return [bboxes[0]]


@NODES.register_module()
class HandGestureRecognizerNode(Node):
    """Perform hand gesture recognition using MMPose model.

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
        min_frame (int): Set the lower bound of clip length for gesture
            recognition. Default: 16
        fps (int): Camera fps. Default: 30
        score_thr (float): Threshold of probability to recognize salient
            gesture. Default: 0.7

    Example::
        >>> cfg = dict(
        ...     type='HandGestureRecognizerNode',
        ...     name='hand gesture recognition',
        ...     model_config='configs/hand/gesture_sview_rgbd_vid/mtut/'
        ...     'nvgesture/i3d_nvgesture_bbox_112x112_fps15_rgb.py',
        ...     model_checkpoint='https://download.openmmlab.com/mmpose/'
        ...    'gesture/mtut/'
        ...    'i3d_nvgesture_bbox_112x112_fps15-363b5956_20220530.pth',
        ...     input_buffer='det_result',
        ...     output_buffer='geature',
        ...     fps=15)

        >>> from annotator.mmpkg.mmpose.apis.webcam.nodes import NODES
        >>> node = NODES.build(cfg)
    """

    def __init__(self,
                 name: str,
                 model_config: str,
                 model_checkpoint: str,
                 input_buffer: str,
                 output_buffer: Union[str, List[str]],
                 enable_key: Optional[Union[str, int]] = None,
                 enable: bool = True,
                 device: str = 'cuda:0',
                 min_frame: int = 16,
                 fps: int = 30,
                 score_thr: float = 0.7,
                 multi_input: bool = True):

        super().__init__(
            name=name,
            enable_key=enable_key,
            enable=enable,
            multi_input=multi_input)

        self._clip_buffer = []  # items: (clip message, num of frames)
        self.score_thr = score_thr
        self.min_frame = min_frame
        self.fps = fps

        # Init model
        self.model_config = model_config
        self.model_checkpoint = model_checkpoint
        self.device = device.lower()
        self.model = init_pose_model(
            self.model_config, self.model_checkpoint, device=self.device)

        # Register buffers
        self.register_input_buffer(input_buffer, 'input', trigger=True)
        self.register_output_buffer(output_buffer)

    def bypass(self, input_msgs):
        return input_msgs['input']

    @property
    def totol_clip_length(self):
        return sum([clip[1] for clip in self._clip_buffer])

    def _extend_clips(self, clips: List[Message]):
        """Push the newly loaded clips from buffer, and discard old clips."""
        for clip in clips:
            clip_length = clip.get_image().shape[0]
            self._clip_buffer.append((clip, clip_length))

        total_length = 0
        for i in range(-2, -len(self._clip_buffer) - 1, -1):
            total_length += self._clip_buffer[i][1]
            if total_length >= self.min_frame:
                self._clip_buffer = self._clip_buffer[i:]
                break

    def _merge_clips(self):
        """Concat the clips into a longer video, and gather bboxes."""
        videos = [clip[0].get_image() for clip in self._clip_buffer]
        video = np.concatenate(videos)

        bboxes = []
        for clip in self._clip_buffer:
            objects = clip[0].get_objects(lambda x: x.get('label') == 'hand')
            bboxes.append(_merge_bbox(objects))
        bboxes = list(filter(len, bboxes))
        return video, bboxes

    def process(self, input_msgs: Dict[str, Message]) -> Message:
        """Load and process the clips with hand detection result, and recognize
        the gesture."""

        input_msg = input_msgs['input']

        if not self.multi_input:
            input_msg = [input_msg]

        self._extend_clips(input_msg)
        video, bboxes = self._merge_clips()
        msg = input_msg[-1]

        if self.totol_clip_length >= self.min_frame and len(
                bboxes) > 0 and max(map(len, bboxes)) > 0:
            # Inference gesture
            pred_label, pred_score = inference_gesture_model(
                self.model,
                video,
                bboxes=bboxes,
                dataset_info=dict(
                    name='camera', fps=self.fps, modality=['rgb']))
            pred_label, pred_score = pred_label[0], pred_score[0]

            # assign gesture
            result = bboxes[-1][0]
            if pred_score > self.score_thr:
                label = pred_label.item()
                label = self.model.cfg.dataset_info.category_info[label]
                result['label'] = label
                result['gesture'] = label

            msg.update_objects([result])

        return msg
