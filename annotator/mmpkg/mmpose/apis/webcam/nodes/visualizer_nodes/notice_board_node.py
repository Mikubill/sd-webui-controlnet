# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from annotator.mmpkg.mmcv import color_val

from ...utils import FrameMessage
from ..base_visualizer_node import BaseVisualizerNode
from ..registry import NODES


@NODES.register_module()
class NoticeBoardNode(BaseVisualizerNode):
    """Show text messages in the frame.

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
        content_lines (list[str], optional): The lines of text message to show
            in the frame. If not given, a default message will be shown.
            Default: ``None``
        x_offset (int): The position of the notice board's left border in
            pixels. Default: 20
        y_offset (int): The position of the notice board's top border in
            pixels. Default: 20
        y_delta (int): The line height in pixels. Default: 15
        text_color (str|tuple): The font color represented in a color name or
            a BGR tuple. Default: ``'black'``
        backbround_color (str|tuple): The background color represented in a
            color name or a BGR tuple. Default: (255, 183, 0)
        text_scale (float): The font scale factor that is multiplied by the
            base size. Default: 0.4

    Example::
        >>> cfg = dict(
        ...     type='NoticeBoardNode',
        ...     name='instruction',
        ...     enable_key='h',
        ...     enable=True,
        ...     input_buffer='vis_bigeye',
        ...     output_buffer='vis_notice',
        ...     content_lines=[
        ...         'This is a demo for pose visualization and simple image '
        ...         'effects. Have fun!', '', 'Hot-keys:',
        ...         '"v": Pose estimation result visualization',
        ...         '"s": Sunglasses effect B-)', '"b": Big-eye effect 0_0',
        ...         '"h": Show help information',
        ...         '"m": Show diagnostic information', '"q": Exit'
        ...     ],
        ... )

        >>> from annotator.mmpkg.mmpose.apis.webcam.nodes import NODES
        >>> node = NODES.build(cfg)
    """

    default_content_lines = ['This is a notice board!']

    def __init__(self,
                 name: str,
                 input_buffer: str,
                 output_buffer: Union[str, List[str]],
                 enable_key: Optional[Union[str, int]] = None,
                 enable: bool = True,
                 content_lines: Optional[List[str]] = None,
                 x_offset: int = 20,
                 y_offset: int = 20,
                 y_delta: int = 15,
                 text_color: Union[str, Tuple[int, int, int]] = 'black',
                 background_color: Union[str, Tuple[int, int,
                                                    int]] = (255, 183, 0),
                 text_scale: float = 0.4):
        super().__init__(
            name=name,
            input_buffer=input_buffer,
            output_buffer=output_buffer,
            enable_key=enable_key,
            enable=enable)

        self.x_offset = x_offset
        self.y_offset = y_offset
        self.y_delta = y_delta
        self.text_color = color_val(text_color)
        self.background_color = color_val(background_color)
        self.text_scale = text_scale

        if content_lines:
            self.content_lines = content_lines
        else:
            self.content_lines = self.default_content_lines

    def draw(self, input_msg: FrameMessage) -> np.ndarray:
        img = input_msg.get_image()
        canvas = np.full(img.shape, self.background_color, dtype=img.dtype)

        x = self.x_offset
        y = self.y_offset

        max_len = max([len(line) for line in self.content_lines])

        def _put_line(line=''):
            nonlocal y
            cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_DUPLEX,
                        self.text_scale, self.text_color, 1)
            y += self.y_delta

        for line in self.content_lines:
            _put_line(line)

        x1 = max(0, self.x_offset)
        x2 = min(img.shape[1], int(x + max_len * self.text_scale * 20))
        y1 = max(0, self.y_offset - self.y_delta)
        y2 = min(img.shape[0], y)

        src1 = canvas[y1:y2, x1:x2]
        src2 = img[y1:y2, x1:x2]
        img[y1:y2, x1:x2] = cv2.addWeighted(src1, 0.5, src2, 0.5, 0)

        return img
