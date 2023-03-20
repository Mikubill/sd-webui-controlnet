# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
from annotator.mmpkg.mmcv import color_val

from ..node import Node
from ..registry import NODES

try:
    import psutil
    psutil_proc = psutil.Process()
except (ImportError, ModuleNotFoundError):
    psutil_proc = None


@NODES.register_module()
class MonitorNode(Node):
    """Show diagnostic information.

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
        x_offset (int): The position of the text box's left border in
            pixels. Default: 20
        y_offset (int): The position of the text box's top border in
            pixels. Default: 20
        y_delta (int): The line height in pixels. Default: 15
        text_color (str|tuple): The font color represented in a color name or
            a BGR tuple. Default: ``'black'``
        backbround_color (str|tuple): The background color represented in a
            color name or a BGR tuple. Default: (255, 183, 0)
        text_scale (float): The font scale factor that is multiplied by the
            base size. Default: 0.4
        ignore_items (list[str], optional): Specify the node information items
            that will not be shown. See ``MonitorNode._default_ignore_items``
            for the default setting.

    Example::
        >>> cfg = dict(
        ...     type='MonitorNode',
        ...     name='monitor',
        ...     enable_key='m',
        ...     enable=False,
        ...     input_buffer='vis_notice',
        ...     output_buffer='display')

        >>> from annotator.mmpkg.mmpose.apis.webcam.nodes import NODES
        >>> node = NODES.build(cfg)
    """

    _default_ignore_items = ['timestamp']

    def __init__(self,
                 name: str,
                 input_buffer: str,
                 output_buffer: Union[str, List[str]],
                 enable_key: Optional[Union[str, int]] = None,
                 enable: bool = False,
                 x_offset=20,
                 y_offset=20,
                 y_delta=15,
                 text_color='black',
                 background_color=(255, 183, 0),
                 text_scale=0.4,
                 ignore_items: Optional[List[str]] = None):
        super().__init__(name=name, enable_key=enable_key, enable=enable)

        self.x_offset = x_offset
        self.y_offset = y_offset
        self.y_delta = y_delta
        self.text_color = color_val(text_color)
        self.background_color = color_val(background_color)
        self.text_scale = text_scale
        if ignore_items is None:
            self.ignore_items = self._default_ignore_items
        else:
            self.ignore_items = ignore_items

        self.register_input_buffer(input_buffer, 'input', trigger=True)
        self.register_output_buffer(output_buffer)

    def process(self, input_msgs):
        input_msg = input_msgs['input']

        input_msg.update_route_info(
            node_name='System Info',
            node_type='none',
            info=self._get_system_info())

        img = input_msg.get_image()
        route_info = input_msg.get_route_info()
        img = self._show_route_info(img, route_info)

        input_msg.set_image(img)
        return input_msg

    def _get_system_info(self):
        """Get the system information including CPU and memory usage.

        Returns:
            dict: The system information items.
        """
        sys_info = {}
        if psutil_proc is not None:
            sys_info['CPU(%)'] = psutil_proc.cpu_percent()
            sys_info['Memory(%)'] = psutil_proc.memory_percent()
        return sys_info

    def _show_route_info(self, img: np.ndarray,
                         route_info: List[Dict]) -> np.ndarray:
        """Show the route information in the frame.

        Args:
            img (np.ndarray): The frame image.
            route_info (list[dict]): The route information of the frame.

        Returns:
            np.ndarray: The processed image.
        """
        canvas = np.full(img.shape, self.background_color, dtype=img.dtype)

        x = self.x_offset
        y = self.y_offset

        max_len = 0

        def _put_line(line=''):
            nonlocal y, max_len
            cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_DUPLEX,
                        self.text_scale, self.text_color, 1)
            y += self.y_delta
            max_len = max(max_len, len(line))

        for node_info in route_info:
            title = f'{node_info["node"]}({node_info["node_type"]})'
            _put_line(title)
            for k, v in node_info['info'].items():
                if k in self.ignore_items:
                    continue
                if isinstance(v, float):
                    v = f'{v:.1f}'
                _put_line(f'    {k}: {v}')

        x1 = max(0, self.x_offset)
        x2 = min(img.shape[1], int(x + max_len * self.text_scale * 20))
        y1 = max(0, self.y_offset - self.y_delta)
        y2 = min(img.shape[0], y)

        src1 = canvas[y1:y2, x1:x2]
        src2 = img[y1:y2, x1:x2]
        img[y1:y2, x1:x2] = cv2.addWeighted(src1, 0.5, src2, 0.5, 0)

        return img

    def bypass(self, input_msgs):
        return input_msgs['input']
