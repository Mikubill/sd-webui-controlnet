# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod
from typing import Dict, List, Optional, Union

import numpy as np

from ..utils import FrameMessage, Message
from .node import Node


class BaseVisualizerNode(Node):
    """Base class for nodes whose function is to create visual effects, like
    visualizing model predictions, showing graphics or showing text messages.

    All subclass should implement the method ``draw()``.

    Args:
        name (str): The node name (also thread name)
        input_buffer (str): The name of the input buffer
        output_buffer (str | list): The name(s) of the output buffer(s).
        enable_key (str|int, optional): Set a hot-key to toggle enable/disable
            of the node. If an int value is given, it will be treated as an
            ascii code of a key. Please note: (1) If ``enable_key`` is set,
            the ``bypass()`` method need to be overridden to define the node
            behavior when disabled; (2) Some hot-keys are reserved for
            particular use. For example: 'q', 'Q' and 27 are used for exiting.
            Default: ``None``
        enable (bool): Default enable/disable status. Default: ``True``
    """

    def __init__(self,
                 name: str,
                 input_buffer: str,
                 output_buffer: Union[str, List[str]],
                 enable_key: Optional[Union[str, int]] = None,
                 enable: bool = True):

        super().__init__(name=name, enable_key=enable_key, enable=enable)

        # Register buffers
        self.register_input_buffer(input_buffer, 'input', trigger=True)
        self.register_output_buffer(output_buffer)

    def process(self, input_msgs: Dict[str, Message]) -> Union[Message, None]:
        input_msg = input_msgs['input']

        img = self.draw(input_msg)
        input_msg.set_image(img)

        return input_msg

    def bypass(self, input_msgs: Dict[str, Message]) -> Union[Message, None]:
        return input_msgs['input']

    @abstractmethod
    def draw(self, input_msg: FrameMessage) -> np.ndarray:
        """Draw on the frame image of the input FrameMessage.

        Args:
            input_msg (:obj:`FrameMessage`): The message of the frame to draw
                on

        Returns:
            np.array: The processed image.
        """
