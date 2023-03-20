# Copyright (c) OpenMMLab. All rights reserved.
import time
from typing import List, Union

from annotator.mmpkg.mmpose.utils.timer import RunningAverage
from ..node import Node
from ..registry import NODES


@NODES.register_module()
class ObjectAssignerNode(Node):
    """Assign the object information to the frame message.

    :class:`ObjectAssignerNode` enables asynchronous processing of model
    inference and video I/O, so the video will be captured and displayed
    smoothly regardless of the model inference speed. Specifically,
    :class:`ObjectAssignerNode` takes messages from both model branch and
    video I/O branch as its input, indicated as "object message" and "frame
    message" respectively. When an object message arrives it will update the
    latest object information; and when a frame message arrives, it will be
    assigned with the latest object information and output.

    Specially, if the webcam executor is set to synchrounous mode, the
    behavior of :class:`ObjectAssignerNode` will be different: When an object
    message arrives, it will trigger an output of itself; and the frame
    messages will be ignored.

    Args:
        name (str): The node name (also thread name)
        frame_buffer (str): Buffer name for frame messages
        object_buffer (str): Buffer name for object messages
        output_buffer (str): The name(s) of the output buffer(s)

    Example::
        >>> cfg =dict(
        ...     type='ObjectAssignerNode',
        ...     name='object assigner',
        ...     frame_buffer='_frame_',
        ...     # `_frame_` is an executor-reserved buffer
        ...     object_buffer='animal_pose',
        ...     output_buffer='frame')

        >>> from annotator.mmpkg.mmpose.apis.webcam.nodes import NODES
        >>> node = NODES.build(cfg)
    """

    def __init__(self, name: str, frame_buffer: str, object_buffer: str,
                 output_buffer: Union[str, List[str]]):
        super().__init__(name=name, enable=True)
        self.synchronous = None

        # Cache the latest model result
        self.last_object_msg = None
        self.last_output_msg = None

        # Inference speed analysis
        self.frame_fps = RunningAverage(window=10)
        self.frame_lag = RunningAverage(window=10)
        self.object_fps = RunningAverage(window=10)
        self.object_lag = RunningAverage(window=10)

        # Register buffers
        # The trigger buffer depends on the executor.synchronous attribute,
        # so it will be set later after the executor is assigned in
        # ``set_executor``.
        self.register_input_buffer(object_buffer, 'object', trigger=False)
        self.register_input_buffer(frame_buffer, 'frame', trigger=False)
        self.register_output_buffer(output_buffer)

    def set_executor(self, executor):
        super().set_executor(executor)
        # Set synchronous according to the executor
        if executor.synchronous:
            self.synchronous = True
            trigger = 'object'
        else:
            self.synchronous = False
            trigger = 'frame'

        # Set trigger input buffer according to the synchronous setting
        for buffer_info in self._input_buffers:
            if buffer_info.input_name == trigger:
                buffer_info.trigger = True

    def process(self, input_msgs):
        object_msg = input_msgs['object']

        # Update last result
        if object_msg is not None:
            # Update result FPS
            if self.last_object_msg is not None:
                self.object_fps.update(
                    1.0 /
                    (object_msg.timestamp - self.last_object_msg.timestamp))
            # Update inference latency
            self.object_lag.update(time.time() - object_msg.timestamp)
            # Update last inference result
            self.last_object_msg = object_msg

        if not self.synchronous:
            # Asynchronous mode:
            # Assign the latest object information to the
            # current frame.
            frame_msg = input_msgs['frame']

            self.frame_lag.update(time.time() - frame_msg.timestamp)

            # Assign objects to frame
            if self.last_object_msg is not None:
                frame_msg.update_objects(self.last_object_msg.get_objects())
                frame_msg.merge_route_info(
                    self.last_object_msg.get_route_info())

            output_msg = frame_msg

        else:
            # Synchronous mode:
            # The current frame will be ignored. Instead,
            # the frame from which the latest object information is obtained
            # will be used.
            self.frame_lag.update(time.time() - object_msg.timestamp)
            output_msg = object_msg

        # Update frame fps and lag
        if self.last_output_msg is not None:
            self.frame_lag.update(time.time() - output_msg.timestamp)
            self.frame_fps.update(
                1.0 / (output_msg.timestamp - self.last_output_msg.timestamp))
        self.last_output_msg = output_msg

        return output_msg

    def _get_node_info(self):
        info = super()._get_node_info()
        info['object_fps'] = self.object_fps.average()
        info['object_lag (ms)'] = self.object_lag.average() * 1000
        info['frame_fps'] = self.frame_fps.average()
        info['frame_lag (ms)'] = self.frame_lag.average() * 1000
        return info
