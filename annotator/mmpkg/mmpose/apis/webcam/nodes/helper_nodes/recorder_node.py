# Copyright (c) OpenMMLab. All rights reserved.
from queue import Full, Queue
from threading import Thread
from typing import List, Union

import cv2

from ..node import Node
from ..registry import NODES


@NODES.register_module()
class RecorderNode(Node):
    """Record the video frames into a local file.

    :class:`RecorderNode` uses OpenCV backend to record the video. Recording
    is performed in a separate thread to avoid blocking the data stream. A
    buffer queue is used to cached the arrived frame images.

    Args:
        name (str): The node name (also thread name)
        input_buffer (str): The name of the input buffer
        output_buffer (str|list): The name(s) of the output buffer(s)
        out_video_file (str): The path of the output video file
        out_video_fps (int): The frame rate of the output video. Default: 30
        out_video_codec (str): The codec of the output video. Default: 'mp4v'
        buffer_size (int): Size of the buffer queue that caches the arrived
            frame images.
        enable (bool): Default enable/disable status. Default: ``True``.

    Example::
        >>> cfg = dict(
        ...     type='RecorderNode',
        ...     name='recorder',
        ...     out_video_file='webcam_demo.mp4',
        ...     input_buffer='display',
        ...     output_buffer='_display_'
        ...     # `_display_` is an executor-reserved buffer
        ... )

        >>> from annotator.mmpkg.mmpose.apis.webcam.nodes import NODES
        >>> node = NODES.build(cfg)
    """

    def __init__(
        self,
        name: str,
        input_buffer: str,
        output_buffer: Union[str, List[str]],
        out_video_file: str,
        out_video_fps: int = 30,
        out_video_codec: str = 'mp4v',
        buffer_size: int = 30,
        enable: bool = True,
    ):
        super().__init__(name=name, enable_key=None, enable=enable)

        self.queue = Queue(maxsize=buffer_size)
        self.out_video_file = out_video_file
        self.out_video_fps = out_video_fps
        self.out_video_codec = out_video_codec
        self.vwriter = None

        # Register buffers
        self.register_input_buffer(input_buffer, 'input', trigger=True)
        self.register_output_buffer(output_buffer)

        # Start a new thread to write frame
        self.t_record = Thread(target=self._record, args=(), daemon=True)
        self.t_record.start()

    def process(self, input_msgs):

        input_msg = input_msgs['input']
        img = input_msg.get_image() if input_msg is not None else None
        img_queued = False

        while not img_queued:
            try:
                self.queue.put(img, timeout=1)
                img_queued = True
                self.logger.info('Recorder received one frame.')
            except Full:
                self.logger.warn('Recorder jamed!')

        return input_msg

    def _record(self):
        """This method is used to create a thread to get frame images from the
        buffer queue and write them into the file."""

        while True:

            img = self.queue.get()

            if img is None:
                break

            if self.vwriter is None:
                fourcc = cv2.VideoWriter_fourcc(*self.out_video_codec)
                fps = self.out_video_fps
                frame_size = (img.shape[1], img.shape[0])
                self.vwriter = cv2.VideoWriter(self.out_video_file, fourcc,
                                               fps, frame_size)
                assert self.vwriter.isOpened()

            self.vwriter.write(img)

        self.logger.info('Recorder released.')
        if self.vwriter is not None:
            self.vwriter.release()

    def on_exit(self):
        try:
            # Try putting a None into the output queue so the self.vwriter will
            # be released after all queue frames have been written to file.
            self.queue.put(None, timeout=1)
            self.t_record.join(timeout=1)
        except Full:
            pass

        if self.t_record.is_alive():
            # Force to release self.vwriter
            self.logger.warn('Recorder forced release!')
            if self.vwriter is not None:
                self.vwriter.release()
