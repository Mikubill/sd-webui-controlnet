# Copyright (c) OpenMMLab. All rights reserved.
import logging
import sys
import time
import warnings
from contextlib import nullcontext
from threading import Thread
from typing import Dict, List, Optional, Tuple, Union

import cv2

from .nodes import NODES
from .utils import (BufferManager, EventManager, FrameMessage, ImageCapture,
                    VideoEndingMessage, is_image_file, limit_max_fps)

DEFAULT_FRAME_BUFFER_SIZE = 1
DEFAULT_INPUT_BUFFER_SIZE = 1
DEFAULT_DISPLAY_BUFFER_SIZE = 0
DEFAULT_USER_BUFFER_SIZE = 1

logger = logging.getLogger('Executor')


class WebcamExecutor():
    """The interface to build and execute webcam applications from configs.

    Parameters:
        nodes (list[dict]): Node configs. See :class:`webcam.nodes.Node` for
            details
        name (str): Executor name. Default: 'MMPose Webcam App'.
        camera_id (int | str): The camera ID (usually the ID of the default
            camera is 0). Alternatively a file path or a URL can be given
            to load from a video or image file.
        camera_frame_shape (tuple, optional): Set the frame shape of the
            camera in (width, height). If not given, the default frame shape
            will be used. This argument is only valid when using a camera
            as the input source. Default: ``None``
        camera_max_fps (int): Video reading maximum FPS. Default: 30
        buffer_sizes (dict, optional): A dict to specify buffer sizes. The
            key is the buffer name and the value is the buffer size.
            Default: ``None``

    Example::
        >>> cfg = dict(
        >>>     name='Test Webcam',
        >>>     camera_id=0,
        >>>     camera_max_fps=30,
        >>>     nodes=[
        >>>         dict(
        >>>             type='MonitorNode',
        >>>             name='monitor',
        >>>             enable_key='m',
        >>>             enable=False,
        >>>             input_buffer='_frame_',
        >>>             output_buffer='display'),
        >>>         dict(
        >>>             type='RecorderNode',
        >>>             name='recorder',
        >>>             out_video_file='webcam_output.mp4',
        >>>             input_buffer='display',
        >>>             output_buffer='_display_')
        >>>     ])

        >>> executor = WebcamExecutor(**cfg)
    """

    def __init__(self,
                 nodes: List[Dict],
                 name: str = 'MMPose Webcam App',
                 camera_id: Union[int, str] = 0,
                 camera_max_fps: int = 30,
                 camera_frame_shape: Optional[Tuple[int, int]] = None,
                 synchronous: bool = False,
                 buffer_sizes: Optional[Dict[str, int]] = None):

        # Basic parameters
        self.name = name
        self.camera_id = camera_id
        self.camera_max_fps = camera_max_fps
        self.camera_frame_shape = camera_frame_shape
        self.synchronous = synchronous

        # self.buffer_manager manages data flow between executor and nodes
        self.buffer_manager = BufferManager()
        # self.event_manager manages event-based asynchronous communication
        self.event_manager = EventManager()
        # self.node_list holds all node instance
        self.node_list = []
        # self.vcap is used to read camera frames. It will be built when the
        # executor starts running
        self.vcap = None

        # Register executor events
        self.event_manager.register_event('_exit_', is_keyboard=False)
        if self.synchronous:
            self.event_manager.register_event('_idle_', is_keyboard=False)

        # Register nodes
        if not nodes:
            raise ValueError('No node is registered to the executor.')

        # Register default buffers
        if buffer_sizes is None:
            buffer_sizes = {}
        # _frame_ buffer
        frame_buffer_size = buffer_sizes.get('_frame_',
                                             DEFAULT_FRAME_BUFFER_SIZE)
        self.buffer_manager.register_buffer('_frame_', frame_buffer_size)
        # _input_ buffer
        input_buffer_size = buffer_sizes.get('_input_',
                                             DEFAULT_INPUT_BUFFER_SIZE)
        self.buffer_manager.register_buffer('_input_', input_buffer_size)
        # _display_ buffer
        display_buffer_size = buffer_sizes.get('_display_',
                                               DEFAULT_DISPLAY_BUFFER_SIZE)
        self.buffer_manager.register_buffer('_display_', display_buffer_size)

        # Build all nodes:
        for node_cfg in nodes:
            logger.info(f'Create node: {node_cfg.name}({node_cfg.type})')
            node = NODES.build(node_cfg)

            # Register node
            self.node_list.append(node)

            # Register buffers
            for buffer_info in node.registered_buffers:
                buffer_name = buffer_info.buffer_name
                if buffer_name in self.buffer_manager:
                    continue
                buffer_size = buffer_sizes.get(buffer_name,
                                               DEFAULT_USER_BUFFER_SIZE)
                self.buffer_manager.register_buffer(buffer_name, buffer_size)
                logger.info(
                    f'Register user buffer: {buffer_name}({buffer_size})')

            # Register events
            for event_info in node.registered_events:
                self.event_manager.register_event(
                    event_name=event_info.event_name,
                    is_keyboard=event_info.is_keyboard)
                logger.info(f'Register event: {event_info.event_name}')

        # Set executor for nodes
        # This step is performed after node building when the executor has
        # create full buffer/event managers and can
        for node in self.node_list:
            logger.info(f'Set executor for node: {node.name})')
            node.set_executor(self)

    def _read_camera(self):
        """Read video frames from the caemra (or the source video/image) and
        put them into input buffers."""

        camera_id = self.camera_id
        fps = self.camera_max_fps

        # Build video capture
        if is_image_file(camera_id):
            self.vcap = ImageCapture(camera_id)
        else:
            self.vcap = cv2.VideoCapture(camera_id)
            if self.camera_frame_shape is not None:
                width, height = self.camera_frame_shape
                self.vcap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.vcap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.vcap.isOpened():
            warnings.warn(f'Cannot open camera (ID={camera_id})')
            sys.exit()

        # Read video frames in a loop
        first_frame = True
        while not self.event_manager.is_set('_exit_'):
            if self.synchronous:
                if first_frame:
                    cm = nullcontext()
                else:
                    # Read a new frame until the last frame has been processed
                    cm = self.event_manager.wait_and_handle('_idle_')
            else:
                # Read frames with a maximum FPS
                cm = limit_max_fps(fps)

            first_frame = False

            with cm:
                # Read a frame
                ret_val, frame = self.vcap.read()
                if ret_val:
                    # Put frame message (for display) into buffer `_frame_`
                    frame_msg = FrameMessage(frame)
                    self.buffer_manager.put('_frame_', frame_msg)

                    # Put input message (for model inference or other use)
                    # into buffer `_input_`
                    input_msg = FrameMessage(frame.copy())
                    input_msg.update_route_info(
                        node_name='Camera Info',
                        node_type='none',
                        info=self._get_camera_info())
                    self.buffer_manager.put_force('_input_', input_msg)
                    logger.info('Read one frame.')
                else:
                    logger.info('Reached the end of the video.')
                    # Put a video ending signal
                    self.buffer_manager.put('_frame_', VideoEndingMessage())
                    self.buffer_manager.put('_input_', VideoEndingMessage())
                    # Wait for `_exit_` event util a timeout occurs
                    if not self.event_manager.wait('_exit_', timeout=5.0):
                        break

        self.vcap.release()

    def _display(self):
        """Receive processed frames from the output buffer and display on
        screen."""

        output_msg = None

        while not self.event_manager.is_set('_exit_'):
            while self.buffer_manager.is_empty('_display_'):
                time.sleep(0.001)

            # Set _idle_ to allow reading next frame
            if self.synchronous:
                self.event_manager.set('_idle_')

            # acquire output from buffer
            output_msg = self.buffer_manager.get('_display_')

            # None indicates input stream ends
            if isinstance(output_msg, VideoEndingMessage):
                self.event_manager.set('_exit_')
                break

            img = output_msg.get_image()

            # show in a window
            cv2.imshow(self.name, img)

            # handle keyboard input
            key = cv2.waitKey(1)
            if key != -1:
                self._on_keyboard_input(key)

        cv2.destroyAllWindows()

        # Avoid dead lock
        if self.synchronous:
            self.event_manager.set('_idle_')

    def _on_keyboard_input(self, key):
        """Handle the keyboard input.

        The key 'Q' and `ESC` will trigger an '_exit_' event, which will be
        responded by all nodes and the executor itself to exit. Other keys will
        trigger keyboard event to be responded by the nodes which has
        registered corresponding event. See :class:`webcam.utils.EventManager`
        for details.
        """

        if key in (27, ord('q'), ord('Q')):
            logger.info(f'Exit event captured: {key}')
            self.event_manager.set('_exit_')
        else:
            logger.info(f'Keyboard event captured: {key}')
            self.event_manager.set(key, is_keyboard=True)

    def _get_camera_info(self):
        """Return the camera information in a dict."""

        frame_width = self.vcap.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_height = self.vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frame_rate = self.vcap.get(cv2.CAP_PROP_FPS)

        cam_info = {
            'Camera ID': self.camera_id,
            'Camera resolution': f'{frame_width}x{frame_height}',
            'Camera FPS': frame_rate,
        }

        return cam_info

    def run(self):
        """Start the executor.

        This method starts all nodes as well as video I/O in separate threads.
        """

        try:
            # Start node threads
            non_daemon_nodes = []
            for node in self.node_list:
                node.start()
                if not node.daemon:
                    non_daemon_nodes.append(node)

            # Create a thread to read video frames
            t_read = Thread(target=self._read_camera, args=())
            t_read.start()

            # Run display in the main thread
            self._display()
            logger.info('Display has stopped.')

            # joint non-daemon nodes and executor threads
            logger.info('Camera reading is about to join.')
            t_read.join()

            for node in non_daemon_nodes:
                logger.info(f'Node {node.name} is about to join.')
                node.join()
            logger.info('All nodes jointed successfully.')

        except KeyboardInterrupt:
            pass
