# Copyright (c) OpenMMLab. All rights reserved.
from functools import wraps
from queue import Queue
from typing import Any, Dict, List, Optional

from annotator.mmpkg.mmcv import is_seq_of

__all__ = ['BufferManager']


def check_buffer_registered(exist=True):
    """A function wrapper to check the buffer existence before it is being used
    by the wrapped function.

    Args:
        exist (bool): If set to ``True``, assert the buffer exists; if set to
            ``False``, assert the buffer does not exist. Default: ``True``
    """

    def wrapper(func):

        @wraps(func)
        def wrapped(manager, name, *args, **kwargs):
            if exist:
                # Assert buffer exist
                if name not in manager:
                    raise ValueError(f'Fail to call {func.__name__}: '
                                     f'buffer "{name}" is not registered.')
            else:
                # Assert buffer not exist
                if name in manager:
                    raise ValueError(f'Fail to call {func.__name__}: '
                                     f'buffer "{name}" is already registered.')
            return func(manager, name, *args, **kwargs)

        return wrapped

    return wrapper


class Buffer(Queue):

    def put_force(self, item: Any):
        """Force to put an item into the buffer.

        If the buffer is already full, the earliest item in the buffer will be
        remove to make room for the incoming item.

        Args:
            item (any): The item to put into the buffer
        """
        with self.mutex:
            if self.maxsize > 0:
                while self._qsize() >= self.maxsize:
                    _ = self._get()
                    self.unfinished_tasks -= 1

            self._put(item)
            self.unfinished_tasks += 1
            self.not_empty.notify()


class BufferManager():
    """A helper class to manage multiple buffers.

    Parameters:
        buffer_type (type): The class to build buffer instances. Default:
            :class:`mmpose.apis.webcam.utils.buffer.Buffer`.
        buffers (dict, optional): Create :class:`BufferManager` from existing
            buffers. Each item should a buffer name and the buffer. If not
            given, an empty buffer manager will be create. Default: ``None``
    """

    def __init__(self,
                 buffer_type: type = Buffer,
                 buffers: Optional[Dict] = None):
        self.buffer_type = buffer_type
        if buffers is None:
            self._buffers = {}
        else:
            if is_seq_of(list(buffers.values()), buffer_type):
                self._buffers = buffers.copy()
            else:
                raise ValueError('The values of buffers should be instance '
                                 f'of {buffer_type}')

    def __contains__(self, name):
        return name in self._buffers

    @check_buffer_registered(False)
    def register_buffer(self, name, maxsize: int = 0):
        """Register a buffer.

        If the buffer already exists, an ValueError will be raised.

        Args:
            name (any): The buffer name
            maxsize (int): The capacity of the buffer. If set to 0, the
                capacity is unlimited. Default: 0
        """
        self._buffers[name] = self.buffer_type(maxsize)

    @check_buffer_registered()
    def put(self, name, item, block: bool = True, timeout: float = None):
        """Put an item into specified buffer.

        Args:
            name (any): The buffer name
            item (any): The item to put into the buffer
            block (bool): If set to ``True``, block if necessary util a free
                slot is available in the target buffer. It blocks at most
                ``timeout`` seconds and raises the ``Full`` exception.
                Otherwise, put an item on the queue if a free slot is
                immediately available, else raise the ``Full`` exception.
                Default: ``True``
            timeout (float, optional): The most waiting time in seconds if
                ``block`` is ``True``. Default: ``None``
        """
        self._buffers[name].put(item, block, timeout)

    @check_buffer_registered()
    def put_force(self, name, item):
        """Force to put an item into specified buffer. If the buffer was full,
        the earliest item within the buffer will be popped out to make a free
        slot.

        Args:
            name (any): The buffer name
            item (any): The item to put into the buffer
        """
        self._buffers[name].put_force(item)

    @check_buffer_registered()
    def get(self, name, block: bool = True, timeout: float = None) -> Any:
        """Remove an return an item from the specified buffer.

        Args:
            name (any): The buffer name
            block (bool): If set to ``True``, block if necessary until an item
                is available in the target buffer. It blocks at most
                ``timeout`` seconds and raises the ``Empty`` exception.
                Otherwise, return an item if one is immediately available,
                else raise the ``Empty`` exception. Default: ``True``
            timeout (float, optional): The most waiting time in seconds if
                ``block`` is ``True``. Default: ``None``

        Returns:
            any: The returned item.
        """
        return self._buffers[name].get(block, timeout)

    @check_buffer_registered()
    def is_empty(self, name) -> bool:
        """Check if a buffer is empty.

        Args:
            name (any): The buffer name

        Returns:
            bool: Weather the buffer is empty.
        """
        return self._buffers[name].empty()

    @check_buffer_registered()
    def is_full(self, name):
        """Check if a buffer is full.

        Args:
            name (any): The buffer name

        Returns:
            bool: Weather the buffer is full.
        """
        return self._buffers[name].full()

    def get_sub_manager(self, buffer_names: List[str]) -> 'BufferManager':
        """Return a :class:`BufferManager` instance that covers a subset of the
        buffers in the parent. The is usually used to partially share the
        buffers of the executor to the node.

        Args:
            buffer_names (list): The list of buffers to create the sub manager

        Returns:
            BufferManager: The created sub buffer manager.
        """
        buffers = {name: self._buffers[name] for name in buffer_names}
        return BufferManager(self.buffer_type, buffers)

    def get_info(self):
        """Returns the information of all buffers in the manager.

        Returns:
            dict[any, dict]: Each item is a buffer name and the information
            dict of that buffer.
        """
        buffer_info = {}
        for name, buffer in self._buffers.items():
            buffer_info[name] = {
                'size': buffer.size,
                'maxsize': buffer.maxsize
            }
        return buffer_info
