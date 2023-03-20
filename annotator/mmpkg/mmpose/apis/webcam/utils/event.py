# Copyright (c) OpenMMLab. All rights reserved.
import logging
from collections import defaultdict
from contextlib import contextmanager
from threading import Event
from typing import Optional

logger = logging.getLogger('Event')


class EventManager():
    """A helper class to manage events.

    :class:`EventManager` provides interfaces to register, set, clear and
    check events by name.
    """

    def __init__(self):
        self._events = defaultdict(Event)

    def register_event(self, event_name: str, is_keyboard: bool = False):
        """Register an event. A event must be registered first before being
        set, cleared or checked.

        Args:
            event_name (str): The indicator of the event. The name should be
                unique in one :class:`EventManager` instance
            is_keyboard (bool): Specify weather it is a keyboard event. If so,
                the ``event_name`` should be the key value, and the indicator
                will be set as ``'_keyboard_{event_name}'``. Otherwise, the
                ``event_name`` will be directly used as the indicator.
                Default: ``False``
        """
        if is_keyboard:
            event_name = self._get_keyboard_event_name(event_name)
        self._events[event_name] = Event()

    def set(self, event_name: str, is_keyboard: bool = False):
        """Set the internal flag of an event to ``True``.

        Args:
            event_name (str): The indicator of the event
            is_keyboard (bool): Specify weather it is a keyboard event. See
                ``register_event()`` for details. Default: False
        """
        if is_keyboard:
            event_name = self._get_keyboard_event_name(event_name)
        self._events[event_name].set()
        logger.info(f'Event {event_name} is set.')

    def wait(self,
             event_name: str = None,
             is_keyboard: bool = False,
             timeout: Optional[float] = None) -> bool:
        """Block until the internal flag of an event is ``True``.

        Args:
            event_name (str): The indicator of the event
            is_keyboard (bool): Specify weather it is a keyboard event. See
                ``register_event()`` for details. Default: False
            timeout (float, optional): The optional maximum blocking time in
                seconds. Default: ``None``

        Returns:
            bool: The internal event flag on exit.
        """
        if is_keyboard:
            event_name = self._get_keyboard_event_name(event_name)
        return self._events[event_name].wait(timeout)

    def is_set(self,
               event_name: str = None,
               is_keyboard: Optional[bool] = False) -> bool:
        """Check weather the internal flag of an event is ``True``.

        Args:
            event_name (str): The indicator of the event
            is_keyboard (bool): Specify weather it is a keyboard event. See
                ``register_event()`` for details. Default: False
        Returns:
            bool: The internal event flag.
        """
        if is_keyboard:
            event_name = self._get_keyboard_event_name(event_name)
        return self._events[event_name].is_set()

    def clear(self,
              event_name: str = None,
              is_keyboard: Optional[bool] = False):
        """Reset the internal flag of en event to False.

        Args:
            event_name (str): The indicator of the event
            is_keyboard (bool): Specify weather it is a keyboard event. See
                ``register_event()`` for details. Default: False
        """
        if is_keyboard:
            event_name = self._get_keyboard_event_name(event_name)
        self._events[event_name].clear()
        logger.info(f'Event {event_name} is cleared.')

    @staticmethod
    def _get_keyboard_event_name(key):
        """Get keyboard event name from the key value."""
        return f'_keyboard_{chr(key) if isinstance(key,int) else key}'

    @contextmanager
    def wait_and_handle(self,
                        event_name: str = None,
                        is_keyboard: Optional[bool] = False):
        """Context manager that blocks until an evenet is set ``True`` and then
        goes into the context.

        The internal event flag will be reset ``False`` automatically before
        entering the context.

        Args:
            event_name (str): The indicator of the event
            is_keyboard (bool): Specify weather it is a keyboard event. See
                ``register_event()`` for details. Default: False

        Example::
            >>> from annotator.mmpkg.mmpose.apis.webcam.utils import EventManager
            >>> manager = EventManager()
            >>> manager.register_event('q', is_keybard=True)

            >>> # Once the keyboard event `q` is set, ``wait_and_handle``
            >>> # will reset the event and enter the context to invoke
            >>> # ``foo()``
            >>> with manager.wait_and_handle('q', is_keybard=True):
            ...     foo()
        """
        self.wait(event_name, is_keyboard)
        try:
            yield
        finally:
            self.clear(event_name, is_keyboard)
