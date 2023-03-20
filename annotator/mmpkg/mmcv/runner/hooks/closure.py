# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable

from .hook import HOOKS, Hook


@HOOKS.register_module()
class ClosureHook(Hook):

    def __init__(self, fn_name: str, fn: Callable):
        assert hasattr(self, fn_name)
        assert callable(fn)
        setattr(self, fn_name, fn)
