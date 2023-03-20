# Copyright (c) OpenMMLab. All rights reserved.
import torch

from .hook import HOOKS, Hook


@HOOKS.register_module()
class EmptyCacheHook(Hook):

    def __init__(self,
                 before_epoch: bool = False,
                 after_epoch: bool = True,
                 after_iter: bool = False):
        self._before_epoch = before_epoch
        self._after_epoch = after_epoch
        self._after_iter = after_iter

    def after_iter(self, runner):
        if self._after_iter:
            torch.cuda.empty_cache()

    def before_epoch(self, runner):
        if self._before_epoch:
            torch.cuda.empty_cache()

    def after_epoch(self, runner):
        if self._after_epoch:
            torch.cuda.empty_cache()
