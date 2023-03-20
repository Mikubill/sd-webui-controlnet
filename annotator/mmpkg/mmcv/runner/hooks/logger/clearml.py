# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict, Optional

from ...dist_utils import master_only
from ..hook import HOOKS
from .base import LoggerHook


@HOOKS.register_module()
class ClearMLLoggerHook(LoggerHook):
    """Class to log metrics with clearml.

    It requires `clearml`_ to be installed.


    Args:
        init_kwargs (dict): A dict contains the `clearml.Task.init`
            initialization keys. See `taskinit`_  for more details.
        interval (int): Logging interval (every k iterations). Default 10.
        ignore_last (bool): Ignore the log of last iterations in each epoch
            if less than `interval`. Default: True.
        reset_flag (bool): Whether to clear the output buffer after logging.
            Default: False.
        by_epoch (bool): Whether EpochBasedRunner is used. Default: True.

    .. _clearml:
        https://clear.ml/docs/latest/docs/
    .. _taskinit:
        https://clear.ml/docs/latest/docs/references/sdk/task/#taskinit
    """

    def __init__(self,
                 init_kwargs: Optional[Dict] = None,
                 interval: int = 10,
                 ignore_last: bool = True,
                 reset_flag: bool = False,
                 by_epoch: bool = True):
        super().__init__(interval, ignore_last, reset_flag, by_epoch)
        self.import_clearml()
        self.init_kwargs = init_kwargs

    def import_clearml(self):
        try:
            import clearml
        except ImportError:
            raise ImportError(
                'Please run "pip install clearml" to install clearml')
        self.clearml = clearml

    @master_only
    def before_run(self, runner) -> None:
        super().before_run(runner)
        task_kwargs = self.init_kwargs if self.init_kwargs else {}
        self.task = self.clearml.Task.init(**task_kwargs)
        self.task_logger = self.task.get_logger()

    @master_only
    def log(self, runner) -> None:
        tags = self.get_loggable_tags(runner)
        for tag, val in tags.items():
            self.task_logger.report_scalar(tag, tag, val,
                                           self.get_iter(runner))
