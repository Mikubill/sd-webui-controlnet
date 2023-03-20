# Copyright (c) OpenMMLab. All rights reserved.
from ...dist_utils import master_only
from ..hook import HOOKS
from .base import LoggerHook


@HOOKS.register_module()
class SegmindLoggerHook(LoggerHook):
    """Class to log metrics to Segmind.

    It requires `Segmind`_ to be installed.

    Args:
        interval (int): Logging interval (every k iterations). Default: 10.
        ignore_last (bool): Ignore the log of last iterations in each epoch
            if less than `interval`. Default True.
        reset_flag (bool): Whether to clear the output buffer after logging.
            Default False.
        by_epoch (bool): Whether EpochBasedRunner is used. Default True.

    .. _Segmind:
        https://docs.segmind.com/python-library
    """

    def __init__(self,
                 interval: int = 10,
                 ignore_last: bool = True,
                 reset_flag: bool = False,
                 by_epoch=True):
        super().__init__(interval, ignore_last, reset_flag, by_epoch)
        self.import_segmind()

    def import_segmind(self) -> None:
        try:
            import segmind
        except ImportError:
            raise ImportError(
                "Please run 'pip install segmind' to install segmind")
        self.log_metrics = segmind.tracking.fluent.log_metrics
        self.mlflow_log = segmind.utils.logging_utils.try_mlflow_log

    @master_only
    def log(self, runner) -> None:
        tags = self.get_loggable_tags(runner)
        if tags:
            # logging metrics to segmind
            self.mlflow_log(
                self.log_metrics, tags, step=runner.epoch, epoch=runner.epoch)
