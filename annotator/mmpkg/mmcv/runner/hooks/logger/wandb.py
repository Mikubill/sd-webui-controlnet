# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from typing import Dict, Optional, Union

from annotator.mmpkg.mmcv.utils import scandir
from ...dist_utils import master_only
from ..hook import HOOKS
from .base import LoggerHook


@HOOKS.register_module()
class WandbLoggerHook(LoggerHook):
    """Class to log metrics with wandb.

    It requires `wandb`_ to be installed.


    Args:
        init_kwargs (dict): A dict contains the initialization keys. Check
            https://docs.wandb.ai/ref/python/init for more init arguments.
        interval (int): Logging interval (every k iterations).
            Default 10.
        ignore_last (bool): Ignore the log of last iterations in each epoch
            if less than `interval`.
            Default: True.
        reset_flag (bool): Whether to clear the output buffer after logging.
            Default: False.
        commit (bool): Save the metrics dict to the wandb server and increment
            the step. If false ``wandb.log`` just updates the current metrics
            dict with the row argument and metrics won't be saved until
            ``wandb.log`` is called with ``commit=True``.
            Default: True.
        by_epoch (bool): Whether EpochBasedRunner is used.
            Default: True.
        with_step (bool): If True, the step will be logged from
            ``self.get_iters``. Otherwise, step will not be logged.
            Default: True.
        log_artifact (bool): If True, artifacts in {work_dir} will be uploaded
            to wandb after training ends.
            Default: True
            `New in version 1.4.3.`
        out_suffix (str or tuple[str], optional): Those filenames ending with
            ``out_suffix`` will be uploaded to wandb.
            Default: ('.log.json', '.log', '.py').
            `New in version 1.4.3.`
        define_metric_cfg (dict, optional): A dict of metrics and summaries for
            wandb.define_metric. The key is metric and the value is summary.
            The summary should be in ["min", "max", "mean" ,"best", "last",
             "none"].
            For example, if setting
            ``define_metric_cfg={'coco/bbox_mAP': 'max'}``, the maximum value
            of ``coco/bbox_mAP`` will be logged on wandb UI. See
            `wandb docs <https://docs.wandb.ai/ref/python/run#define_metric>`_
            for details.
            Defaults to None.
            `New in version 1.6.3.`

    .. _wandb:
        https://docs.wandb.ai
    """

    def __init__(self,
                 init_kwargs: Optional[Dict] = None,
                 interval: int = 10,
                 ignore_last: bool = True,
                 reset_flag: bool = False,
                 commit: bool = True,
                 by_epoch: bool = True,
                 with_step: bool = True,
                 log_artifact: bool = True,
                 out_suffix: Union[str, tuple] = ('.log.json', '.log', '.py'),
                 define_metric_cfg: Optional[Dict] = None):
        super().__init__(interval, ignore_last, reset_flag, by_epoch)
        self.import_wandb()
        self.init_kwargs = init_kwargs
        self.commit = commit
        self.with_step = with_step
        self.log_artifact = log_artifact
        self.out_suffix = out_suffix
        self.define_metric_cfg = define_metric_cfg

    def import_wandb(self) -> None:
        try:
            import wandb
        except ImportError:
            raise ImportError(
                'Please run "pip install wandb" to install wandb')
        self.wandb = wandb

    @master_only
    def before_run(self, runner) -> None:
        super().before_run(runner)
        if self.wandb is None:
            self.import_wandb()
        if self.init_kwargs:
            self.wandb.init(**self.init_kwargs)  # type: ignore
        else:
            self.wandb.init()  # type: ignore
        summary_choice = ['min', 'max', 'mean', 'best', 'last', 'none']
        if self.define_metric_cfg is not None:
            for metric, summary in self.define_metric_cfg.items():
                if summary not in summary_choice:
                    warnings.warn(
                        f'summary should be in {summary_choice}. '
                        f'metric={metric}, summary={summary} will be skipped.')
                self.wandb.define_metric(  # type: ignore
                    metric, summary=summary)

    @master_only
    def log(self, runner) -> None:
        tags = self.get_loggable_tags(runner)
        if tags:
            if self.with_step:
                self.wandb.log(
                    tags, step=self.get_iter(runner), commit=self.commit)
            else:
                tags['global_step'] = self.get_iter(runner)
                self.wandb.log(tags, commit=self.commit)

    @master_only
    def after_run(self, runner) -> None:
        if self.log_artifact:
            wandb_artifact = self.wandb.Artifact(
                name='artifacts', type='model')
            for filename in scandir(runner.work_dir, self.out_suffix, True):
                local_filepath = osp.join(runner.work_dir, filename)
                wandb_artifact.add_file(local_filepath)
            self.wandb.log_artifact(wandb_artifact)
        self.wandb.join()
