# Copyright (c) OpenMMLab. All rights reserved.
import numbers
from math import cos, pi
from typing import Callable, List, Optional, Union

import annotator.mmpkg.mmcv as mmcv
from annotator.mmpkg.mmcv import runner
from .hook import HOOKS, Hook


class LrUpdaterHook(Hook):
    """LR Scheduler in MMCV.

    Args:
        by_epoch (bool): LR changes epoch by epoch
        warmup (string): Type of warmup used. It can be None(use no warmup),
            'constant', 'linear' or 'exp'
        warmup_iters (int): The number of iterations or epochs that warmup
            lasts
        warmup_ratio (float): LR used at the beginning of warmup equals to
            warmup_ratio * initial_lr
        warmup_by_epoch (bool): When warmup_by_epoch == True, warmup_iters
            means the number of epochs that warmup lasts, otherwise means the
            number of iteration that warmup lasts
    """

    def __init__(self,
                 by_epoch: bool = True,
                 warmup: Optional[str] = None,
                 warmup_iters: int = 0,
                 warmup_ratio: float = 0.1,
                 warmup_by_epoch: bool = False) -> None:
        # validate the "warmup" argument
        if warmup is not None:
            if warmup not in ['constant', 'linear', 'exp']:
                raise ValueError(
                    f'"{warmup}" is not a supported type for warming up, valid'
                    ' types are "constant", "linear" and "exp"')
        if warmup is not None:
            assert warmup_iters > 0, \
                '"warmup_iters" must be a positive integer'
            assert 0 < warmup_ratio <= 1.0, \
                '"warmup_ratio" must be in range (0,1]'

        self.by_epoch = by_epoch
        self.warmup = warmup
        self.warmup_iters: Optional[int] = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.warmup_by_epoch = warmup_by_epoch

        if self.warmup_by_epoch:
            self.warmup_epochs: Optional[int] = self.warmup_iters
            self.warmup_iters = None
        else:
            self.warmup_epochs = None

        self.base_lr: Union[list, dict] = []  # initial lr for all param groups
        self.regular_lr: list = []  # expected lr if no warming up is performed

    def _set_lr(self, runner, lr_groups):
        if isinstance(runner.optimizer, dict):
            for k, optim in runner.optimizer.items():
                for param_group, lr in zip(optim.param_groups, lr_groups[k]):
                    param_group['lr'] = lr
        else:
            for param_group, lr in zip(runner.optimizer.param_groups,
                                       lr_groups):
                param_group['lr'] = lr

    def get_lr(self, runner: 'runner.BaseRunner', base_lr: float):
        raise NotImplementedError

    def get_regular_lr(self, runner: 'runner.BaseRunner'):
        if isinstance(runner.optimizer, dict):
            lr_groups = {}
            for k in runner.optimizer.keys():
                _lr_group = [
                    self.get_lr(runner, _base_lr)
                    for _base_lr in self.base_lr[k]
                ]
                lr_groups.update({k: _lr_group})

            return lr_groups
        else:
            return [self.get_lr(runner, _base_lr) for _base_lr in self.base_lr]

    def get_warmup_lr(self, cur_iters: int):

        def _get_warmup_lr(cur_iters, regular_lr):
            if self.warmup == 'constant':
                warmup_lr = [_lr * self.warmup_ratio for _lr in regular_lr]
            elif self.warmup == 'linear':
                k = (1 - cur_iters / self.warmup_iters) * (1 -
                                                           self.warmup_ratio)
                warmup_lr = [_lr * (1 - k) for _lr in regular_lr]
            elif self.warmup == 'exp':
                k = self.warmup_ratio**(1 - cur_iters / self.warmup_iters)
                warmup_lr = [_lr * k for _lr in regular_lr]
            return warmup_lr

        if isinstance(self.regular_lr, dict):
            lr_groups = {}
            for key, regular_lr in self.regular_lr.items():
                lr_groups[key] = _get_warmup_lr(cur_iters, regular_lr)
            return lr_groups
        else:
            return _get_warmup_lr(cur_iters, self.regular_lr)

    def before_run(self, runner: 'runner.BaseRunner'):
        # NOTE: when resuming from a checkpoint, if 'initial_lr' is not saved,
        # it will be set according to the optimizer params
        if isinstance(runner.optimizer, dict):
            self.base_lr = {}
            for k, optim in runner.optimizer.items():
                for group in optim.param_groups:
                    group.setdefault('initial_lr', group['lr'])
                _base_lr = [
                    group['initial_lr'] for group in optim.param_groups
                ]
                self.base_lr.update({k: _base_lr})
        else:
            for group in runner.optimizer.param_groups:  # type: ignore
                group.setdefault('initial_lr', group['lr'])
            self.base_lr = [
                group['initial_lr']
                for group in runner.optimizer.param_groups  # type: ignore
            ]

    def before_train_epoch(self, runner: 'runner.BaseRunner'):
        if self.warmup_iters is None:
            epoch_len = len(runner.data_loader)  # type: ignore
            self.warmup_iters = self.warmup_epochs * epoch_len  # type: ignore

        if not self.by_epoch:
            return

        self.regular_lr = self.get_regular_lr(runner)
        self._set_lr(runner, self.regular_lr)

    def before_train_iter(self, runner: 'runner.BaseRunner'):
        cur_iter = runner.iter
        assert isinstance(self.warmup_iters, int)
        if not self.by_epoch:
            self.regular_lr = self.get_regular_lr(runner)
            if self.warmup is None or cur_iter >= self.warmup_iters:
                self._set_lr(runner, self.regular_lr)
            else:
                warmup_lr = self.get_warmup_lr(cur_iter)
                self._set_lr(runner, warmup_lr)
        elif self.by_epoch:
            if self.warmup is None or cur_iter > self.warmup_iters:
                return
            elif cur_iter == self.warmup_iters:
                self._set_lr(runner, self.regular_lr)
            else:
                warmup_lr = self.get_warmup_lr(cur_iter)
                self._set_lr(runner, warmup_lr)


@HOOKS.register_module()
class FixedLrUpdaterHook(LrUpdaterHook):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        return base_lr


@HOOKS.register_module()
class StepLrUpdaterHook(LrUpdaterHook):
    """Step LR scheduler with min_lr clipping.

    Args:
        step (int | list[int]): Step to decay the LR. If an int value is given,
            regard it as the decay interval. If a list is given, decay LR at
            these steps.
        gamma (float): Decay LR ratio. Defaults to 0.1.
        min_lr (float, optional): Minimum LR value to keep. If LR after decay
            is lower than `min_lr`, it will be clipped to this value. If None
            is given, we don't perform lr clipping. Default: None.
    """

    def __init__(self,
                 step: Union[int, List[int]],
                 gamma: float = 0.1,
                 min_lr: Optional[float] = None,
                 **kwargs) -> None:
        if isinstance(step, list):
            assert mmcv.is_list_of(step, int)
            assert all([s > 0 for s in step])
        elif isinstance(step, int):
            assert step > 0
        else:
            raise TypeError('"step" must be a list or integer')
        self.step = step
        self.gamma = gamma
        self.min_lr = min_lr
        super().__init__(**kwargs)

    def get_lr(self, runner: 'runner.BaseRunner', base_lr: float):
        progress = runner.epoch if self.by_epoch else runner.iter

        # calculate exponential term
        if isinstance(self.step, int):
            exp = progress // self.step
        else:
            exp = len(self.step)
            for i, s in enumerate(self.step):
                if progress < s:
                    exp = i
                    break

        lr = base_lr * (self.gamma**exp)
        if self.min_lr is not None:
            # clip to a minimum value
            lr = max(lr, self.min_lr)
        return lr


@HOOKS.register_module()
class ExpLrUpdaterHook(LrUpdaterHook):

    def __init__(self, gamma: float, **kwargs) -> None:
        self.gamma = gamma
        super().__init__(**kwargs)

    def get_lr(self, runner: 'runner.BaseRunner', base_lr: float):
        progress = runner.epoch if self.by_epoch else runner.iter
        return base_lr * self.gamma**progress


@HOOKS.register_module()
class PolyLrUpdaterHook(LrUpdaterHook):

    def __init__(self,
                 power: float = 1.,
                 min_lr: float = 0.,
                 **kwargs) -> None:
        self.power = power
        self.min_lr = min_lr
        super().__init__(**kwargs)

    def get_lr(self, runner: 'runner.BaseRunner', base_lr: float):
        if self.by_epoch:
            progress = runner.epoch
            max_progress = runner.max_epochs
        else:
            progress = runner.iter
            max_progress = runner.max_iters
        coeff = (1 - progress / max_progress)**self.power
        return (base_lr - self.min_lr) * coeff + self.min_lr


@HOOKS.register_module()
class InvLrUpdaterHook(LrUpdaterHook):

    def __init__(self, gamma: float, power: float = 1., **kwargs) -> None:
        self.gamma = gamma
        self.power = power
        super().__init__(**kwargs)

    def get_lr(self, runner: 'runner.BaseRunner', base_lr: float):
        progress = runner.epoch if self.by_epoch else runner.iter
        return base_lr * (1 + self.gamma * progress)**(-self.power)


@HOOKS.register_module()
class CosineAnnealingLrUpdaterHook(LrUpdaterHook):
    """CosineAnnealing LR scheduler.

    Args:
        min_lr (float, optional): The minimum lr. Default: None.
        min_lr_ratio (float, optional): The ratio of minimum lr to the base lr.
            Either `min_lr` or `min_lr_ratio` should be specified.
            Default: None.
    """

    def __init__(self,
                 min_lr: Optional[float] = None,
                 min_lr_ratio: Optional[float] = None,
                 **kwargs) -> None:
        assert (min_lr is None) ^ (min_lr_ratio is None)
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        super().__init__(**kwargs)

    def get_lr(self, runner: 'runner.BaseRunner', base_lr: float):
        if self.by_epoch:
            progress = runner.epoch
            max_progress = runner.max_epochs
        else:
            progress = runner.iter
            max_progress = runner.max_iters

        if self.min_lr_ratio is not None:
            target_lr = base_lr * self.min_lr_ratio
        else:
            target_lr = self.min_lr  # type:ignore
        return annealing_cos(base_lr, target_lr, progress / max_progress)


@HOOKS.register_module()
class FlatCosineAnnealingLrUpdaterHook(LrUpdaterHook):
    """Flat + Cosine lr schedule.

    Modified from https://github.com/fastai/fastai/blob/master/fastai/callback/schedule.py#L128 # noqa: E501

    Args:
        start_percent (float): When to start annealing the learning rate
            after the percentage of the total training steps.
            The value should be in range [0, 1).
            Default: 0.75
        min_lr (float, optional): The minimum lr. Default: None.
        min_lr_ratio (float, optional): The ratio of minimum lr to the base lr.
            Either `min_lr` or `min_lr_ratio` should be specified.
            Default: None.
    """

    def __init__(self,
                 start_percent: float = 0.75,
                 min_lr: Optional[float] = None,
                 min_lr_ratio: Optional[float] = None,
                 **kwargs) -> None:
        assert (min_lr is None) ^ (min_lr_ratio is None)
        if start_percent < 0 or start_percent > 1 or not isinstance(
                start_percent, float):
            raise ValueError(
                'expected float between 0 and 1 start_percent, but '
                f'got {start_percent}')
        self.start_percent = start_percent
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        super().__init__(**kwargs)

    def get_lr(self, runner: 'runner.BaseRunner', base_lr: float):
        if self.by_epoch:
            start = round(runner.max_epochs * self.start_percent)
            progress = runner.epoch - start
            max_progress = runner.max_epochs - start
        else:
            start = round(runner.max_iters * self.start_percent)
            progress = runner.iter - start
            max_progress = runner.max_iters - start

        if self.min_lr_ratio is not None:
            target_lr = base_lr * self.min_lr_ratio
        else:
            target_lr = self.min_lr  # type:ignore

        if progress < 0:
            return base_lr
        else:
            return annealing_cos(base_lr, target_lr, progress / max_progress)


@HOOKS.register_module()
class CosineRestartLrUpdaterHook(LrUpdaterHook):
    """Cosine annealing with restarts learning rate scheme.

    Args:
        periods (list[int]): Periods for each cosine anneling cycle.
        restart_weights (list[float]): Restart weights at each
            restart iteration. Defaults to [1].
        min_lr (float, optional): The minimum lr. Default: None.
        min_lr_ratio (float, optional): The ratio of minimum lr to the base lr.
            Either `min_lr` or `min_lr_ratio` should be specified.
            Default: None.
    """

    def __init__(self,
                 periods: List[int],
                 restart_weights: List[float] = [1],
                 min_lr: Optional[float] = None,
                 min_lr_ratio: Optional[float] = None,
                 **kwargs) -> None:
        assert (min_lr is None) ^ (min_lr_ratio is None)
        self.periods = periods
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        self.restart_weights = restart_weights
        assert (len(self.periods) == len(self.restart_weights)
                ), 'periods and restart_weights should have the same length.'
        super().__init__(**kwargs)

        self.cumulative_periods = [
            sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))
        ]

    def get_lr(self, runner: 'runner.BaseRunner', base_lr: float):
        if self.by_epoch:
            progress = runner.epoch
        else:
            progress = runner.iter

        if self.min_lr_ratio is not None:
            target_lr = base_lr * self.min_lr_ratio
        else:
            target_lr = self.min_lr  # type:ignore

        idx = get_position_from_periods(progress, self.cumulative_periods)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_periods[idx - 1]
        current_periods = self.periods[idx]

        alpha = min((progress - nearest_restart) / current_periods, 1)
        return annealing_cos(base_lr, target_lr, alpha, current_weight)


def get_position_from_periods(iteration: int, cumulative_periods: List[int]):
    """Get the position from a period list.

    It will return the index of the right-closest number in the period list.
    For example, the cumulative_periods = [100, 200, 300, 400],
    if iteration == 50, return 0;
    if iteration == 210, return 2;
    if iteration == 300, return 3.

    Args:
        iteration (int): Current iteration.
        cumulative_periods (list[int]): Cumulative period list.

    Returns:
        int: The position of the right-closest number in the period list.
    """
    for i, period in enumerate(cumulative_periods):
        if iteration < period:
            return i
    raise ValueError(f'Current iteration {iteration} exceeds '
                     f'cumulative_periods {cumulative_periods}')


@HOOKS.register_module()
class CyclicLrUpdaterHook(LrUpdaterHook):
    """Cyclic LR Scheduler.

    Implement the cyclical learning rate policy (CLR) described in
    https://arxiv.org/pdf/1506.01186.pdf

    Different from the original paper, we use cosine annealing rather than
    triangular policy inside a cycle. This improves the performance in the
    3D detection area.

    Args:
        by_epoch (bool, optional): Whether to update LR by epoch.
        target_ratio (tuple[float], optional): Relative ratio of the highest LR
            and the lowest LR to the initial LR.
        cyclic_times (int, optional): Number of cycles during training
        step_ratio_up (float, optional): The ratio of the increasing process of
            LR in the total cycle.
        anneal_strategy (str, optional): {'cos', 'linear'}
            Specifies the annealing strategy: 'cos' for cosine annealing,
            'linear' for linear annealing. Default: 'cos'.
        gamma (float, optional): Cycle decay ratio. Default: 1.
            It takes values in the range (0, 1]. The difference between the
            maximum learning rate and the minimum learning rate decreases
            periodically when it is less than 1. `New in version 1.4.4.`
    """

    def __init__(self,
                 by_epoch: bool = False,
                 target_ratio: Union[float, tuple] = (10, 1e-4),
                 cyclic_times: int = 1,
                 step_ratio_up: float = 0.4,
                 anneal_strategy: str = 'cos',
                 gamma: float = 1,
                 **kwargs) -> None:
        if isinstance(target_ratio, float):
            target_ratio = (target_ratio, target_ratio / 1e5)
        elif isinstance(target_ratio, tuple):
            target_ratio = (target_ratio[0], target_ratio[0] / 1e5) \
                if len(target_ratio) == 1 else target_ratio
        else:
            raise ValueError('target_ratio should be either float '
                             f'or tuple, got {type(target_ratio)}')

        assert len(target_ratio) == 2, \
            '"target_ratio" must be list or tuple of two floats'
        assert 0 <= step_ratio_up < 1.0, \
            '"step_ratio_up" must be in range [0,1)'
        assert 0 < gamma <= 1, \
            '"gamma" must be in range (0, 1]'

        self.target_ratio = target_ratio
        self.cyclic_times = cyclic_times
        self.step_ratio_up = step_ratio_up
        self.gamma = gamma
        self.max_iter_per_phase = None
        self.lr_phases: list = []  # init lr_phases
        # validate anneal_strategy
        if anneal_strategy not in ['cos', 'linear']:
            raise ValueError('anneal_strategy must be one of "cos" or '
                             f'"linear", instead got {anneal_strategy}')
        elif anneal_strategy == 'cos':
            self.anneal_func: Callable[[float, float, float],
                                       float] = annealing_cos
        elif anneal_strategy == 'linear':
            self.anneal_func = annealing_linear

        assert not by_epoch, \
            'currently only support "by_epoch" = False'
        super().__init__(by_epoch, **kwargs)

    def before_run(self, runner: 'runner.BaseRunner'):
        super().before_run(runner)
        # initiate lr_phases
        # total lr_phases are separated as up and down
        self.max_iter_per_phase = runner.max_iters // self.cyclic_times
        iter_up_phase = int(self.step_ratio_up *
                            self.max_iter_per_phase)  # type: ignore
        self.lr_phases.append([0, iter_up_phase, 1, self.target_ratio[0]])
        self.lr_phases.append([
            iter_up_phase, self.max_iter_per_phase, self.target_ratio[0],
            self.target_ratio[1]
        ])

    def get_lr(self, runner: 'runner.BaseRunner', base_lr: float):
        curr_iter = runner.iter % self.max_iter_per_phase  # type: ignore
        curr_cycle = runner.iter // self.max_iter_per_phase  # type: ignore
        # Update weight decay
        scale = self.gamma**curr_cycle

        for (start_iter, end_iter, start_ratio, end_ratio) in self.lr_phases:
            if start_iter <= curr_iter < end_iter:
                # Apply cycle scaling to gradually reduce the difference
                # between max_lr and base lr. The target end_ratio can be
                # expressed as:
                # end_ratio = (base_lr + scale * (max_lr - base_lr)) / base_lr
                # iteration: 0-iter_up_phase:
                if start_iter == 0:
                    end_ratio = 1 - scale + end_ratio * scale
                # iteration: iter_up_phase-self.max_iter_per_phase
                else:
                    start_ratio = 1 - scale + start_ratio * scale
                progress = curr_iter - start_iter
                return self.anneal_func(base_lr * start_ratio,
                                        base_lr * end_ratio,
                                        progress / (end_iter - start_iter))


@HOOKS.register_module()
class OneCycleLrUpdaterHook(LrUpdaterHook):
    """One Cycle LR Scheduler.

    The 1cycle learning rate policy changes the learning rate after every
    batch. The one cycle learning rate policy is described in
    https://arxiv.org/pdf/1708.07120.pdf

    Args:
        max_lr (float or list): Upper learning rate boundaries in the cycle
            for each parameter group.
        total_steps (int, optional): The total number of steps in the cycle.
            Note that if a value is not provided here, it will be the max_iter
            of runner. Default: None.
        pct_start (float): The percentage of the cycle (in number of steps)
            spent increasing the learning rate.
            Default: 0.3
        anneal_strategy (str): {'cos', 'linear'}
            Specifies the annealing strategy: 'cos' for cosine annealing,
            'linear' for linear annealing.
            Default: 'cos'
        div_factor (float): Determines the initial learning rate via
            initial_lr = max_lr/div_factor
            Default: 25
        final_div_factor (float): Determines the minimum learning rate via
            min_lr = initial_lr/final_div_factor
            Default: 1e4
        three_phase (bool): If three_phase is True, use a third phase of the
            schedule to annihilate the learning rate according to
            final_div_factor instead of modifying the second phase (the first
            two phases will be symmetrical about the step indicated by
            pct_start).
            Default: False
    """

    def __init__(self,
                 max_lr: Union[float, List],
                 total_steps: Optional[int] = None,
                 pct_start: float = 0.3,
                 anneal_strategy: str = 'cos',
                 div_factor: float = 25,
                 final_div_factor: float = 1e4,
                 three_phase: bool = False,
                 **kwargs) -> None:
        # validate by_epoch, currently only support by_epoch = False
        if 'by_epoch' not in kwargs:
            kwargs['by_epoch'] = False
        else:
            assert not kwargs['by_epoch'], \
                'currently only support "by_epoch" = False'
        if not isinstance(max_lr, (numbers.Number, list, dict)):
            raise ValueError('the type of max_lr must be the one of list or '
                             f'dict, but got {type(max_lr)}')
        self._max_lr = max_lr
        if total_steps is not None:
            if not isinstance(total_steps, int):
                raise ValueError('the type of total_steps must be int, but'
                                 f'got {type(total_steps)}')
            self.total_steps = total_steps
        # validate pct_start
        if pct_start < 0 or pct_start > 1 or not isinstance(pct_start, float):
            raise ValueError('expected float between 0 and 1 pct_start, but '
                             f'got {pct_start}')
        self.pct_start = pct_start
        # validate anneal_strategy
        if anneal_strategy not in ['cos', 'linear']:
            raise ValueError('anneal_strategy must be one of "cos" or '
                             f'"linear", instead got {anneal_strategy}')
        elif anneal_strategy == 'cos':
            self.anneal_func: Callable[[float, float, float],
                                       float] = annealing_cos
        elif anneal_strategy == 'linear':
            self.anneal_func = annealing_linear
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.three_phase = three_phase
        self.lr_phases: list = []  # init lr_phases
        super().__init__(**kwargs)

    def before_run(self, runner: 'runner.BaseRunner'):
        if hasattr(self, 'total_steps'):
            total_steps = self.total_steps
        else:
            total_steps = runner.max_iters
        if total_steps < runner.max_iters:
            raise ValueError(
                'The total steps must be greater than or equal to max '
                f'iterations {runner.max_iters} of runner, but total steps '
                f'is {total_steps}.')

        if isinstance(runner.optimizer, dict):
            self.base_lr = {}
            for k, optim in runner.optimizer.items():
                _max_lr = format_param(k, optim, self._max_lr)
                self.base_lr[k] = [lr / self.div_factor for lr in _max_lr]
                for group, lr in zip(optim.param_groups, self.base_lr[k]):
                    group.setdefault('initial_lr', lr)
        else:
            k = type(runner.optimizer).__name__
            _max_lr = format_param(k, runner.optimizer, self._max_lr)
            self.base_lr = [lr / self.div_factor for lr in _max_lr]
            optim_param_groups = runner.optimizer.param_groups  # type: ignore
            for group, lr in zip(optim_param_groups, self.base_lr):
                group.setdefault('initial_lr', lr)

        if self.three_phase:
            self.lr_phases.append(
                [float(self.pct_start * total_steps) - 1, 1, self.div_factor])
            self.lr_phases.append([
                float(2 * self.pct_start * total_steps) - 2, self.div_factor, 1
            ])
            self.lr_phases.append(
                [total_steps - 1, 1, 1 / self.final_div_factor])
        else:
            self.lr_phases.append(
                [float(self.pct_start * total_steps) - 1, 1, self.div_factor])
            self.lr_phases.append(
                [total_steps - 1, self.div_factor, 1 / self.final_div_factor])

    def get_lr(self, runner: 'runner.BaseRunner', base_lr: float):
        curr_iter = runner.iter
        start_iter = 0
        for i, (end_iter, start_lr, end_lr) in enumerate(self.lr_phases):
            if curr_iter <= end_iter:
                pct = (curr_iter - start_iter) / (end_iter - start_iter)
                lr = self.anneal_func(base_lr * start_lr, base_lr * end_lr,
                                      pct)
                break
            start_iter = end_iter
        return lr


@HOOKS.register_module()
class LinearAnnealingLrUpdaterHook(LrUpdaterHook):
    """Linear annealing LR Scheduler decays the learning rate of each parameter
    group linearly.

    Args:
        min_lr (float, optional): The minimum lr. Default: None.
        min_lr_ratio (float, optional): The ratio of minimum lr to the base lr.
            Either `min_lr` or `min_lr_ratio` should be specified.
            Default: None.
    """

    def __init__(self,
                 min_lr: Optional[float] = None,
                 min_lr_ratio: Optional[float] = None,
                 **kwargs):
        assert (min_lr is None) ^ (min_lr_ratio is None)
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        super().__init__(**kwargs)

    def get_lr(self, runner: 'runner.BaseRunner', base_lr: float):
        if self.by_epoch:
            progress = runner.epoch
            max_progress = runner.max_epochs
        else:
            progress = runner.iter
            max_progress = runner.max_iters
        if self.min_lr_ratio is not None:
            target_lr = base_lr * self.min_lr_ratio
        else:
            target_lr = self.min_lr  # type:ignore
        return annealing_linear(base_lr, target_lr, progress / max_progress)


def annealing_cos(start: float,
                  end: float,
                  factor: float,
                  weight: float = 1.) -> float:
    """Calculate annealing cos learning rate.

    Cosine anneal from `weight * start + (1 - weight) * end` to `end` as
    percentage goes from 0.0 to 1.0.

    Args:
        start (float): The starting learning rate of the cosine annealing.
        end (float): The ending learing rate of the cosine annealing.
        factor (float): The coefficient of `pi` when calculating the current
            percentage. Range from 0.0 to 1.0.
        weight (float, optional): The combination factor of `start` and `end`
            when calculating the actual starting learning rate. Default to 1.
    """
    cos_out = cos(pi * factor) + 1
    return end + 0.5 * weight * (start - end) * cos_out


def annealing_linear(start: float, end: float, factor: float) -> float:
    """Calculate annealing linear learning rate.

    Linear anneal from `start` to `end` as percentage goes from 0.0 to 1.0.

    Args:
        start (float): The starting learning rate of the linear annealing.
        end (float): The ending learing rate of the linear annealing.
        factor (float): The coefficient of `pi` when calculating the current
            percentage. Range from 0.0 to 1.0.
    """
    return start + (end - start) * factor


def format_param(name, optim, param):
    if isinstance(param, numbers.Number):
        return [param] * len(optim.param_groups)
    elif isinstance(param, (list, tuple)):  # multi param groups
        if len(param) != len(optim.param_groups):
            raise ValueError(f'expected {len(optim.param_groups)} '
                             f'values for {name}, got {len(param)}')
        return param
    else:  # multi optimizers
        if name not in param:
            raise KeyError(f'{name} is not found in {param.keys()}')
        return param[name]
