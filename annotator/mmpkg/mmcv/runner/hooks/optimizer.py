# Copyright (c) OpenMMLab. All rights reserved.
import copy
import logging
from collections import defaultdict
from itertools import chain
from typing import Optional, Union

import torch.nn as nn
from torch import Tensor
from torch.nn.utils import clip_grad

from annotator.mmpkg.mmcv.utils import (IS_NPU_AVAILABLE, TORCH_VERSION, _BatchNorm,
                        digit_version)
from ..dist_utils import allreduce_grads
from ..fp16_utils import LossScaler, wrap_fp16_model
from .hook import HOOKS, Hook

try:
    # If PyTorch version >= 1.6.0, torch.cuda.amp.GradScaler would be imported
    # and used; otherwise, auto fp16 will adopt mmcv's implementation.
    if IS_NPU_AVAILABLE:
        from torch.npu.amp import GradScaler
    else:
        from torch.cuda.amp import GradScaler
except ImportError:
    pass


@HOOKS.register_module()
class OptimizerHook(Hook):
    """A hook contains custom operations for the optimizer.

    Args:
        grad_clip (dict, optional): A config dict to control the clip_grad.
            Default: None.
        detect_anomalous_params (bool): This option is only used for
            debugging which will slow down the training speed.
            Detect anomalous parameters that are not included in
            the computational graph with `loss` as the root.
            There are two cases

                - Parameters were not used during
                  forward pass.
                - Parameters were not used to produce
                  loss.
            Default: False.
    """

    def __init__(self,
                 grad_clip: Optional[dict] = None,
                 detect_anomalous_params: bool = False):
        self.grad_clip = grad_clip
        self.detect_anomalous_params = detect_anomalous_params

    def clip_grads(self, params):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **self.grad_clip)

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        if self.detect_anomalous_params:
            self.detect_anomalous_parameters(runner.outputs['loss'], runner)
        runner.outputs['loss'].backward()

        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])
        runner.optimizer.step()

    def detect_anomalous_parameters(self, loss: Tensor, runner) -> None:
        logger = runner.logger
        parameters_in_graph = set()
        visited = set()

        def traverse(grad_fn):
            if grad_fn is None:
                return
            if grad_fn not in visited:
                visited.add(grad_fn)
                if hasattr(grad_fn, 'variable'):
                    parameters_in_graph.add(grad_fn.variable)
                parents = grad_fn.next_functions
                if parents is not None:
                    for parent in parents:
                        grad_fn = parent[0]
                        traverse(grad_fn)

        traverse(loss.grad_fn)
        for n, p in runner.model.named_parameters():
            if p not in parameters_in_graph and p.requires_grad:
                logger.log(
                    level=logging.ERROR,
                    msg=f'{n} with shape {p.size()} is not '
                    f'in the computational graph \n')


@HOOKS.register_module()
class GradientCumulativeOptimizerHook(OptimizerHook):
    """Optimizer Hook implements multi-iters gradient cumulating.

    Args:
        cumulative_iters (int, optional): Num of gradient cumulative iters.
            The optimizer will step every `cumulative_iters` iters.
            Defaults to 1.

    Examples:
        >>> # Use cumulative_iters to simulate a large batch size
        >>> # It is helpful when the hardware cannot handle a large batch size.
        >>> loader = DataLoader(data, batch_size=64)
        >>> optim_hook = GradientCumulativeOptimizerHook(cumulative_iters=4)
        >>> # almost equals to
        >>> loader = DataLoader(data, batch_size=256)
        >>> optim_hook = OptimizerHook()
    """

    def __init__(self, cumulative_iters: int = 1, **kwargs):
        super().__init__(**kwargs)

        assert isinstance(cumulative_iters, int) and cumulative_iters > 0, \
            f'cumulative_iters only accepts positive int, but got ' \
            f'{type(cumulative_iters)} instead.'

        self.cumulative_iters = cumulative_iters
        self.divisible_iters = 0
        self.remainder_iters = 0
        self.initialized = False

    def has_batch_norm(self, module: nn.Module) -> bool:
        if isinstance(module, _BatchNorm):
            return True
        for m in module.children():
            if self.has_batch_norm(m):
                return True
        return False

    def _init(self, runner):
        if runner.iter % self.cumulative_iters != 0:
            runner.logger.warning(
                'Resume iter number is not divisible by cumulative_iters in '
                'GradientCumulativeOptimizerHook, which means the gradient of '
                'some iters is lost and the result may be influenced slightly.'
            )

        if self.has_batch_norm(runner.model) and self.cumulative_iters > 1:
            runner.logger.warning(
                'GradientCumulativeOptimizerHook may slightly decrease '
                'performance if the model has BatchNorm layers.')

        self.divisible_iters = (
            runner.max_iters // self.cumulative_iters * self.cumulative_iters)
        self.remainder_iters = runner.max_iters - self.divisible_iters

        self.initialized = True

    def _get_loss_factor(self, runner):
        """Get loss division factor for the current iteration."""
        if runner.iter < runner.max_iters - self.remainder_iters:
            loss_factor = self.cumulative_iters
        else:
            loss_factor = self.remainder_iters
            runner.logger.warning(
                f'Loss will be divided by {loss_factor} in the last '
                f'{self.remainder_iters} iterations because they are not '
                f'enough for {self.cumulative_iters} cumulative_iters.')
            assert loss_factor > 0

        return loss_factor

    def after_train_iter(self, runner):
        if not self.initialized:
            self._init(runner)

        loss = runner.outputs['loss'] / self._get_loss_factor(runner)
        loss.backward()

        if (self.every_n_iters(runner, self.cumulative_iters)
                or self.is_last_iter(runner)):

            if self.grad_clip is not None:
                grad_norm = self.clip_grads(runner.model.parameters())
                if grad_norm is not None:
                    # Add grad norm to the logger
                    runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                             runner.outputs['num_samples'])
            runner.optimizer.step()
            runner.optimizer.zero_grad()


if (TORCH_VERSION != 'parrots'
        and digit_version(TORCH_VERSION) >= digit_version('1.6.0')):

    @HOOKS.register_module()
    class Fp16OptimizerHook(OptimizerHook):
        """FP16 optimizer hook (using PyTorch's implementation).

        If you are using PyTorch >= 1.6, torch.cuda.amp is used as the backend,
        to take care of the optimization procedure.

        Args:
            loss_scale (float | str | dict): Scale factor configuration.
                If loss_scale is a float, static loss scaling will be used with
                the specified scale. If loss_scale is a string, it must be
                'dynamic', then dynamic loss scaling will be used.
                It can also be a dict containing arguments of GradScalar.
                Defaults to 512. For Pytorch >= 1.6, mmcv uses official
                implementation of GradScaler. If you use a dict version of
                loss_scale to create GradScaler, please refer to:
                https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler
                for the parameters.

        Examples:
            >>> loss_scale = dict(
            ...     init_scale=65536.0,
            ...     growth_factor=2.0,
            ...     backoff_factor=0.5,
            ...     growth_interval=2000
            ... )
            >>> optimizer_hook = Fp16OptimizerHook(loss_scale=loss_scale)
        """

        def __init__(self,
                     grad_clip: Optional[dict] = None,
                     coalesce: bool = True,
                     bucket_size_mb: int = -1,
                     loss_scale: Union[float, str, dict] = 512.,
                     distributed: bool = True):
            self.grad_clip = grad_clip
            self.coalesce = coalesce
            self.bucket_size_mb = bucket_size_mb
            self.distributed = distributed
            self._scale_update_param = None
            if loss_scale == 'dynamic':
                self.loss_scaler = GradScaler()
            elif isinstance(loss_scale, float):
                self._scale_update_param = loss_scale
                self.loss_scaler = GradScaler(init_scale=loss_scale)
            elif isinstance(loss_scale, dict):
                self.loss_scaler = GradScaler(**loss_scale)
            else:
                raise ValueError('loss_scale must be of type float, dict, or '
                                 f'"dynamic", got {loss_scale}')

        def before_run(self, runner) -> None:
            """Preparing steps before Mixed Precision Training."""
            # wrap model mode to fp16
            wrap_fp16_model(runner.model)
            # resume from state dict
            if 'fp16' in runner.meta and 'loss_scaler' in runner.meta['fp16']:
                scaler_state_dict = runner.meta['fp16']['loss_scaler']
                self.loss_scaler.load_state_dict(scaler_state_dict)

        def copy_grads_to_fp32(self, fp16_net: nn.Module,
                               fp32_weights: Tensor) -> None:
            """Copy gradients from fp16 model to fp32 weight copy."""
            for fp32_param, fp16_param in zip(fp32_weights,
                                              fp16_net.parameters()):
                if fp16_param.grad is not None:
                    if fp32_param.grad is None:
                        fp32_param.grad = fp32_param.data.new(
                            fp32_param.size())
                    fp32_param.grad.copy_(fp16_param.grad)

        def copy_params_to_fp16(self, fp16_net: nn.Module,
                                fp32_weights: Tensor) -> None:
            """Copy updated params from fp32 weight copy to fp16 model."""
            for fp16_param, fp32_param in zip(fp16_net.parameters(),
                                              fp32_weights):
                fp16_param.data.copy_(fp32_param.data)

        def after_train_iter(self, runner) -> None:
            """Backward optimization steps for Mixed Precision Training. For
            dynamic loss scaling, please refer to
            https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler.

            1. Scale the loss by a scale factor.
            2. Backward the loss to obtain the gradients.
            3. Unscale the optimizer’s gradient tensors.
            4. Call optimizer.step() and update scale factor.
            5. Save loss_scaler state_dict for resume purpose.
            """
            # clear grads of last iteration
            runner.model.zero_grad()
            runner.optimizer.zero_grad()

            self.loss_scaler.scale(runner.outputs['loss']).backward()
            self.loss_scaler.unscale_(runner.optimizer)
            # grad clip
            if self.grad_clip is not None:
                grad_norm = self.clip_grads(runner.model.parameters())
                if grad_norm is not None:
                    # Add grad norm to the logger
                    runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                             runner.outputs['num_samples'])
            # backward and update scaler
            self.loss_scaler.step(runner.optimizer)
            self.loss_scaler.update(self._scale_update_param)

            # save state_dict of loss_scaler
            runner.meta.setdefault(
                'fp16', {})['loss_scaler'] = self.loss_scaler.state_dict()

    @HOOKS.register_module()
    class GradientCumulativeFp16OptimizerHook(GradientCumulativeOptimizerHook,
                                              Fp16OptimizerHook):
        """Fp16 optimizer Hook (using PyTorch's implementation) implements
        multi-iters gradient cumulating.

        If you are using PyTorch >= 1.6, torch.cuda.amp is used as the backend,
        to take care of the optimization procedure.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def after_train_iter(self, runner) -> None:
            if not self.initialized:
                self._init(runner)

            loss = runner.outputs['loss'] / self._get_loss_factor(runner)
            self.loss_scaler.scale(loss).backward()

            if (self.every_n_iters(runner, self.cumulative_iters)
                    or self.is_last_iter(runner)):

                # copy fp16 grads in the model to fp32 params in the optimizer
                self.loss_scaler.unscale_(runner.optimizer)

                if self.grad_clip is not None:
                    grad_norm = self.clip_grads(runner.model.parameters())
                    if grad_norm is not None:
                        # Add grad norm to the logger
                        runner.log_buffer.update(
                            {'grad_norm': float(grad_norm)},
                            runner.outputs['num_samples'])

                # backward and update scaler
                self.loss_scaler.step(runner.optimizer)
                self.loss_scaler.update(self._scale_update_param)

                # save state_dict of loss_scaler
                runner.meta.setdefault(
                    'fp16', {})['loss_scaler'] = self.loss_scaler.state_dict()

                # clear grads
                runner.model.zero_grad()
                runner.optimizer.zero_grad()

else:

    @HOOKS.register_module()
    class Fp16OptimizerHook(OptimizerHook):  # type: ignore
        """FP16 optimizer hook (mmcv's implementation).

        The steps of fp16 optimizer is as follows.
        1. Scale the loss value.
        2. BP in the fp16 model.
        2. Copy gradients from fp16 model to fp32 weights.
        3. Update fp32 weights.
        4. Copy updated parameters from fp32 weights to fp16 model.

        Refer to https://arxiv.org/abs/1710.03740 for more details.

        Args:
            loss_scale (float | str | dict): Scale factor configuration.
                If loss_scale is a float, static loss scaling will be used with
                the specified scale. If loss_scale is a string, it must be
                'dynamic', then dynamic loss scaling will be used.
                It can also be a dict containing arguments of LossScaler.
                Defaults to 512.
        """

        def __init__(self,
                     grad_clip: Optional[dict] = None,
                     coalesce: bool = True,
                     bucket_size_mb: int = -1,
                     loss_scale: Union[float, str, dict] = 512.,
                     distributed: bool = True):
            self.grad_clip = grad_clip
            self.coalesce = coalesce
            self.bucket_size_mb = bucket_size_mb
            self.distributed = distributed
            if loss_scale == 'dynamic':
                self.loss_scaler = LossScaler(mode='dynamic')
            elif isinstance(loss_scale, float):
                self.loss_scaler = LossScaler(
                    init_scale=loss_scale, mode='static')
            elif isinstance(loss_scale, dict):
                self.loss_scaler = LossScaler(**loss_scale)
            else:
                raise ValueError('loss_scale must be of type float, dict, or '
                                 f'"dynamic", got {loss_scale}')

        def before_run(self, runner) -> None:
            """Preparing steps before Mixed Precision Training.

            1. Make a master copy of fp32 weights for optimization.
            2. Convert the main model from fp32 to fp16.
            """
            # keep a copy of fp32 weights
            old_groups = runner.optimizer.param_groups
            runner.optimizer.param_groups = copy.deepcopy(
                runner.optimizer.param_groups)
            state: defaultdict = defaultdict(dict)
            p_map = {
                old_p: p
                for old_p, p in zip(
                    chain(*(g['params'] for g in old_groups)),
                    chain(*(g['params']
                            for g in runner.optimizer.param_groups)))
            }
            for k, v in runner.optimizer.state.items():
                state[p_map[k]] = v
            runner.optimizer.state = state
            # convert model to fp16
            wrap_fp16_model(runner.model)
            # resume from state dict
            if 'fp16' in runner.meta and 'loss_scaler' in runner.meta['fp16']:
                scaler_state_dict = runner.meta['fp16']['loss_scaler']
                self.loss_scaler.load_state_dict(scaler_state_dict)

        def copy_grads_to_fp32(self, fp16_net: nn.Module,
                               fp32_weights: Tensor) -> None:
            """Copy gradients from fp16 model to fp32 weight copy."""
            for fp32_param, fp16_param in zip(fp32_weights,
                                              fp16_net.parameters()):
                if fp16_param.grad is not None:
                    if fp32_param.grad is None:
                        fp32_param.grad = fp32_param.data.new(
                            fp32_param.size())
                    fp32_param.grad.copy_(fp16_param.grad)

        def copy_params_to_fp16(self, fp16_net: nn.Module,
                                fp32_weights: Tensor) -> None:
            """Copy updated params from fp32 weight copy to fp16 model."""
            for fp16_param, fp32_param in zip(fp16_net.parameters(),
                                              fp32_weights):
                fp16_param.data.copy_(fp32_param.data)

        def after_train_iter(self, runner) -> None:
            """Backward optimization steps for Mixed Precision Training. For
            dynamic loss scaling, please refer `loss_scalar.py`

            1. Scale the loss by a scale factor.
            2. Backward the loss to obtain the gradients (fp16).
            3. Copy gradients from the model to the fp32 weight copy.
            4. Scale the gradients back and update the fp32 weight copy.
            5. Copy back the params from fp32 weight copy to the fp16 model.
            6. Save loss_scaler state_dict for resume purpose.
            """
            # clear grads of last iteration
            runner.model.zero_grad()
            runner.optimizer.zero_grad()
            # scale the loss value
            scaled_loss = runner.outputs['loss'] * self.loss_scaler.loss_scale
            scaled_loss.backward()
            # copy fp16 grads in the model to fp32 params in the optimizer

            fp32_weights = []
            for param_group in runner.optimizer.param_groups:
                fp32_weights += param_group['params']
            self.copy_grads_to_fp32(runner.model, fp32_weights)
            # allreduce grads
            if self.distributed:
                allreduce_grads(fp32_weights, self.coalesce,
                                self.bucket_size_mb)

            has_overflow = self.loss_scaler.has_overflow(fp32_weights)
            # if has overflow, skip this iteration
            if not has_overflow:
                # scale the gradients back
                for param in fp32_weights:
                    if param.grad is not None:
                        param.grad.div_(self.loss_scaler.loss_scale)
                if self.grad_clip is not None:
                    grad_norm = self.clip_grads(fp32_weights)
                    if grad_norm is not None:
                        # Add grad norm to the logger
                        runner.log_buffer.update(
                            {'grad_norm': float(grad_norm)},
                            runner.outputs['num_samples'])
                # update fp32 params
                runner.optimizer.step()
                # copy fp32 params to the fp16 model
                self.copy_params_to_fp16(runner.model, fp32_weights)
            self.loss_scaler.update_scale(has_overflow)
            if has_overflow:
                runner.logger.warning('Check overflow, downscale loss scale '
                                      f'to {self.loss_scaler.cur_scale}')

            # save state_dict of loss_scaler
            runner.meta.setdefault(
                'fp16', {})['loss_scaler'] = self.loss_scaler.state_dict()

    @HOOKS.register_module()
    class GradientCumulativeFp16OptimizerHook(  # type: ignore
            GradientCumulativeOptimizerHook, Fp16OptimizerHook):
        """Fp16 optimizer Hook (using mmcv implementation) implements multi-
        iters gradient cumulating."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def after_train_iter(self, runner) -> None:
            if not self.initialized:
                self._init(runner)

            loss = runner.outputs['loss'] / self._get_loss_factor(runner)
            scaled_loss = loss * self.loss_scaler.loss_scale
            scaled_loss.backward()

            if (self.every_n_iters(runner, self.cumulative_iters)
                    or self.is_last_iter(runner)):

                # copy fp16 grads in the model to fp32 params in the optimizer
                fp32_weights = []
                for param_group in runner.optimizer.param_groups:
                    fp32_weights += param_group['params']
                self.copy_grads_to_fp32(runner.model, fp32_weights)
                # allreduce grads
                if self.distributed:
                    allreduce_grads(fp32_weights, self.coalesce,
                                    self.bucket_size_mb)

                has_overflow = self.loss_scaler.has_overflow(fp32_weights)
                # if has overflow, skip this iteration
                if not has_overflow:
                    # scale the gradients back
                    for param in fp32_weights:
                        if param.grad is not None:
                            param.grad.div_(self.loss_scaler.loss_scale)
                    if self.grad_clip is not None:
                        grad_norm = self.clip_grads(fp32_weights)
                        if grad_norm is not None:
                            # Add grad norm to the logger
                            runner.log_buffer.update(
                                {'grad_norm': float(grad_norm)},
                                runner.outputs['num_samples'])
                    # update fp32 params
                    runner.optimizer.step()
                    # copy fp32 params to the fp16 model
                    self.copy_params_to_fp16(runner.model, fp32_weights)
                else:
                    runner.logger.warning(
                        'Check overflow, downscale loss scale '
                        f'to {self.loss_scaler.cur_scale}')

                self.loss_scaler.update_scale(has_overflow)

                # save state_dict of loss_scaler
                runner.meta.setdefault(
                    'fp16', {})['loss_scaler'] = self.loss_scaler.state_dict()

                # clear grads
                runner.model.zero_grad()
                runner.optimizer.zero_grad()
