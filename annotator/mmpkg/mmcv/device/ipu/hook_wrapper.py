# Copyright (c) OpenMMLab. All rights reserved.
from annotator.mmpkg.mmcv.runner import HOOKS, LrUpdaterHook, OptimizerHook
from annotator.mmpkg.mmcv.utils import TORCH_VERSION, digit_version


def wrap_lr_updater_hook(lr_hook_class):
    """A wrapper function to wrap any subclass of LrUpdaterHook.

    IPU needs extra operations to upload optimizer settings. This wrapper will
    override function(_set_lr) of a subclass of LrUpdaterHook.
    """
    assert issubclass(lr_hook_class, LrUpdaterHook)

    class ipu_lr_hook_class(lr_hook_class):

        def _set_lr(self, runner, *args, **kwargs):
            super()._set_lr(runner, *args, **kwargs)
            # convert torch optimizer to poptorch optimizer
            runner.model.setOptimizer(runner.optimizer)

    return ipu_lr_hook_class


def wrap_optimizer_hook(optimizer_hook_class):
    """A wrapper function to wrap OptimizerHook.

    This is an non-intrusive implementation of wrapping optimizer hook (or you
    need to change every config file to use IPU optimizer hook) IPU's clip-norm
    implementation is different from pytorch, so there should be an error
    raised when using clip-norm.
    """

    class ipu_optimizer_hook_class(OptimizerHook):

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            if self.grad_clip is not None:
                raise NotImplementedError('IPU does not support gradient clip')

    return ipu_optimizer_hook_class


if (TORCH_VERSION != 'parrots'
        and digit_version(TORCH_VERSION) >= digit_version('1.6.0')):

    @HOOKS.register_module()
    class IPUFp16OptimizerHook(OptimizerHook):
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
                     grad_clip=None,
                     coalesce=True,
                     bucket_size_mb=-1,
                     loss_scale=512.,
                     distributed=True):
            assert grad_clip is None,\
                'IPU mode does not support `grad_clip` currently'
            assert coalesce,\
                'implemented all reduce in distributed training currently'
            assert bucket_size_mb == -1,\
                '`bucket_size_mb` should not be set in IPU mode'
            self.distributed = distributed
            self._scale_update_param = None
            if loss_scale == 'dynamic':
                raise NotImplementedError(
                    'IPU mode does not support dynamic loss scale currently')
            elif isinstance(loss_scale, float):
                self.loss_scale = loss_scale
            elif isinstance(loss_scale, dict):
                raise NotImplementedError(
                    'IPU mode supports single scale currently')
            else:
                raise ValueError(
                    f'loss_scale should be float, but got {loss_scale} ')

        def after_train_iter(self, runner):
            pass

else:
    raise RuntimeError('The IPU mode only supports torch 1.6 and above')
