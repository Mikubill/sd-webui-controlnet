# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from annotator.mmpkg.mmcv.runner import DistEvalHook as _DistEvalHook
from annotator.mmpkg.mmcv.runner import EvalHook as _EvalHook

MMPOSE_GREATER_KEYS = [
    'acc', 'ap', 'ar', 'pck', 'auc', '3dpck', 'p-3dpck', '3dauc', 'p-3dauc',
    'pcp'
]
MMPOSE_LESS_KEYS = ['loss', 'epe', 'nme', 'mpjpe', 'p-mpjpe', 'n-mpjpe']


class EvalHook(_EvalHook):

    def __init__(self,
                 dataloader,
                 start=None,
                 interval=1,
                 by_epoch=True,
                 save_best=None,
                 rule=None,
                 test_fn=None,
                 greater_keys=MMPOSE_GREATER_KEYS,
                 less_keys=MMPOSE_LESS_KEYS,
                 **eval_kwargs):

        if test_fn is None:
            from annotator.mmpkg.mmpose.apis import single_gpu_test
            test_fn = single_gpu_test

        # to be compatible with the config before v0.16.0

        # remove "gpu_collect" from eval_kwargs
        if 'gpu_collect' in eval_kwargs:
            warnings.warn(
                '"gpu_collect" will be deprecated in EvalHook.'
                'Please remove it from the config.', DeprecationWarning)
            _ = eval_kwargs.pop('gpu_collect')

        # update "save_best" according to "key_indicator" and remove the
        # latter from eval_kwargs
        if 'key_indicator' in eval_kwargs or isinstance(save_best, bool):
            warnings.warn(
                '"key_indicator" will be deprecated in EvalHook.'
                'Please use "save_best" to specify the metric key,'
                'e.g., save_best="AP".', DeprecationWarning)

            key_indicator = eval_kwargs.pop('key_indicator', 'AP')
            if save_best is True and key_indicator is None:
                raise ValueError('key_indicator should not be None, when '
                                 'save_best is set to True.')
            save_best = key_indicator

        super().__init__(dataloader, start, interval, by_epoch, save_best,
                         rule, test_fn, greater_keys, less_keys, **eval_kwargs)


class DistEvalHook(_DistEvalHook):

    def __init__(self,
                 dataloader,
                 start=None,
                 interval=1,
                 by_epoch=True,
                 save_best=None,
                 rule=None,
                 test_fn=None,
                 greater_keys=MMPOSE_GREATER_KEYS,
                 less_keys=MMPOSE_LESS_KEYS,
                 broadcast_bn_buffer=True,
                 tmpdir=None,
                 gpu_collect=False,
                 **eval_kwargs):

        if test_fn is None:
            from annotator.mmpkg.mmpose.apis import multi_gpu_test
            test_fn = multi_gpu_test

        # to be compatible with the config before v0.16.0

        # update "save_best" according to "key_indicator" and remove the
        # latter from eval_kwargs
        if 'key_indicator' in eval_kwargs or isinstance(save_best, bool):
            warnings.warn(
                '"key_indicator" will be deprecated in EvalHook.'
                'Please use "save_best" to specify the metric key,'
                'e.g., save_best="AP".', DeprecationWarning)

            key_indicator = eval_kwargs.pop('key_indicator', 'AP')
            if save_best is True and key_indicator is None:
                raise ValueError('key_indicator should not be None, when '
                                 'save_best is set to True.')
            save_best = key_indicator

        super().__init__(dataloader, start, interval, by_epoch, save_best,
                         rule, test_fn, greater_keys, less_keys,
                         broadcast_bn_buffer, tmpdir, gpu_collect,
                         **eval_kwargs)
