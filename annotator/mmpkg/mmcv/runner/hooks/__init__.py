# Copyright (c) OpenMMLab. All rights reserved.
from .checkpoint import CheckpointHook
from .closure import ClosureHook
from .ema import EMAHook
from .evaluation import DistEvalHook, EvalHook
from .hook import HOOKS, Hook
from .iter_timer import IterTimerHook
from .logger import (ClearMLLoggerHook, DvcliveLoggerHook, LoggerHook,
                     MlflowLoggerHook, NeptuneLoggerHook, PaviLoggerHook,
                     SegmindLoggerHook, TensorboardLoggerHook, TextLoggerHook,
                     WandbLoggerHook)
from .lr_updater import (CosineAnnealingLrUpdaterHook,
                         CosineRestartLrUpdaterHook, CyclicLrUpdaterHook,
                         ExpLrUpdaterHook, FixedLrUpdaterHook,
                         FlatCosineAnnealingLrUpdaterHook, InvLrUpdaterHook,
                         LinearAnnealingLrUpdaterHook, LrUpdaterHook,
                         OneCycleLrUpdaterHook, PolyLrUpdaterHook,
                         StepLrUpdaterHook)
from .memory import EmptyCacheHook
from .momentum_updater import (CosineAnnealingMomentumUpdaterHook,
                               CyclicMomentumUpdaterHook,
                               LinearAnnealingMomentumUpdaterHook,
                               MomentumUpdaterHook,
                               OneCycleMomentumUpdaterHook,
                               StepMomentumUpdaterHook)
from .optimizer import (Fp16OptimizerHook, GradientCumulativeFp16OptimizerHook,
                        GradientCumulativeOptimizerHook, OptimizerHook)
from .profiler import ProfilerHook
from .sampler_seed import DistSamplerSeedHook
from .sync_buffer import SyncBuffersHook

__all__ = [
    'HOOKS', 'Hook', 'CheckpointHook', 'ClosureHook', 'LrUpdaterHook',
    'FixedLrUpdaterHook', 'StepLrUpdaterHook', 'ExpLrUpdaterHook',
    'PolyLrUpdaterHook', 'InvLrUpdaterHook', 'CosineAnnealingLrUpdaterHook',
    'FlatCosineAnnealingLrUpdaterHook', 'CosineRestartLrUpdaterHook',
    'CyclicLrUpdaterHook', 'OneCycleLrUpdaterHook', 'OptimizerHook',
    'Fp16OptimizerHook', 'IterTimerHook', 'DistSamplerSeedHook',
    'EmptyCacheHook', 'LoggerHook', 'MlflowLoggerHook', 'PaviLoggerHook',
    'TextLoggerHook', 'TensorboardLoggerHook', 'NeptuneLoggerHook',
    'WandbLoggerHook', 'DvcliveLoggerHook', 'MomentumUpdaterHook',
    'StepMomentumUpdaterHook', 'CosineAnnealingMomentumUpdaterHook',
    'CyclicMomentumUpdaterHook', 'OneCycleMomentumUpdaterHook',
    'SyncBuffersHook', 'EMAHook', 'EvalHook', 'DistEvalHook', 'ProfilerHook',
    'GradientCumulativeOptimizerHook', 'GradientCumulativeFp16OptimizerHook',
    'SegmindLoggerHook', 'LinearAnnealingLrUpdaterHook',
    'LinearAnnealingMomentumUpdaterHook', 'ClearMLLoggerHook'
]
