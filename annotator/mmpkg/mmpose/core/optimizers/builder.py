# Copyright (c) OpenMMLab. All rights reserved.
from annotator.mmpkg.mmcv.runner import build_optimizer
from annotator.mmpkg.mmcv.runner.optimizer import OPTIMIZER_BUILDERS as MMCV_OPTIMIZER_BUILDERS
from annotator.mmpkg.mmcv.utils import Registry, build_from_cfg

OPTIMIZERS = Registry('optimizers')
OPTIMIZER_BUILDERS = Registry(
    'optimizer builder', parent=MMCV_OPTIMIZER_BUILDERS)


def build_optimizer_constructor(cfg):
    constructor_type = cfg.get('type')
    if constructor_type in OPTIMIZER_BUILDERS:
        return build_from_cfg(cfg, OPTIMIZER_BUILDERS)
    elif constructor_type in MMCV_OPTIMIZER_BUILDERS:
        return build_from_cfg(cfg, MMCV_OPTIMIZER_BUILDERS)
    else:
        raise KeyError(f'{constructor_type} is not registered '
                       'in the optimizer builder registry.')


def build_optimizers(model, cfgs):
    """Build multiple optimizers from configs.

    If `cfgs` contains several dicts for optimizers, then a dict for each
    constructed optimizers will be returned.
    If `cfgs` only contains one optimizer config, the constructed optimizer
    itself will be returned.

    For example,

    1) Multiple optimizer configs:

    .. code-block:: python

        optimizer_cfg = dict(
            model1=dict(type='SGD', lr=lr),
            model2=dict(type='SGD', lr=lr))

    The return dict is
    ``dict('model1': torch.optim.Optimizer, 'model2': torch.optim.Optimizer)``

    2) Single optimizer config:

    .. code-block:: python

        optimizer_cfg = dict(type='SGD', lr=lr)

    The return is ``torch.optim.Optimizer``.

    Args:
        model (:obj:`nn.Module`): The model with parameters to be optimized.
        cfgs (dict): The config dict of the optimizer.

    Returns:
        dict[:obj:`torch.optim.Optimizer`] | :obj:`torch.optim.Optimizer`:
            The initialized optimizers.
    """
    optimizers = {}
    if hasattr(model, 'module'):
        model = model.module
    # determine whether 'cfgs' has several dicts for optimizers
    if all(isinstance(v, dict) for v in cfgs.values()):
        for key, cfg in cfgs.items():
            cfg_ = cfg.copy()
            module = getattr(model, key)
            optimizers[key] = build_optimizer(module, cfg_)
        return optimizers

    return build_optimizer(model, cfgs)
