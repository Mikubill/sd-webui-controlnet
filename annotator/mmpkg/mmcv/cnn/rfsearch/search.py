# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Dict, Optional

import torch  # noqa
import torch.nn as nn

import annotator.mmpkg.mmcv as mmcv
from annotator.mmpkg.mmcv.cnn.rfsearch.utils import get_single_padding, write_to_json
from annotator.mmpkg.mmcv.runner import HOOKS, Hook
from annotator.mmpkg.mmcv.utils import get_logger
from .operator import BaseConvRFSearchOp, Conv2dRFSearchOp  # noqa

logger = get_logger('mmcv')


@HOOKS.register_module()
class RFSearchHook(Hook):
    """Rcecptive field search via dilation rates.

    Please refer to `RF-Next: Efficient Receptive Field
    Search for Convolutional Neural Networks
    <https://arxiv.org/abs/2206.06637>`_ for more details.


    Args:
        mode (str, optional): It can be set to the following types:
            'search', 'fixed_single_branch', or 'fixed_multi_branch'.
            Defaults to 'search'.
        config (Dict, optional): config dict of search.
            By default this config contains "search",
            and config["search"] must include:

            - "step": recording the current searching step.
            - "max_step": The maximum number of searching steps
              to update the structures.
            - "search_interval": The interval (epoch/iteration)
              between two updates.
            - "exp_rate": The controller of the sparsity of search space.
            - "init_alphas": The value for initializing weights of each branch.
            - "mmin": The minimum dilation rate.
            - "mmax": The maximum dilation rate.
            - "num_branches": The controller of the size of
              search space (the number of branches).
            - "skip_layer": The modules in skip_layer will be ignored
              during the receptive field search.
        rfstructure_file (str, optional): Path to load searched receptive
            fields of the model. Defaults to None.
        by_epoch (bool, optional): Determine to perform step by epoch or
            by iteration. If set to True, it will step by epoch. Otherwise, by
            iteration. Defaults to True.
        verbose (bool): Determines whether to print rf-next related logging
            messages. Defaults to True.
    """

    def __init__(self,
                 mode: str = 'search',
                 config: Dict = {},
                 rfstructure_file: Optional[str] = None,
                 by_epoch: bool = True,
                 verbose: bool = True):
        assert mode in ['search', 'fixed_single_branch', 'fixed_multi_branch']
        assert config is not None
        self.config = config
        self.config['structure'] = {}
        self.verbose = verbose
        if rfstructure_file is not None:
            rfstructure = mmcv.load(rfstructure_file)['structure']
            self.config['structure'] = rfstructure
        self.mode = mode
        self.num_branches = self.config['search']['num_branches']
        self.by_epoch = by_epoch

    def init_model(self, model: nn.Module):
        """init model with search ability.

        Args:
            model (nn.Module): pytorch model

        Raises:
            NotImplementedError: only support three modes:
                search/fixed_single_branch/fixed_multi_branch
        """
        if self.verbose:
            logger.info('RFSearch init begin.')
        if self.mode == 'search':
            if self.config['structure']:
                self.set_model(model, search_op='Conv2d')
            self.wrap_model(model, search_op='Conv2d')
        elif self.mode == 'fixed_single_branch':
            self.set_model(model, search_op='Conv2d')
        elif self.mode == 'fixed_multi_branch':
            self.set_model(model, search_op='Conv2d')
            self.wrap_model(model, search_op='Conv2d')
        else:
            raise NotImplementedError
        if self.verbose:
            logger.info('RFSearch init end.')

    def after_train_epoch(self, runner):
        """Performs a dilation searching step after one training epoch."""
        if self.by_epoch and self.mode == 'search':
            self.step(runner.model, runner.work_dir)

    def after_train_iter(self, runner):
        """Performs a dilation searching step after one training iteration."""
        if not self.by_epoch and self.mode == 'search':
            self.step(runner.model, runner.work_dir)

    def step(self, model: nn.Module, work_dir: str):
        """Performs a dilation searching step.

        Args:
            model (nn.Module): pytorch model
            work_dir (str): Directory to save the searching results.
        """
        self.config['search']['step'] += 1
        if (self.config['search']['step']
            ) % self.config['search']['search_interval'] == 0 and (self.config[
                'search']['step']) < self.config['search']['max_step']:
            self.estimate_and_expand(model)
            for name, module in model.named_modules():
                if isinstance(module, BaseConvRFSearchOp):
                    self.config['structure'][name] = module.op_layer.dilation

            write_to_json(
                self.config,
                os.path.join(
                    work_dir,
                    'local_search_config_step%d.json' %
                    self.config['search']['step'],
                ),
            )

    def estimate_and_expand(self, model: nn.Module):
        """estimate and search for RFConvOp.

        Args:
            model (nn.Module): pytorch model
        """
        for module in model.modules():
            if isinstance(module, BaseConvRFSearchOp):
                module.estimate_rates()
                module.expand_rates()

    def wrap_model(self,
                   model: nn.Module,
                   search_op: str = 'Conv2d',
                   prefix: str = ''):
        """wrap model to support searchable conv op.

        Args:
            model (nn.Module): pytorch model
            search_op (str): The module that uses RF search.
                Defaults to 'Conv2d'.
            init_rates (int, optional): Set to other initial dilation rates.
                Defaults to None.
            prefix (str): Prefix for function recursion. Defaults to ''.
        """
        op = 'torch.nn.' + search_op
        for name, module in model.named_children():
            if prefix == '':
                fullname = 'module.' + name
            else:
                fullname = prefix + '.' + name
            if self.config['search']['skip_layer'] is not None:
                if any(layer in fullname
                       for layer in self.config['search']['skip_layer']):
                    continue
            if isinstance(module, eval(op)):
                if 1 < module.kernel_size[0] and \
                    0 != module.kernel_size[0] % 2 or \
                    1 < module.kernel_size[1] and \
                        0 != module.kernel_size[1] % 2:
                    moduleWrap = eval(search_op + 'RFSearchOp')(
                        module, self.config['search'], self.verbose)
                    moduleWrap = moduleWrap.to(module.weight.device)
                    if self.verbose:
                        logger.info('Wrap model %s to %s.' %
                                    (str(module), str(moduleWrap)))
                    setattr(model, name, moduleWrap)
            elif not isinstance(module, BaseConvRFSearchOp):
                self.wrap_model(module, search_op, fullname)

    def set_model(self,
                  model: nn.Module,
                  search_op: str = 'Conv2d',
                  init_rates: Optional[int] = None,
                  prefix: str = ''):
        """set model based on config.

        Args:
            model (nn.Module): pytorch model
            config (Dict): config file
            search_op (str): The module that uses RF search.
                Defaults to 'Conv2d'.
            init_rates (int, optional):  Set to other initial dilation rates.
                Defaults to None.
            prefix (str): Prefix for function recursion. Defaults to ''.
        """
        op = 'torch.nn.' + search_op
        for name, module in model.named_children():
            if prefix == '':
                fullname = 'module.' + name
            else:
                fullname = prefix + '.' + name
            if self.config['search']['skip_layer'] is not None:
                if any(layer in fullname
                       for layer in self.config['search']['skip_layer']):
                    continue
            if isinstance(module, eval(op)):
                if 1 < module.kernel_size[0] and \
                    0 != module.kernel_size[0] % 2 or \
                    1 < module.kernel_size[1] and \
                        0 != module.kernel_size[1] % 2:
                    if isinstance(self.config['structure'][fullname], int):
                        self.config['structure'][fullname] = [
                            self.config['structure'][fullname],
                            self.config['structure'][fullname]
                        ]
                    module.dilation = (
                        self.config['structure'][fullname][0],
                        self.config['structure'][fullname][1],
                    )
                    module.padding = (
                        get_single_padding(
                            module.kernel_size[0], module.stride[0],
                            self.config['structure'][fullname][0]),
                        get_single_padding(
                            module.kernel_size[1], module.stride[1],
                            self.config['structure'][fullname][1]))
                    setattr(model, name, module)
                    if self.verbose:
                        logger.info(
                            'Set module %s dilation as: [%d %d]' %
                            (fullname, module.dilation[0], module.dilation[1]))
            elif not isinstance(module, BaseConvRFSearchOp):
                self.set_model(module, search_op, init_rates, fullname)
