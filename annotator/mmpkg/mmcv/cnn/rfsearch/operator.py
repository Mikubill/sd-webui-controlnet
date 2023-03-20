# Copyright (c) OpenMMLab. All rights reserved.
import copy

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from annotator.mmpkg.mmcv.runner import BaseModule
from annotator.mmpkg.mmcv.utils.logging import get_logger
from .utils import expand_rates, get_single_padding

logger = get_logger('mmcv')


class BaseConvRFSearchOp(BaseModule):
    """Based class of ConvRFSearchOp.

    Args:
        op_layer (nn.Module): pytorch module, e,g, Conv2d
        global_config (dict): config dict.
    """

    def __init__(self, op_layer: nn.Module, global_config: dict):
        super().__init__()
        self.op_layer = op_layer
        self.global_config = global_config

    def normlize(self, weights: nn.Parameter) -> nn.Parameter:
        """Normalize weights.

        Args:
            weights (nn.Parameter): Weights to be normalized.

        Returns:
            nn.Parameters: Normalized weights.
        """
        abs_weights = torch.abs(weights)
        normalized_weights = abs_weights / torch.sum(abs_weights)
        return normalized_weights


class Conv2dRFSearchOp(BaseConvRFSearchOp):
    """Enable Conv2d with receptive field searching ability.

    Args:
        op_layer (nn.Module): pytorch module, e,g, Conv2d
        global_config (dict): config dict. Defaults to None.
            By default this must include:

            - "init_alphas": The value for initializing weights of each branch.
            - "num_branches": The controller of the size of
              search space (the number of branches).
            - "exp_rate": The controller of the sparsity of search space.
            - "mmin": The minimum dilation rate.
            - "mmax": The maximum dilation rate.

            Extra keys may exist, but are used by RFSearchHook, e.g., "step",
            "max_step", "search_interval", and "skip_layer".
        verbose (bool): Determines whether to print rf-next
            related logging messages.
            Defaults to True.
    """

    def __init__(self,
                 op_layer: nn.Module,
                 global_config: dict,
                 verbose: bool = True):
        super().__init__(op_layer, global_config)
        assert global_config is not None, 'global_config is None'
        self.num_branches = global_config['num_branches']
        assert self.num_branches in [2, 3]
        self.verbose = verbose
        init_dilation = op_layer.dilation
        self.dilation_rates = expand_rates(init_dilation, global_config)
        if self.op_layer.kernel_size[
                0] == 1 or self.op_layer.kernel_size[0] % 2 == 0:
            self.dilation_rates = [(op_layer.dilation[0], r[1])
                                   for r in self.dilation_rates]
        if self.op_layer.kernel_size[
                1] == 1 or self.op_layer.kernel_size[1] % 2 == 0:
            self.dilation_rates = [(r[0], op_layer.dilation[1])
                                   for r in self.dilation_rates]

        self.branch_weights = nn.Parameter(torch.Tensor(self.num_branches))
        if self.verbose:
            logger.info(f'Expand as {self.dilation_rates}')
        nn.init.constant_(self.branch_weights, global_config['init_alphas'])

    def forward(self, input: Tensor) -> Tensor:
        norm_w = self.normlize(self.branch_weights[:len(self.dilation_rates)])
        if len(self.dilation_rates) == 1:
            outputs = [
                nn.functional.conv2d(
                    input,
                    weight=self.op_layer.weight,
                    bias=self.op_layer.bias,
                    stride=self.op_layer.stride,
                    padding=self.get_padding(self.dilation_rates[0]),
                    dilation=self.dilation_rates[0],
                    groups=self.op_layer.groups,
                )
            ]
        else:
            outputs = [
                nn.functional.conv2d(
                    input,
                    weight=self.op_layer.weight,
                    bias=self.op_layer.bias,
                    stride=self.op_layer.stride,
                    padding=self.get_padding(r),
                    dilation=r,
                    groups=self.op_layer.groups,
                ) * norm_w[i] for i, r in enumerate(self.dilation_rates)
            ]
        output = outputs[0]
        for i in range(1, len(self.dilation_rates)):
            output += outputs[i]
        return output

    def estimate_rates(self):
        """Estimate new dilation rate based on trained branch_weights."""
        norm_w = self.normlize(self.branch_weights[:len(self.dilation_rates)])
        if self.verbose:
            logger.info('Estimate dilation {} with weight {}.'.format(
                self.dilation_rates,
                norm_w.detach().cpu().numpy().tolist()))

        sum0, sum1, w_sum = 0, 0, 0
        for i in range(len(self.dilation_rates)):
            sum0 += norm_w[i].item() * self.dilation_rates[i][0]
            sum1 += norm_w[i].item() * self.dilation_rates[i][1]
            w_sum += norm_w[i].item()
        estimated = [
            np.clip(
                int(round(sum0 / w_sum)), self.global_config['mmin'],
                self.global_config['mmax']).item(),
            np.clip(
                int(round(sum1 / w_sum)), self.global_config['mmin'],
                self.global_config['mmax']).item()
        ]
        self.op_layer.dilation = tuple(estimated)
        self.op_layer.padding = self.get_padding(self.op_layer.dilation)
        self.dilation_rates = [tuple(estimated)]
        if self.verbose:
            logger.info(f'Estimate as {tuple(estimated)}')

    def expand_rates(self):
        """Expand dilation rate."""
        dilation = self.op_layer.dilation
        dilation_rates = expand_rates(dilation, self.global_config)
        if self.op_layer.kernel_size[
                0] == 1 or self.op_layer.kernel_size[0] % 2 == 0:
            dilation_rates = [(dilation[0], r[1]) for r in dilation_rates]
        if self.op_layer.kernel_size[
                1] == 1 or self.op_layer.kernel_size[1] % 2 == 0:
            dilation_rates = [(r[0], dilation[1]) for r in dilation_rates]

        self.dilation_rates = copy.deepcopy(dilation_rates)
        if self.verbose:
            logger.info(f'Expand as {self.dilation_rates}')
        nn.init.constant_(self.branch_weights,
                          self.global_config['init_alphas'])

    def get_padding(self, dilation):
        padding = (get_single_padding(self.op_layer.kernel_size[0],
                                      self.op_layer.stride[0], dilation[0]),
                   get_single_padding(self.op_layer.kernel_size[1],
                                      self.op_layer.stride[1], dilation[1]))
        return padding
