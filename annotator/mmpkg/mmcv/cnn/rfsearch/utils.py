# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

import annotator.mmpkg.mmcv as mmcv


def write_to_json(config: dict, filename: str):
    """save config to json file.

    Args:
        config (dict): Config to be saved.
        filename (str): Path to save config.
    """

    with open(filename, 'w', encoding='utf-8') as f:
        mmcv.dump(config, f, file_format='json')


def expand_rates(dilation: tuple, config: dict) -> list:
    """expand dilation rate according to config.

    Args:
        dilation (int): _description_
        config (dict): config dict

    Returns:
        list: list of expanded dilation rates
    """
    exp_rate = config['exp_rate']

    large_rates = []
    small_rates = []
    for _ in range(config['num_branches'] // 2):
        large_rates.append(
            tuple([
                np.clip(
                    int(round((1 + exp_rate) * dilation[0])), config['mmin'],
                    config['mmax']).item(),
                np.clip(
                    int(round((1 + exp_rate) * dilation[1])), config['mmin'],
                    config['mmax']).item()
            ]))
        small_rates.append(
            tuple([
                np.clip(
                    int(round((1 - exp_rate) * dilation[0])), config['mmin'],
                    config['mmax']).item(),
                np.clip(
                    int(round((1 - exp_rate) * dilation[1])), config['mmin'],
                    config['mmax']).item()
            ]))

    small_rates.reverse()

    if config['num_branches'] % 2 == 0:
        rate_list = small_rates + large_rates
    else:
        rate_list = small_rates + [dilation] + large_rates

    unique_rate_list = list(set(rate_list))
    unique_rate_list.sort(key=rate_list.index)
    return unique_rate_list


def get_single_padding(kernel_size: int,
                       stride: int = 1,
                       dilation: int = 1) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding
