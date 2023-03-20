# Copyright (c) OpenMMLab. All rights reserved.
import copy
import platform
import random
from functools import partial

import numpy as np
import torch
from annotator.mmpkg.mmcv.parallel import collate
from annotator.mmpkg.mmcv.runner import get_dist_info
from annotator.mmpkg.mmcv.utils import Registry, build_from_cfg, is_seq_of
from annotator.mmpkg.mmcv.utils.parrots_wrapper import _get_dataloader
from torch.utils.data.dataset import ConcatDataset

from .samplers import DistributedSampler

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')


def _concat_dataset(cfg, default_args=None):
    types = cfg['type']
    ann_files = cfg['ann_file']
    img_prefixes = cfg.get('img_prefix', None)
    dataset_infos = cfg.get('dataset_info', None)

    num_joints = cfg['data_cfg'].get('num_joints', None)
    dataset_channel = cfg['data_cfg'].get('dataset_channel', None)

    datasets = []
    num_dset = len(ann_files)
    for i in range(num_dset):
        cfg_copy = copy.deepcopy(cfg)
        cfg_copy['ann_file'] = ann_files[i]

        if isinstance(types, (list, tuple)):
            cfg_copy['type'] = types[i]
        if isinstance(img_prefixes, (list, tuple)):
            cfg_copy['img_prefix'] = img_prefixes[i]
        if isinstance(dataset_infos, (list, tuple)):
            cfg_copy['dataset_info'] = dataset_infos[i]

        if isinstance(num_joints, (list, tuple)):
            cfg_copy['data_cfg']['num_joints'] = num_joints[i]

        if is_seq_of(dataset_channel, list):
            cfg_copy['data_cfg']['dataset_channel'] = dataset_channel[i]

        datasets.append(build_dataset(cfg_copy, default_args))

    return ConcatDataset(datasets)


def build_dataset(cfg, default_args=None):
    """Build a dataset from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        default_args (dict, optional): Default initialization arguments.
            Default: None.

    Returns:
        Dataset: The constructed dataset.
    """
    from .dataset_wrappers import RepeatDataset

    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'ConcatDataset':
        dataset = ConcatDataset(
            [build_dataset(c, default_args) for c in cfg['datasets']])
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif isinstance(cfg.get('ann_file'), (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)
    return dataset


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     drop_last=True,
                     pin_memory=True,
                     **kwargs):
    """Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        drop_last (bool): Whether to drop the last incomplete batch in epoch.
            Default: True
        pin_memory (bool): Whether to use pin_memory in DataLoader.
            Default: True
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()
    if dist:
        sampler = DistributedSampler(
            dataset, world_size, rank, shuffle=shuffle, seed=seed)
        shuffle = False
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        sampler = None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    _, DataLoader = _get_dataloader()
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=pin_memory,
        shuffle=shuffle,
        worker_init_fn=init_fn,
        drop_last=drop_last,
        **kwargs)

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Init the random seed for various workers."""
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
