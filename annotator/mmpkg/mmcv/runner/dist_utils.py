# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import functools
import os
import socket
import subprocess
from collections import OrderedDict
from typing import Callable, List, Optional, Tuple

import torch
import torch.multiprocessing as mp
from torch import distributed as dist
from torch._utils import (_flatten_dense_tensors, _take_tensors,
                          _unflatten_dense_tensors)

from annotator.mmpkg.mmcv.utils import IS_MLU_AVAILABLE, IS_NPU_AVAILABLE


def _find_free_port() -> str:
    # Copied from https://github.com/facebookresearch/detectron2/blob/main/detectron2/engine/launch.py # noqa: E501
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def _is_free_port(port: int) -> bool:
    ips = socket.gethostbyname_ex(socket.gethostname())[-1]
    ips.append('localhost')
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return all(s.connect_ex((ip, port)) != 0 for ip in ips)


def init_dist(launcher: str, backend: str = 'nccl', **kwargs) -> None:
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    if launcher == 'pytorch':
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == 'mpi':
        _init_dist_mpi(backend, **kwargs)
    elif launcher == 'slurm':
        _init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError(f'Invalid launcher type: {launcher}')


def _init_dist_pytorch(backend: str, **kwargs) -> None:
    # TODO: use local_rank instead of rank % num_gpus
    rank = int(os.environ['RANK'])
    if IS_MLU_AVAILABLE:
        import torch_mlu  # noqa: F401
        torch.mlu.set_device(rank)
        dist.init_process_group(
            backend='cncl',
            rank=rank,
            world_size=int(os.environ['WORLD_SIZE']),
            **kwargs)
    elif IS_NPU_AVAILABLE:
        import torch_npu  # noqa: F401
        num_npus = torch.npu.device_count()
        torch.npu.set_device(rank % num_npus)
        dist.init_process_group(
            backend='hccl',
            rank=rank,
            world_size=int(os.environ['WORLD_SIZE']),
            **kwargs)
    else:
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(rank % num_gpus)
        dist.init_process_group(backend=backend, **kwargs)


def _init_dist_mpi(backend: str, **kwargs) -> None:
    local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    if 'MASTER_PORT' not in os.environ:
        # 29500 is torch.distributed default port
        os.environ['MASTER_PORT'] = '29500'
    if 'MASTER_ADDR' not in os.environ:
        raise KeyError('The environment variable MASTER_ADDR is not set')
    os.environ['WORLD_SIZE'] = os.environ['OMPI_COMM_WORLD_SIZE']
    os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
    dist.init_process_group(backend=backend, **kwargs)


def _init_dist_slurm(backend: str, port: Optional[int] = None) -> None:
    """Initialize slurm distributed training environment.

    If argument ``port`` is not specified, then the master port will be system
    environment variable ``MASTER_PORT``. If ``MASTER_PORT`` is not in system
    environment variable, then a default port ``29500`` will be used.

    Args:
        backend (str): Backend of torch.distributed.
        port (int, optional): Master port. Defaults to None.
    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(
        f'scontrol show hostname {node_list} | head -n1')
    # specify master port
    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    elif 'MASTER_PORT' in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        # if torch.distributed default port(29500) is available
        # then use it, else find a free port
        if _is_free_port(29500):
            os.environ['MASTER_PORT'] = '29500'
        else:
            os.environ['MASTER_PORT'] = str(_find_free_port())
    # use MASTER_ADDR in the environment variable if it already exists
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)


def get_dist_info() -> Tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def master_only(func: Callable) -> Callable:

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper


def allreduce_params(params: List[torch.nn.Parameter],
                     coalesce: bool = True,
                     bucket_size_mb: int = -1) -> None:
    """Allreduce parameters.

    Args:
        params (list[torch.nn.Parameter]): List of parameters or buffers
            of a model.
        coalesce (bool, optional): Whether allreduce parameters as a whole.
            Defaults to True.
        bucket_size_mb (int, optional): Size of bucket, the unit is MB.
            Defaults to -1.
    """
    _, world_size = get_dist_info()
    if world_size == 1:
        return
    params = [param.data for param in params]
    if coalesce:
        _allreduce_coalesced(params, world_size, bucket_size_mb)
    else:
        for tensor in params:
            dist.all_reduce(tensor.div_(world_size))


def allreduce_grads(params: List[torch.nn.Parameter],
                    coalesce: bool = True,
                    bucket_size_mb: int = -1) -> None:
    """Allreduce gradients.

    Args:
        params (list[torch.nn.Parameter]): List of parameters of a model.
        coalesce (bool, optional): Whether allreduce parameters as a whole.
            Defaults to True.
        bucket_size_mb (int, optional): Size of bucket, the unit is MB.
            Defaults to -1.
    """
    grads = [
        param.grad.data for param in params
        if param.requires_grad and param.grad is not None
    ]
    _, world_size = get_dist_info()
    if world_size == 1:
        return
    if coalesce:
        _allreduce_coalesced(grads, world_size, bucket_size_mb)
    else:
        for tensor in grads:
            dist.all_reduce(tensor.div_(world_size))


def _allreduce_coalesced(tensors: torch.Tensor,
                         world_size: int,
                         bucket_size_mb: int = -1) -> None:
    if bucket_size_mb > 0:
        bucket_size_bytes = bucket_size_mb * 1024 * 1024
        buckets = _take_tensors(tensors, bucket_size_bytes)
    else:
        buckets = OrderedDict()
        for tensor in tensors:
            tp = tensor.type()
            if tp not in buckets:
                buckets[tp] = []
            buckets[tp].append(tensor)
        buckets = buckets.values()

    for bucket in buckets:
        flat_tensors = _flatten_dense_tensors(bucket)
        dist.all_reduce(flat_tensors)
        flat_tensors.div_(world_size)
        for tensor, synced in zip(
                bucket, _unflatten_dense_tensors(flat_tensors, bucket)):
            tensor.copy_(synced)
