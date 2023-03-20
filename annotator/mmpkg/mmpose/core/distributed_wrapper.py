# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from annotator.mmpkg.mmcv.parallel import MODULE_WRAPPERS as MMCV_MODULE_WRAPPERS
from annotator.mmpkg.mmcv.parallel import MMDistributedDataParallel
from annotator.mmpkg.mmcv.parallel.scatter_gather import scatter_kwargs
from annotator.mmpkg.mmcv.utils import Registry
from torch.cuda._utils import _get_device_index

MODULE_WRAPPERS = Registry('module wrapper', parent=MMCV_MODULE_WRAPPERS)


@MODULE_WRAPPERS.register_module()
class DistributedDataParallelWrapper(nn.Module):
    """A DistributedDataParallel wrapper for models in 3D mesh estimation task.

    In  3D mesh estimation task, there is a need to wrap different modules in
    the models with separate DistributedDataParallel. Otherwise, it will cause
    errors for GAN training.
    More specific, the GAN model, usually has two sub-modules:
    generator and discriminator. If we wrap both of them in one
    standard DistributedDataParallel, it will cause errors during training,
    because when we update the parameters of the generator (or discriminator),
    the parameters of the discriminator (or generator) is not updated, which is
    not allowed for DistributedDataParallel.
    So we design this wrapper to separately wrap DistributedDataParallel
    for generator and discriminator.

    In this wrapper, we perform two operations:
    1. Wrap the modules in the models with separate MMDistributedDataParallel.
        Note that only modules with parameters will be wrapped.
    2. Do scatter operation for 'forward', 'train_step' and 'val_step'.

    Note that the arguments of this wrapper is the same as those in
    `torch.nn.parallel.distributed.DistributedDataParallel`.

    Args:
        module (nn.Module): Module that needs to be wrapped.
        device_ids (list[int | `torch.device`]): Same as that in
            `torch.nn.parallel.distributed.DistributedDataParallel`.
        dim (int, optional): Same as that in the official scatter function in
            pytorch. Defaults to 0.
        broadcast_buffers (bool): Same as that in
            `torch.nn.parallel.distributed.DistributedDataParallel`.
            Defaults to False.
        find_unused_parameters (bool, optional): Same as that in
            `torch.nn.parallel.distributed.DistributedDataParallel`.
            Traverse the autograd graph of all tensors contained in returned
            value of the wrapped module’s forward function. Defaults to False.
        kwargs (dict): Other arguments used in
            `torch.nn.parallel.distributed.DistributedDataParallel`.
    """

    def __init__(self,
                 module,
                 device_ids,
                 dim=0,
                 broadcast_buffers=False,
                 find_unused_parameters=False,
                 **kwargs):
        super().__init__()
        assert len(device_ids) == 1, (
            'Currently, DistributedDataParallelWrapper only supports one'
            'single CUDA device for each process.'
            f'The length of device_ids must be 1, but got {len(device_ids)}.')
        self.module = module
        self.dim = dim
        self.to_ddp(
            device_ids=device_ids,
            dim=dim,
            broadcast_buffers=broadcast_buffers,
            find_unused_parameters=find_unused_parameters,
            **kwargs)
        self.output_device = _get_device_index(device_ids[0], True)

    def to_ddp(self, device_ids, dim, broadcast_buffers,
               find_unused_parameters, **kwargs):
        """Wrap models with separate MMDistributedDataParallel.

        It only wraps the modules with parameters.
        """
        for name, module in self.module._modules.items():
            if next(module.parameters(), None) is None:
                module = module.cuda()
            elif all(not p.requires_grad for p in module.parameters()):
                module = module.cuda()
            else:
                module = MMDistributedDataParallel(
                    module.cuda(),
                    device_ids=device_ids,
                    dim=dim,
                    broadcast_buffers=broadcast_buffers,
                    find_unused_parameters=find_unused_parameters,
                    **kwargs)
            self.module._modules[name] = module

    def scatter(self, inputs, kwargs, device_ids):
        """Scatter function.

        Args:
            inputs (Tensor): Input Tensor.
            kwargs (dict): Args for
                ``mmcv.parallel.scatter_gather.scatter_kwargs``.
            device_ids (int): Device id.
        """
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def forward(self, *inputs, **kwargs):
        """Forward function.

        Args:
            inputs (tuple): Input data.
            kwargs (dict): Args for
                ``mmcv.parallel.scatter_gather.scatter_kwargs``.
        """
        inputs, kwargs = self.scatter(inputs, kwargs,
                                      [torch.cuda.current_device()])
        return self.module(*inputs[0], **kwargs[0])

    def train_step(self, *inputs, **kwargs):
        """Train step function.

        Args:
            inputs (Tensor): Input Tensor.
            kwargs (dict): Args for
                ``mmcv.parallel.scatter_gather.scatter_kwargs``.
        """
        inputs, kwargs = self.scatter(inputs, kwargs,
                                      [torch.cuda.current_device()])
        output = self.module.train_step(*inputs[0], **kwargs[0])
        return output

    def val_step(self, *inputs, **kwargs):
        """Validation step function.

        Args:
            inputs (tuple): Input data.
            kwargs (dict): Args for ``scatter_kwargs``.
        """
        inputs, kwargs = self.scatter(inputs, kwargs,
                                      [torch.cuda.current_device()])
        output = self.module.val_step(*inputs[0], **kwargs[0])
        return output
