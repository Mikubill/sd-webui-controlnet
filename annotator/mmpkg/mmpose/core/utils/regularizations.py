# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod, abstractproperty

import torch


class PytorchModuleHook(metaclass=ABCMeta):
    """Base class for PyTorch module hook registers.

    An instance of a subclass of PytorchModuleHook can be used to
    register hook to a pytorch module using the `register` method like:
        hook_register.register(module)

    Subclasses should add/overwrite the following methods:
        - __init__
        - hook
        - hook_type
    """

    @abstractmethod
    def hook(self, *args, **kwargs):
        """Hook function."""

    @abstractproperty
    def hook_type(self) -> str:
        """Hook type Subclasses should overwrite this function to return a
        string value in.

        {`forward`, `forward_pre`, `backward`}
        """

    def register(self, module):
        """Register the hook function to the module.

        Args:
            module (pytorch module): the module to register the hook.

        Returns:
            handle (torch.utils.hooks.RemovableHandle): a handle to remove
                the hook by calling handle.remove()
        """
        assert isinstance(module, torch.nn.Module)

        if self.hook_type == 'forward':
            h = module.register_forward_hook(self.hook)
        elif self.hook_type == 'forward_pre':
            h = module.register_forward_pre_hook(self.hook)
        elif self.hook_type == 'backward':
            h = module.register_backward_hook(self.hook)
        else:
            raise ValueError(f'Invalid hook type {self.hook}')

        return h


class WeightNormClipHook(PytorchModuleHook):
    """Apply weight norm clip regularization.

    The module's parameter will be clip to a given maximum norm before each
    forward pass.

    Args:
        max_norm (float): The maximum norm of the parameter.
        module_param_names (str|list): The parameter name (or name list) to
            apply weight norm clip.
    """

    def __init__(self, max_norm=1.0, module_param_names='weight'):
        self.module_param_names = module_param_names if isinstance(
            module_param_names, list) else [module_param_names]
        self.max_norm = max_norm

    @property
    def hook_type(self):
        return 'forward_pre'

    def hook(self, module, _input):
        for name in self.module_param_names:
            assert name in module._parameters, f'{name} is not a parameter' \
                f' of the module {type(module)}'
            param = module._parameters[name]

            with torch.no_grad():
                m = param.norm().item()
                if m > self.max_norm:
                    param.mul_(self.max_norm / (m + 1e-6))
