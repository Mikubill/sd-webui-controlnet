# Copyright (c) OpenMMLab. All rights reserved.
from torch import nn

from .registry import MODULE_WRAPPERS


def is_module_wrapper(module: nn.Module) -> bool:
    """Check if a module is a module wrapper.

    The following 3 modules in MMCV (and their subclasses) are regarded as
    module wrappers: DataParallel, DistributedDataParallel,
    MMDistributedDataParallel (the deprecated version). You may add you own
    module wrapper by registering it to mmcv.parallel.MODULE_WRAPPERS or
    its children registries.

    Args:
        module (nn.Module): The module to be checked.

    Returns:
        bool: True if the input module is a module wrapper.
    """

    def is_module_in_wrapper(module, module_wrapper):
        module_wrappers = tuple(module_wrapper.module_dict.values())
        if isinstance(module, module_wrappers):
            return True
        for child in module_wrapper.children.values():
            if is_module_in_wrapper(module, child):
                return True
        return False

    return is_module_in_wrapper(module, MODULE_WRAPPERS)
