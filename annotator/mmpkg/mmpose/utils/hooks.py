# Copyright (c) OpenMMLab. All rights reserved.
import functools


class OutputHook:

    def __init__(self, module, outputs=None, as_tensor=False):
        self.outputs = outputs
        self.as_tensor = as_tensor
        self.layer_outputs = {}
        self.register(module)

    def register(self, module):

        def hook_wrapper(name):

            def hook(model, input, output):
                if self.as_tensor:
                    self.layer_outputs[name] = output
                else:
                    if isinstance(output, list):
                        self.layer_outputs[name] = [
                            out.detach().cpu().numpy() for out in output
                        ]
                    else:
                        self.layer_outputs[name] = output.detach().cpu().numpy(
                        )

            return hook

        self.handles = []
        if isinstance(self.outputs, (list, tuple)):
            for name in self.outputs:
                try:
                    layer = rgetattr(module, name)
                    h = layer.register_forward_hook(hook_wrapper(name))
                except ModuleNotFoundError as module_not_found:
                    raise ModuleNotFoundError(
                        f'Module {name} not found') from module_not_found
                self.handles.append(h)

    def remove(self):
        for h in self.handles:
            h.remove()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()


# using wonder's beautiful simplification:
# https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects
def rgetattr(obj, attr, *args):

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))
