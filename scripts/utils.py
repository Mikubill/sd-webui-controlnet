import torch
import os
import functools
import numpy as np

from typing import Any, Callable

def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(
            ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict


def get_state_dict(d):
    return d.get('state_dict', d)


def ndarray_lru_cache(max_size: int = 128, typed: bool = False):
    """
    Decorator to enable caching for functions with numpy array arguments.
    Numpy arrays are mutable, and thus not directly usable as hash keys.

    The idea here is to wrap the incoming arguments with type `np.ndarray`
    as `HashableNpArray` so that `lru_cache` can correctly handles `np.ndarray`
    arguments.

    `HashableNpArray` functions exactly the same way as `np.ndarray` except
    having `__hash__` and `__eq__` overriden.
    """

    def decorator(func: Callable):
        """The actual decorator that accept function as input."""

        class HashableNpArray(np.ndarray):
            def __new__(cls, input_array):
                # Input array is an instance of ndarray.
                # The view makes the input array and returned array share the same data.
                obj = np.asarray(input_array).view(cls)
                return obj

            def __eq__(self, other) -> bool:
                return np.array_equal(self, other)

            def __hash__(self):
                # Hash the bytes representing the data of the array.
                return hash(self.tobytes())

        @functools.lru_cache(maxsize=max_size, typed=typed)
        def cached_func(*args, **kwargs):
            """This function only accepts `HashableNpArray` as input params."""
            return func(*args, **kwargs)

        # Preserves original function.__name__ and __doc__.
        @functools.wraps(func)
        def decorated_func(*args, **kwargs):
            """The decorated function that delegates the original function."""

            def convert_item(item: Any):
                return HashableNpArray(item) if isinstance(item, np.ndarray) else item

            args = [convert_item(arg) for arg in args]
            kwargs = {k: convert_item(arg) for k, arg in kwargs.items()}
            return cached_func(*args, **kwargs)

        return decorated_func

    return decorator
