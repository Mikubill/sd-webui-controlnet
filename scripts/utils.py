import torch
import os
import functools
import time
import base64
import gradio as gr
import numpy as np
import safetensors.torch
import logging
import io

from typing import Any, Callable, Dict
from modules.safe import unsafe_torch_load

from PIL import Image
from scripts.logging import logger


def load_state_dict(ckpt_path, location="cpu"):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = unsafe_torch_load(ckpt_path, map_location=torch.device(location))
    state_dict = get_state_dict(state_dict)
    logger.info(f"Loaded state_dict from [{ckpt_path}]")
    return state_dict


def get_state_dict(d):
    return d.get("state_dict", d)


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


def timer_decorator(func):
    """Time the decorated function and output the result to debug logger."""
    if logger.level != logging.DEBUG:
        return func

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        # Only report function that are significant enough.
        if duration > 1e-3:
            logger.debug(f"{func.__name__} ran in: {duration} sec")
        return result

    return wrapper


class TimeMeta(type):
    """ Metaclass to record execution time on all methods of the
    child class. """
    def __new__(cls, name, bases, attrs):
        for attr_name, attr_value in attrs.items():
            if callable(attr_value):
                attrs[attr_name] = timer_decorator(attr_value)
        return super().__new__(cls, name, bases, attrs)


# svgsupports
svgsupport = False
try:
    import io
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPM

    svgsupport = True
except ImportError:
    pass


def svg_preprocess(inputs: Dict):
    if not inputs:
        return None

    if inputs["image"].startswith("data:image/svg+xml;base64,") and svgsupport:
        svg_data = base64.b64decode(
            inputs["image"].replace("data:image/svg+xml;base64,", "")
        )
        drawing = svg2rlg(io.BytesIO(svg_data))
        png_data = renderPM.drawToString(drawing, fmt="PNG")
        encoded_string = base64.b64encode(png_data)
        base64_str = str(encoded_string, "utf-8")
        base64_str = "data:image/png;base64," + base64_str
        inputs["image"] = base64_str
    return inputs


# This is a patch for `gr.Image.preprocess()`
def image_preprocess(self, x: Dict):
    inputs = x # lackluster name `x` is the method keyword name used by `gr.Image.preprocess`
    inputs = svg_preprocess(inputs)

    if self.image_mode is None:
        # XXX(teding): Not sure if this early return is necessary, since I don't know why the caller is using `self.image_mode=None`.
        # If 16-bit grayscale images are getting incorrectly converted to 8-bit, try removing this early return.
        ret = gr.Image.preprocess(self, inputs)
        return ret

    # Work around for 16 bit grayscale images being improperly converted to 8 bit
    image_mode = self.image_mode
    image_type = self.type
    if self.image_mode not in [None, "RGB"]:
        raise Exception(f"Unsupported image mode {self.image_mode}")
    if self.type not in ["numpy", "pil"]:
        raise Exception(f"Unsupported image type {self.type}")
    self.image_mode = None
    self.type = "pil"

    ret = gr.Image.preprocess(self, inputs)

    def go_image(image, input):
        if image is None:
            return None
        # gr.Image.preprocess() throws away depth information, so we need to decode again.
        # Naively just checking if any elem is greater than 255 is not enough, because
        # the image may be a (very dark) 16 bit grayscale image with all values below 256.
        png_encoded = gr.processing_utils.extract_base64_data(input)
        png = Image.open(io.BytesIO(base64.b64decode(png_encoded)))
        if len(png.tile) == 1 and len(png.tile[0]) == 4:
            data_mode = png.tile[0][3] # This is not necessarily the same as `image.mode` or `png.mode`
            # map for encodings that have more channel depth than 256
            mode_to_oversized_depth = {
                "I": 0x100000000,
                "I;16": 0x10000,
                "I;16L": 0x10000,
                "I;16B": 0x10000,
                "I;16N": 0x10000,
            }
            depth = mode_to_oversized_depth.get(data_mode, 256)
            if depth > 256:
                ratio = 256 / depth
                data = np.array(image)
                data = data * ratio
                data = np.clip(data, 0, 255)
                data = data.astype(np.uint8)
                image = Image.fromarray(data, mode="L")
            image = image.convert("RGB")
        if image_type == "numpy":
            image = np.array(image)
        return image

    # NOTE(teding): This could probably be the same as `go_image`, but when I looked at the
    # local venv source code of `gr.Image.preprocess()`, this is what it does for the mask.
    def go_mask(mask):
        if mask is None:
            return None
        if image_type == "numpy":
            mask = np.array(mask)
        return mask

    if isinstance(ret, dict):
        ret["image"] = go_image(ret["image"], inputs["image"])
        ret["mask"] = go_mask(ret["mask"])
    else:
        ret = go_image(ret, inputs)

    self.image_mode = image_mode
    self.type = image_type
    return ret


def get_unique_axis0(data):
    arr = np.asanyarray(data)
    idxs = np.lexsort(arr.T)
    arr = arr[idxs]
    unique_idxs = np.empty(len(arr), dtype=np.bool_)
    unique_idxs[:1] = True
    unique_idxs[1:] = np.any(arr[:-1, :] != arr[1:, :], axis=-1)
    return arr[unique_idxs]
