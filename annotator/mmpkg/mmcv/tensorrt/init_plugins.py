# Copyright (c) OpenMMLab. All rights reserved.
import ctypes
import glob
import os
import warnings


def get_tensorrt_op_path() -> str:
    """Get TensorRT plugins library path."""
    # Following strings of text style are from colorama package
    bright_style, reset_style = '\x1b[1m', '\x1b[0m'
    red_text, blue_text = '\x1b[31m', '\x1b[34m'
    white_background = '\x1b[107m'

    msg = white_background + bright_style + red_text
    msg += 'DeprecationWarning: This function will be deprecated in future. '
    msg += blue_text + 'Welcome to use the unified model deployment toolbox '
    msg += 'MMDeploy: https://github.com/open-mmlab/mmdeploy'
    msg += reset_style
    warnings.warn(msg)

    wildcard = os.path.join(
        os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
        '_ext_trt.*.so')

    paths = glob.glob(wildcard)
    lib_path = paths[0] if len(paths) > 0 else ''
    return lib_path


plugin_is_loaded = False


def is_tensorrt_plugin_loaded() -> bool:
    """Check if TensorRT plugins library is loaded or not.

    Returns:
        bool: plugin_is_loaded flag
    """

    # Following strings of text style are from colorama package
    bright_style, reset_style = '\x1b[1m', '\x1b[0m'
    red_text, blue_text = '\x1b[31m', '\x1b[34m'
    white_background = '\x1b[107m'

    msg = white_background + bright_style + red_text
    msg += 'DeprecationWarning: This function will be deprecated in future. '
    msg += blue_text + 'Welcome to use the unified model deployment toolbox '
    msg += 'MMDeploy: https://github.com/open-mmlab/mmdeploy'
    msg += reset_style
    warnings.warn(msg)

    global plugin_is_loaded
    return plugin_is_loaded


def load_tensorrt_plugin() -> None:
    """load TensorRT plugins library."""

    # Following strings of text style are from colorama package
    bright_style, reset_style = '\x1b[1m', '\x1b[0m'
    red_text, blue_text = '\x1b[31m', '\x1b[34m'
    white_background = '\x1b[107m'

    msg = white_background + bright_style + red_text
    msg += 'DeprecationWarning: This function will be deprecated in future. '
    msg += blue_text + 'Welcome to use the unified model deployment toolbox '
    msg += 'MMDeploy: https://github.com/open-mmlab/mmdeploy'
    msg += reset_style
    warnings.warn(msg)

    global plugin_is_loaded
    lib_path = get_tensorrt_op_path()
    if (not plugin_is_loaded) and os.path.exists(lib_path):
        ctypes.CDLL(lib_path)
        plugin_is_loaded = True
