# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings

import torch


def is_custom_op_loaded() -> bool:
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

    flag = False
    try:
        from ..tensorrt import is_tensorrt_plugin_loaded
        flag = is_tensorrt_plugin_loaded()
    except (ImportError, ModuleNotFoundError):
        pass
    if not flag:
        try:
            from ..ops import get_onnxruntime_op_path
            ort_lib_path = get_onnxruntime_op_path()
            flag = os.path.exists(ort_lib_path)
        except (ImportError, ModuleNotFoundError):
            pass
    return flag or torch.__version__ == 'parrots'
