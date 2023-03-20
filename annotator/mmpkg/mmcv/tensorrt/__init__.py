# Copyright (c) OpenMMLab. All rights reserved.
# flake8: noqa
from .init_plugins import is_tensorrt_plugin_loaded, load_tensorrt_plugin
from .preprocess import preprocess_onnx


def is_tensorrt_available():
    try:
        import tensorrt
        del tensorrt
        return True
    except ModuleNotFoundError:
        return False


__all__ = []

if is_tensorrt_available():
    from .tensorrt_utils import (TRTWraper, TRTWrapper, load_trt_engine,
                                 onnx2trt, save_trt_engine)

    # load tensorrt plugin lib
    load_tensorrt_plugin()

    __all__.extend([
        'onnx2trt', 'save_trt_engine', 'load_trt_engine', 'TRTWraper',
        'TRTWrapper'
    ])

__all__.extend(['is_tensorrt_plugin_loaded', 'preprocess_onnx'])
