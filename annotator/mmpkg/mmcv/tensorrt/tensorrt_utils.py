# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Union

import onnx
import tensorrt as trt
import torch

from .preprocess import preprocess_onnx


def onnx2trt(onnx_model: Union[str, onnx.ModelProto],
             opt_shape_dict: dict,
             log_level: trt.ILogger.Severity = trt.Logger.ERROR,
             fp16_mode: bool = False,
             max_workspace_size: int = 0,
             device_id: int = 0) -> trt.ICudaEngine:
    """Convert onnx model to tensorrt engine.

    Arguments:
        onnx_model (str or onnx.ModelProto): the onnx model to convert from
        opt_shape_dict (dict): the min/opt/max shape of each input
        log_level (TensorRT log level): the log level of TensorRT
        fp16_mode (bool): enable fp16 mode
        max_workspace_size (int): set max workspace size of TensorRT engine.
            some tactic and layers need large workspace.
        device_id (int): choice the device to create engine.

    Returns:
        tensorrt.ICudaEngine: the TensorRT engine created from onnx_model

    Example:
        >>> engine = onnx2trt(
        >>>             "onnx_model.onnx",
        >>>             {'input': [[1, 3, 160, 160],
        >>>                        [1, 3, 320, 320],
        >>>                        [1, 3, 640, 640]]},
        >>>             log_level=trt.Logger.WARNING,
        >>>             fp16_mode=True,
        >>>             max_workspace_size=1 << 30,
        >>>             device_id=0)
        >>>             })
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

    device = torch.device(f'cuda:{device_id}')
    # create builder and network
    logger = trt.Logger(log_level)
    builder = trt.Builder(logger)
    EXPLICIT_BATCH = 1 << (int)(
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)

    # parse onnx
    parser = trt.OnnxParser(network, logger)

    if isinstance(onnx_model, str):
        onnx_model = onnx.load(onnx_model)

    onnx_model = preprocess_onnx(onnx_model)

    if not parser.parse(onnx_model.SerializeToString()):
        error_msgs = ''
        for error in range(parser.num_errors):
            error_msgs += f'{parser.get_error(error)}\n'
        raise RuntimeError(f'parse onnx failed:\n{error_msgs}')

    # config builder
    builder.max_workspace_size = max_workspace_size

    config = builder.create_builder_config()
    config.max_workspace_size = max_workspace_size
    profile = builder.create_optimization_profile()

    for input_name, param in opt_shape_dict.items():
        min_shape = tuple(param[0][:])
        opt_shape = tuple(param[1][:])
        max_shape = tuple(param[2][:])
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    if fp16_mode:
        builder.fp16_mode = fp16_mode
        config.set_flag(trt.BuilderFlag.FP16)

    # create engine
    with torch.cuda.device(device):
        engine = builder.build_engine(network, config)

    return engine


def save_trt_engine(engine: trt.ICudaEngine, path: str) -> None:
    """Serialize TensorRT engine to disk.

    Arguments:
        engine (tensorrt.ICudaEngine): TensorRT engine to serialize
        path (str): disk path to write the engine
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

    with open(path, mode='wb') as f:
        f.write(bytearray(engine.serialize()))


def load_trt_engine(path: str) -> trt.ICudaEngine:
    """Deserialize TensorRT engine from disk.

    Arguments:
        path (str): disk path to read the engine

    Returns:
        tensorrt.ICudaEngine: the TensorRT engine loaded from disk
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

    with trt.Logger() as logger, trt.Runtime(logger) as runtime:
        with open(path, mode='rb') as f:
            engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        return engine


def torch_dtype_from_trt(dtype: trt.DataType) -> Union[torch.dtype, TypeError]:
    """Convert pytorch dtype to TensorRT dtype."""
    if dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int8:
        return torch.int8
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError('%s is not supported by torch' % dtype)


def torch_device_from_trt(
        device: trt.TensorLocation) -> Union[torch.device, TypeError]:
    """Convert pytorch device to TensorRT device."""
    if device == trt.TensorLocation.DEVICE:
        return torch.device('cuda')
    elif device == trt.TensorLocation.HOST:
        return torch.device('cpu')
    else:
        return TypeError('%s is not supported by torch' % device)


class TRTWrapper(torch.nn.Module):
    """TensorRT engine Wrapper.

    Arguments:
        engine (tensorrt.ICudaEngine): TensorRT engine to wrap
        input_names (list[str]): names of each inputs
        output_names (list[str]): names of each outputs

    Note:
        If the engine is converted from onnx model. The input_names and
        output_names should be the same as onnx model.
    """

    def __init__(self, engine, input_names=None, output_names=None):

        # Following strings of text style are from colorama package
        bright_style, reset_style = '\x1b[1m', '\x1b[0m'
        red_text, blue_text = '\x1b[31m', '\x1b[34m'
        white_background = '\x1b[107m'

        msg = white_background + bright_style + red_text
        msg += 'DeprecationWarning: This tool will be deprecated in future. '
        msg += blue_text + \
            'Welcome to use the unified model deployment toolbox '
        msg += 'MMDeploy: https://github.com/open-mmlab/mmdeploy'
        msg += reset_style
        warnings.warn(msg)

        super().__init__()
        self.engine = engine
        if isinstance(self.engine, str):
            self.engine = load_trt_engine(engine)

        if not isinstance(self.engine, trt.ICudaEngine):
            raise TypeError('engine should be str or trt.ICudaEngine')

        self._register_state_dict_hook(TRTWrapper._on_state_dict)
        self.context = self.engine.create_execution_context()

        # get input and output names from engine
        if input_names is None or output_names is None:
            names = [_ for _ in self.engine]
            input_names = list(filter(self.engine.binding_is_input, names))
            output_names = list(set(names) - set(input_names))
        self.input_names = input_names
        self.output_names = output_names

    def _on_state_dict(self, state_dict, prefix, local_metadata):
        state_dict[prefix + 'engine'] = bytearray(self.engine.serialize())
        state_dict[prefix + 'input_names'] = self.input_names
        state_dict[prefix + 'output_names'] = self.output_names

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        engine_bytes = state_dict[prefix + 'engine']

        with trt.Logger() as logger, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(engine_bytes)
            self.context = self.engine.create_execution_context()

        self.input_names = state_dict[prefix + 'input_names']
        self.output_names = state_dict[prefix + 'output_names']

    def forward(self, inputs):
        """
        Arguments:
            inputs (dict): dict of input name-tensors pair

        Return:
            dict: dict of output name-tensors pair
        """
        assert self.input_names is not None
        assert self.output_names is not None
        bindings = [None] * (len(self.input_names) + len(self.output_names))

        for input_name, input_tensor in inputs.items():
            idx = self.engine.get_binding_index(input_name)

            if input_tensor.dtype == torch.long:
                input_tensor = input_tensor.int()
            self.context.set_binding_shape(idx, tuple(input_tensor.shape))
            bindings[idx] = input_tensor.contiguous().data_ptr()

        # create output tensors
        outputs = {}
        for i, output_name in enumerate(self.output_names):
            idx = self.engine.get_binding_index(output_name)
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            shape = tuple(self.context.get_binding_shape(idx))

            device = torch_device_from_trt(self.engine.get_location(idx))
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[output_name] = output
            bindings[idx] = output.data_ptr()

        self.context.execute_async_v2(bindings,
                                      torch.cuda.current_stream().cuda_stream)

        return outputs


class TRTWraper(TRTWrapper):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(
            'TRTWraper will be deprecated in'
            ' future. Please use TRTWrapper instead', DeprecationWarning)
