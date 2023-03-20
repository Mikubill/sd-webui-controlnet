# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import numpy as np
import onnx


def preprocess_onnx(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
    """Modify onnx model to match with TensorRT plugins in mmcv.

    There are some conflict between onnx node definition and TensorRT limit.
    This function perform preprocess on the onnx model to solve the conflicts.
    For example, onnx `attribute` is loaded in TensorRT on host and onnx
    `input` is loaded on device. The shape inference is performed on host, so
    any `input` related to shape (such as `max_output_boxes_per_class` in
    NonMaxSuppression) should be transformed to `attribute` before conversion.

    Arguments:
        onnx_model (onnx.ModelProto): Input onnx model.

    Returns:
        onnx.ModelProto: Modified onnx model.
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

    graph = onnx_model.graph
    nodes = graph.node
    initializers = graph.initializer
    node_dict = {}
    for node in nodes:
        node_outputs = node.output
        for output in node_outputs:
            if len(output) > 0:
                node_dict[output] = node

    init_dict = {_.name: _ for _ in initializers}

    nodes_name_to_remove = set()

    def is_node_without_output(name):
        for node_name, node in node_dict.items():
            if node_name not in nodes_name_to_remove:
                if name in node.input:
                    return False
        return True

    def mark_nodes_to_remove(name):
        node = node_dict[name]
        nodes_name_to_remove.add(name)
        for input_node_name in node.input:
            if is_node_without_output(input_node_name):
                mark_nodes_to_remove(input_node_name)

    def parse_data(name, typ, default_value=0):
        if name in node_dict:
            node = node_dict[name]
            if node.op_type == 'Constant':
                raw_data = node.attribute[0].t.raw_data
            else:
                mark_nodes_to_remove(name)
                return default_value
        elif name in init_dict:
            raw_data = init_dict[name].raw_data
        else:
            raise ValueError(f'{name} not found in node or initilizer.')
        return np.frombuffer(raw_data, typ).item()

    nrof_node = len(nodes)
    for idx in range(nrof_node):
        node = nodes[idx]
        node_attributes = node.attribute
        node_inputs = node.input
        node_outputs = node.output
        node_name = node.name
        # process NonMaxSuppression node
        if node.op_type == 'NonMaxSuppression':
            center_point_box = 0
            max_output_boxes_per_class = 1000000
            iou_threshold = 0.3
            score_threshold = 0.0
            offset = 0
            for attribute in node_attributes:
                if attribute.name == 'center_point_box':
                    center_point_box = attribute.i
                elif attribute.name == 'offset':
                    offset = attribute.i

            if len(node_inputs) >= 3:
                max_output_boxes_per_class = parse_data(
                    node_inputs[2], np.int64, max_output_boxes_per_class)
                mark_nodes_to_remove(node_inputs[2])

            if len(node_inputs) >= 4:
                iou_threshold = parse_data(node_inputs[3], np.float32,
                                           iou_threshold)
                mark_nodes_to_remove(node_inputs[3])

            if len(node_inputs) >= 5:
                score_threshold = parse_data(node_inputs[4], np.float32)
                mark_nodes_to_remove(node_inputs[4])

            new_node = onnx.helper.make_node(
                'NonMaxSuppression',
                node_inputs[:2],
                node_outputs,
                name=node_name,
                center_point_box=center_point_box,
                max_output_boxes_per_class=max_output_boxes_per_class,
                iou_threshold=iou_threshold,
                score_threshold=score_threshold,
                offset=offset)

            for output in node_outputs:
                if output in node_dict:
                    node_dict[output] = new_node
            nodes.insert(idx, new_node)
            nodes.remove(node)
        elif node.op_type == 'InstanceNormalization':
            # directly change op name
            node.op_type = 'MMCVInstanceNormalization'

    for node_name in nodes_name_to_remove:
        nodes.remove(node_dict[node_name])

    return onnx_model
