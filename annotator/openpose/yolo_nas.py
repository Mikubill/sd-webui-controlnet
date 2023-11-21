# Source: https://github.com/Hyuto/yolo-nas-onnx/tree/master/yolo-nas-py
# Inspired from: https://github.com/Deci-AI/super-gradients/blob/3.1.1/src/super_gradients/training/processing/processing.py

import numpy as np
import cv2

def preprocess(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r

def inference_detector(session, oriImg, detect_classes=[0], dtype=np.uint8):
    """
    This function is only compatible with onnx models exported from the new API with built-in NMS
    ```py
    from super_gradients.conversion.conversion_enums import ExportQuantizationMode
    from super_gradients.common.object_names import Models
    from super_gradients.training import models

    model = models.get(Models.YOLO_NAS_L, pretrained_weights="coco")

    export_result = model.export(
        "yolo_nas/yolo_nas_l_fp16.onnx",
        quantization_mode=ExportQuantizationMode.FP16,
        device="cuda"
    )
    ```
    """
    input_shape = (640,640)
    img, ratio = preprocess(oriImg, input_shape)
    input = img[None, :, :, :]
    input = input.astype(dtype)
    if "InferenceSession" in type(session).__name__:
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: input})
    else:
        outNames = session.getUnconnectedOutLayersNames()
        session.setInput(input)
        output = session.forward(outNames)
    num_preds, pred_boxes, pred_scores, pred_classes = output
    num_preds = num_preds[0,0]
    if num_preds == 0:
        return None
    idxs = np.where((np.isin(pred_classes[0, :num_preds], detect_classes)) & (pred_scores[0, :num_preds] > 0.3))
    if (len(idxs) == 0) or (idxs[0].size == 0):
        return None
    return pred_boxes[0, idxs].squeeze(axis=0) / ratio
