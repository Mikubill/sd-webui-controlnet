# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np
import os

from .cv_ox_det import inference_detector as inference_yolox
from .yolo_nas import inference_detector as inference_yolo_nas
from .cv_ox_pose import inference_pose

from typing import List, Optional
from .types import PoseResult, BodyResult, Keypoint
from .util import guess_onnx_input_shape_dtype

ONNX_PROVIDERS = ["CUDAExecutionProvider", "DirectMLExecutionProvider", "OpenVINOExecutionProvider", "ROCMExecutionProvider"]
SUPPORT_PROVIDERS = []
def check_ort_gpu():
    try:
        import onnxruntime as ort
        for provider in ONNX_PROVIDERS:
            if provider in ort.get_available_providers():
                SUPPORT_PROVIDERS.append(provider)
                return True
        return False
    except:
        return False
        
#Global caching as the startup of onnxruntime is a bit slow
ort_session_det, ort_session_pose = None, None
cached_onnx_det_name, cached_onnx_pose_name = '', ''

class Wholebody:
    def __init__(self, onnx_det: str, onnx_pose: str):
        global ort_session_det, ort_session_pose, cached_onnx_det_name, cached_onnx_pose_name
        pose_filename = os.path.basename(onnx_pose)
        det_filename = os.path.basename(onnx_det)
        if check_ort_gpu():
            import onnxruntime as ort
            if pose_filename != cached_onnx_pose_name and ort_session_pose is None:
                print(f"DWPose: Caching pose session {pose_filename}...")
                SUPPORT_PROVIDERS.append('CPUExecutionProvider')
                ort_session_pose = ort.InferenceSession(onnx_pose, providers=SUPPORT_PROVIDERS)
                cached_onnx_pose_name = pose_filename
            
            if det_filename != cached_onnx_det_name or ort_session_det is None:
                print(f"DWPose: Caching bbox detection session {det_filename}...")
                ort_session_det = ort.InferenceSession(onnx_det, providers=SUPPORT_PROVIDERS)
                cached_onnx_det_name = det_filename

            self.session_det = ort_session_det
            self.session_pose = ort_session_pose
            return
        
        # Always loads to CPU to avoid building OpenCV.
        device = 'cpu'
        backend = cv2.dnn.DNN_BACKEND_OPENCV if device == 'cpu' else cv2.dnn.DNN_BACKEND_CUDA
        # You need to manually build OpenCV through cmake to work with your GPU.
        providers = cv2.dnn.DNN_TARGET_CPU if device == 'cpu' else cv2.dnn.DNN_TARGET_CUDA

        self.session_det = cv2.dnn.readNetFromONNX(onnx_det)
        self.session_det.setPreferableBackend(backend)
        self.session_det.setPreferableTarget(providers)

        self.session_pose = cv2.dnn.readNetFromONNX(onnx_pose)
        self.session_pose.setPreferableBackend(backend)
        self.session_pose.setPreferableTarget(providers)
        cached_onnx_pose_name = pose_filename
        cached_onnx_det_name = det_filename
    
    def __call__(self, oriImg) -> Optional[np.ndarray]:
        pose_input_size, pose_dtype = guess_onnx_input_shape_dtype(cached_onnx_pose_name)
        inference_detector = inference_yolox if "yolox" in cached_onnx_det_name else inference_yolo_nas
        
        #FP16 and INT8 YOLO NAS accept uint8 input
        det_result = inference_detector(self.session_det, oriImg, detect_classes=[0], dtype=np.float32 if "yolox" in cached_onnx_det_name else np.uint8)
        if det_result is None:
            return None

        keypoints, scores = inference_pose(self.session_pose, det_result, oriImg, pose_input_size, pose_dtype)

        keypoints_info = np.concatenate(
            (keypoints, scores[..., None]), axis=-1)
        # compute neck joint
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
        # neck score when visualizing pred
        neck[:, 2:4] = np.logical_and(
            keypoints_info[:, 5, 2:4] > 0.3,
            keypoints_info[:, 6, 2:4] > 0.3).astype(int)
        new_keypoints_info = np.insert(
            keypoints_info, 17, neck, axis=1)
        mmpose_idx = [
            17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3
        ]
        openpose_idx = [
            1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17
        ]
        new_keypoints_info[:, openpose_idx] = \
            new_keypoints_info[:, mmpose_idx]
        keypoints_info = new_keypoints_info

        return keypoints_info

    @staticmethod
    def format_result(keypoints_info: Optional[np.ndarray]) -> List[PoseResult]:
        def format_keypoint_part(
            part: np.ndarray,
        ) -> Optional[List[Optional[Keypoint]]]:
            keypoints = [
                Keypoint(x, y, score, i) if score >= 0.3 else None
                for i, (x, y, score) in enumerate(part)
            ]
            return (
                None if all(keypoint is None for keypoint in keypoints) else keypoints
            )

        def total_score(keypoints: Optional[List[Optional[Keypoint]]]) -> float:
            return (
                sum(keypoint.score for keypoint in keypoints if keypoint is not None)
                if keypoints is not None
                else 0.0
            )

        pose_results = []
        if keypoints_info is None:
            return pose_results

        for instance in keypoints_info:
            body_keypoints = format_keypoint_part(instance[:18]) or ([None] * 18)
            left_hand = format_keypoint_part(instance[92:113])
            right_hand = format_keypoint_part(instance[113:134])
            face = format_keypoint_part(instance[24:92])

            # Openpose face consists of 70 points in total, while DWPose only
            # provides 68 points. Padding the last 2 points.
            if face is not None:
                # left eye
                face.append(body_keypoints[14])
                # right eye
                face.append(body_keypoints[15])

            body = BodyResult(
                body_keypoints, total_score(body_keypoints), len(body_keypoints)
            )
            pose_results.append(PoseResult(body, left_hand, right_hand, face))

        return pose_results
