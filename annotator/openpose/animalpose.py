import cv2
import numpy as np

from .cv_ox_det import inference_detector
from .cv_ox_pose import inference_pose

from typing import Optional


def drawBetweenKeypoints(pose_img, keypoints, indexes, color, scaleFactor):
    ind0 = indexes[0] - 1
    ind1 = indexes[1] - 1

    point1 = (keypoints[ind0][0], keypoints[ind0][1])
    point2 = (keypoints[ind1][0], keypoints[ind1][1])

    thickness = int(5 // scaleFactor)

    cv2.line(
        pose_img,
        (int(point1[0]), int(point1[1])),
        (int(point2[0]), int(point2[1])),
        color,
        thickness,
    )


def drawBetweenKeypointsList(
    pose_img, keypoints, keypointPairsList, colorsList, scaleFactor
):
    for ind, keypointPair in enumerate(keypointPairsList):
        drawBetweenKeypoints(
            pose_img, keypoints, keypointPair, colorsList[ind], scaleFactor
        )


class AnimalPose:
    def __init__(
        self,
        onnx_det: str,
        onnx_pose: str,
    ):
        self.onnx_det = onnx_det
        self.onnx_pose = onnx_pose
        self.model_input_size = (256, 256)

        # Always loads to CPU
        device = "cpu"
        providers = ["CPUExecutionProvider"]

        import onnxruntime as ort

        self.session_det = ort.InferenceSession(onnx_det, providers=providers)

        self.session_pose = ort.InferenceSession(onnx_pose, providers=providers)

    def __call__(self, oriImg) -> Optional[np.ndarray]:
        detect_classes = list(
            range(14, 23 + 1)
        )  # https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml

        det_result = inference_detector(
            self.session_det,
            oriImg,
            detect_classes=detect_classes,
        )

        if (det_result is None) or (det_result.shape[0] == 0):
            openpose_dict = {
                "version": "ap10k",
                "animals": [],
                "canvas_height": oriImg.shape[0],
                "canvas_width": oriImg.shape[1],
            }
            return np.zeros_like(oriImg), openpose_dict

        keypoint_sets, scores = inference_pose(
            self.session_pose,
            det_result,
            oriImg,
            self.model_input_size,
        )

        animal_kps_scores = []
        pose_img = np.zeros((oriImg.shape[0], oriImg.shape[1], 3), dtype=np.uint8)
        for idx, keypoints in enumerate(keypoint_sets):
            # don't use keypoints that go outside the frame in calculations for the center
            interorKeypoints = keypoints[
                ((keypoints[:, 0] > 0) & (keypoints[:, 0] < oriImg.shape[1]))
                & ((keypoints[:, 1] > 0) & (keypoints[:, 1] < oriImg.shape[0]))
            ]

            xVals = interorKeypoints[:, 0]
            yVals = interorKeypoints[:, 1]

            minX = np.amin(xVals)
            minY = np.amin(yVals)
            maxX = np.amax(xVals)
            maxY = np.amax(yVals)

            poseSpanX = maxX - minX
            poseSpanY = maxY - minY

            # find mean center

            xSum = np.sum(xVals)
            ySum = np.sum(yVals)

            xCenter = xSum // xVals.shape[0]
            yCenter = ySum // yVals.shape[0]
            center_of_keypoints = (xCenter, yCenter)

            # order of the keypoints for AP10k and a standardized list of colors for limbs
            keypointPairsList = [
                (1, 2),
                (2, 3),
                (1, 3),
                (3, 4),
                (4, 9),
                (9, 10),
                (10, 11),
                (4, 6),
                (6, 7),
                (7, 8),
                (4, 5),
                (5, 15),
                (15, 16),
                (16, 17),
                (5, 12),
                (12, 13),
                (13, 14),
            ]
            colorsList = [
                (255, 255, 255),
                (100, 255, 100),
                (150, 255, 255),
                (100, 50, 255),
                (50, 150, 200),
                (0, 255, 255),
                (0, 150, 0),
                (0, 0, 255),
                (0, 0, 150),
                (255, 50, 255),
                (255, 0, 255),
                (255, 0, 0),
                (150, 0, 0),
                (255, 255, 100),
                (0, 150, 0),
                (255, 255, 0),
                (150, 150, 150),
            ]  # 16 colors needed

            drawBetweenKeypointsList(
                pose_img, keypoints, keypointPairsList, colorsList, scaleFactor=1.0
            )
            score = scores[idx, ..., None]
            score[score > 1.0] = 1.0
            score[score < 0.0] = 0.0
            animal_kps_scores.append(np.concatenate((keypoints, score), axis=-1))

        openpose_dict = {
            "version": "ap10k",
            "animals": [
                [v for x, y, c in keypoints.tolist() for v in (x, y, c)]
                for keypoints in animal_kps_scores
            ],
            "canvas_height": oriImg.shape[0],
            "canvas_width": oriImg.shape[1],
        }
        return pose_img, openpose_dict
