import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import numpy as np
from . import util
from .body import Body
from .hand import Hand
from .face import Face
from modules import devices
from annotator.annotator_path import models_path

body_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/body_pose_model.pth"
hand_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/hand_pose_model.pth"
face_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/facenet.pth"

def draw_pose(pose, H, W, draw_body=True, draw_hand=True, draw_face=True):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    if draw_body:
        canvas = util.draw_bodypose(canvas, candidate, subset)

    if draw_hand:
        canvas = util.draw_handpose(canvas, hands)

    if draw_face:
        canvas = util.draw_facepose(canvas, faces)

    return canvas

    
class OpenposeDetector:
    model_dir = os.path.join(models_path, "openpose")

    def __init__(self):
        self.device = devices.get_device_for("controlnet")
        self.body_estimation = None
        self.hand_estimation = None
        self.face_estimation = None
        
    def load_model(self):
        body_modelpath = os.path.join(self.model_dir, "body_pose_model.pth")
        hand_modelpath = os.path.join(self.model_dir, "hand_pose_model.pth")
        face_modelpath = os.path.join(self.model_dir, "facenet.pth")

        if not os.path.exists(body_modelpath):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(body_model_path, model_dir=self.model_dir)

        if not os.path.exists(hand_modelpath):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(hand_model_path, model_dir=self.model_dir)

        if not os.path.exists(face_modelpath):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(face_model_path, model_dir=self.model_dir)

        self.body_estimation = Body(body_modelpath)
        self.hand_estimation = Hand(hand_modelpath)
        self.face_estimation = Face(face_modelpath)

    def unload_model(self):
        if self.body_estimation is not None:
            self.body_estimation.model.to("cpu")
            self.hand_estimation.model.to("cpu")
            self.face_estimation.model.to("cpu")
        
    def __call__(self, oriImg, include_body=True, include_hand=False, include_face=False, return_is_index=False):
        if self.body_estimation is None:
            self.load_model()
            
        self.body_estimation.model.to(self.device)
        self.hand_estimation.model.to(self.device)
        self.face_estimation.model.to(self.device)

        self.body_estimation.cn_device = self.device
        self.hand_estimation.cn_device = self.device
        self.face_estimation.cn_device = self.device

        oriImg = oriImg[:, :, ::-1].copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            candidate, subset = self.body_estimation(oriImg)
            hands = []
            faces = []
            if include_hand:
                # Hand
                hands_list = util.handDetect(candidate, subset, oriImg)
                for x, y, w, is_left in hands_list:
                    peaks = self.hand_estimation(oriImg[y:y+w, x:x+w, :]).astype(np.float32)
                    if peaks.ndim == 2 and peaks.shape[1] == 2:
                        peaks[:, 0] = np.where(peaks[:, 0] < 1e-6, -1, peaks[:, 0] + x) / float(W)
                        peaks[:, 1] = np.where(peaks[:, 1] < 1e-6, -1, peaks[:, 1] + y) / float(H)
                        hands.append(peaks.tolist())
            
            if include_face:
                # Face
                faces_list = util.faceDetect(candidate, subset, oriImg)
                for x, y, w in faces_list:
                    heatmaps = self.face_estimation(oriImg[y:y+w, x:x+w, :])
                    peaks = self.face_estimation.compute_peaks_from_heatmaps(heatmaps).astype(np.float32)
                    if peaks.ndim == 2 and peaks.shape[1] == 2:
                        peaks[:, 0] = np.where(peaks[:, 0] < 1e-6, -1, peaks[:, 0] + x) / float(W)
                        peaks[:, 1] = np.where(peaks[:, 1] < 1e-6, -1, peaks[:, 1] + y) / float(H)
                        faces.append(peaks.tolist())
                        
            if candidate.ndim == 2 and candidate.shape[1] == 4:
                candidate = candidate[:, :2]
                candidate[:, 0] /= float(W)
                candidate[:, 1] /= float(H)
                
            bodies = dict(candidate=candidate.tolist(), subset=subset.tolist())
            pose = dict(bodies=bodies, hands=hands, faces=faces)
            if return_is_index:
                return pose
            else:
                return draw_pose(pose, H, W, draw_body=include_body, draw_hand=include_hand, draw_face=include_face)