import os
from typing import List

import torch
from PIL import Image

import sys

import numpy as np
from modules import devices
from .face_analysis import FaceAnalysis
from .face_align import norm_crop
class FaceidEmbedsEstimator:

    def __init__(self):
        self.model = None
        self.device = devices.get_device_for("controlnet")
        self.model = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.model.prepare(ctx_id=0, det_size=(640, 640))

    def __call__(self, input_image):
        faces = self.model.get(input_image)
        faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
        face_image = norm_crop(input_image, landmark=faces[0].kps, image_size=224)
        return faceid_embeds, face_image