"""
Hello, welcome on board,
"""
from __future__ import print_function

import os
import cv2
import numpy as np

import torch

from ted import TED  # TEED architecture
from einops import rearrange
from modules import devices
from .util import load_model,safe_step
from .annotator_path import models_path

class TEEDDector:
    """https://github.com/xavysp/TEED"""

    model_dir = os.path.join(models_path, "TEED")

    def __init__(self):
        self.device = devices.get_device_for("controlnet")
        self.model = TED().to(self.device).eval()
        remote_url = os.environ.get(
            "CONTROLNET_TEED_MODEL_URL",
            "https://huggingface.co/bdsqlsz/qinglong_controlnet-lllite/resolve/main/Annotators/7_model.pth",
        )
        model_path = load_model(
            "7_model.pth", remote_url=remote_url, model_dir=self.model_dir
        )
        self.model.load_state_dict(torch.load(model_path))

    def unload_model(self):
        if self.model is not None:
            self.model.cpu()

    def __call__(self, image: np.ndarray, is_safe: bool = True) -> np.ndarray:
        if self.model is None:
            self.load_model()
        self.model.to(self.device)

        H, W, _ = image.shape
        with torch.no_grad():
            image_teed = torch.from_numpy(image.copy()).float().to(self.device)
            image_teed = rearrange(image_teed, 'h w c -> 1 c h w')
            edges = self.model(image_teed)
            edges = [e.detach().cpu().numpy().astype(np.float32)[0, 0] for e in edges]
            edges = [cv2.resize(e, (W, H), interpolation=cv2.INTER_LINEAR) for e in edges]
            edges = np.stack(edges, axis=2)
            edge = 1 / (1 + np.exp(-np.mean(edges, axis=2).astype(np.float64)))
            if is_safe:
                edge = safe_step(edge)
            edge = cv2.bitwise_not(edge).astype(np.uint8)
            return edge