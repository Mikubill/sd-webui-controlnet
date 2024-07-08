import os
import torch
import torch.nn.functional as F
import numpy as np

from einops import rearrange
from modules import devices
from annotator.annotator_path import models_path
import torchvision.transforms as transforms
import dsine.utils.utils as utils
from dsine.models.dsine import DSINE
from scripts.utils import resize_image_with_pad


class NormalDsineDetector:
    model_dir = os.path.join(models_path, "normal_dsine")

    def __init__(self):
        self.model = None
        self.device = devices.get_device_for("controlnet")

    def load_model(self):
        remote_model_path = "https://huggingface.co/bdsqlsz/qinglong_controlnet-lllite/resolve/main/Annotators/dsine.pt"
        modelpath = os.path.join(self.model_dir, "dsine.pt")
        if not os.path.exists(modelpath):
            from scripts.utils import load_file_from_url
            load_file_from_url(remote_model_path, model_dir=self.model_dir)
        model = DSINE()
        model.pixel_coords = model.pixel_coords.to(self.device)
        model = utils.load_checkpoint(modelpath, model)
        model.eval()
        self.model = model.to(self.device)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def unload_model(self):
        if self.model is not None:
            self.model.cpu()

    def __call__(self, input_image, new_fov=60.0, iterations=5, resulotion=512):
        if self.model is None:
            self.load_model()

        self.model.to(self.device)
        self.model.num_iter = iterations
        orig_H, orig_W = input_image.shape[:2]
        l, r, t, b = utils.pad_input(orig_H, orig_W)
        input_image, remove_pad = resize_image_with_pad(input_image, resulotion)
        assert input_image.ndim == 3
        image_normal = input_image
        with torch.no_grad():
            image_normal = torch.from_numpy(image_normal).float().to(self.device)
            image_normal = image_normal / 255.0
            image_normal = rearrange(image_normal, 'h w c -> 1 c h w')
            image_normal = self.norm(image_normal)
            
            intrins = utils.get_intrins_from_fov(new_fov=new_fov, H=orig_H, W=orig_W, device=self.device).unsqueeze(0)
            
            intrins[:, 0, 2] += l
            intrins[:, 1, 2] += t
            
            normal = self.model(image_normal, intrins=intrins)[-1]
            normal = normal[:, :, t:t+orig_H, l:l+orig_W]

            normal = ((normal + 1) * 0.5).clip(0, 1)

            normal = rearrange(normal[0], 'c h w -> h w c').cpu().numpy()
            normal_image = (normal * 255.0).clip(0, 255).astype(np.uint8)

            return remove_pad(normal_image)
