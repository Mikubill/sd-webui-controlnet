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


class NormalDsineDetector:
    model_dir = os.path.join(models_path, "normal_dsine")

    def __init__(self):
        self.model = None
        self.device = devices.get_device_for("controlnet")

    def load_model(self):
        remote_model_path = "https://huggingface.co/bdsqlsz/qinglong_controlnet-lllite/blob/main/Annotators/dsine.pt"
        modelpath = os.path.join(self.model_dir, "dsine.pt")
        if not os.path.exists(modelpath):
            from scripts.utils import load_file_from_url
            load_file_from_url(remote_model_path, model_dir=self.model_dir)
        model = DSINE(args)
        model.pixel_coords = model.pixel_coords.to(self.device)
        model = utils.load_checkpoint(modelpath, model)
        model.eval()
        self.model = model.to(self.device)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def unload_model(self):
        if self.model is not None:
            self.model.cpu()

    def __call__(self, input_image):
        if self.model is None:
            self.load_model()

        self.model.to(self.device)
        assert input_image.ndim == 3
        image_normal = input_image
        with torch.no_grad():
            image_normal = torch.from_numpy(image_normal).float().to(self.device)
            image_normal = image_normal / 255.0
            image_normal = rearrange(image_normal, 'h w c -> 1 c h w')
            image_normal = self.norm(image_normal)

            _, _, orig_H, orig_W = image_normal.shape
           
            l, r, t, b = utils.pad_input(orig_H, orig_W)
            img = F.pad(img, (l, r, t, b), mode="constant", value=0.0)
            img = self.norm(img)
            
            intrins = utils.get_intrins_from_fov(new_fov=60.0, H=orig_H, W=orig_W, device=self.device).unsqueeze(0)
            
            intrins[:, 0, 2] += l
            intrins[:, 1, 2] += t
            
            normal = self.model(img, intrins=intrins)[-1]
            normal = normal[:, :, t:t+orig_H, l:l+orig_W]

            normal = ((normal + 1) * 0.5).clip(0, 1)

            normal = rearrange(normal[0], 'c h w -> h w c').cpu().numpy()
            normal_image = (normal * 255.0).clip(0, 255).astype(np.uint8)

            return normal_image