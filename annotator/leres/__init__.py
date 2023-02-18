import cv2
import numpy as np
import torch
import os
from modules import extensions

from einops import rearrange
from modules import devices
from torchvision.transforms import Compose, transforms
from modules.paths import models_path

# AdelaiDepth/LeReS imports
from .leres.multi_depth_model_woauxi import RelDepthModel
from .leres.net_tools import strip_prefix_if_present

base_model_path = modeldir = os.path.join(models_path, "ControlNet-Annotator")
remote_model_path = "https://cloudstor.aarnet.edu.au/plus/s/lTIJF4vrvHCAI31/download"

model = None

def unload_leres_model():
    global model
    if model is not None:
        model = model.cpu()

def scale_torch(img):
	"""
	Scale the image and output it in torch.tensor.
	:param img: input rgb is in shape [H, W, C], input depth/disp is in shape [H, W]
	:param scale: the scale factor. float
	:return: img. [C, H, W]
	"""
	if len(img.shape) == 2:
		img = img[np.newaxis, :, :]
	if img.shape[2] == 3:
		transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406) , (0.229, 0.224, 0.225) )])
		img = transform(img.astype(np.float32))
	else:
		img = img.astype(np.float32)
		img = torch.from_numpy(img)
	return img

def estimateleres(img, model, w, h):
	# leres transform input
	rgb_c = img[:, :, ::-1].copy()
	A_resize = cv2.resize(rgb_c, (w, h))
	img_torch = scale_torch(A_resize)[None, :, :, :] 
	
	# compute
	with torch.no_grad():
		img_torch = img_torch.cuda()
		prediction = model.depth_model(img_torch)

	prediction = prediction.squeeze().cpu().numpy()
	prediction = cv2.resize(prediction, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

	return prediction

def apply_leres(input_image, a):
    global model
    if model is None:
        model_path = os.path.join(base_model_path, "res101.pth")
        if not os.path.exists(model_path):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(remote_model_path, model_dir=base_model_path)
            os.rename(os.path.join(base_model_path, 'download'), model_path)
        checkpoint = torch.load(model_path)
        model = RelDepthModel(backbone='resnext101')
        # CPU support? add condition here
        # checkpoint = torch.load(model_path,map_location=torch.device('cpu'))
        model.load_state_dict(strip_prefix_if_present(checkpoint['depth_model'], "module."), strict=True)
        del checkpoint
    
    if devices.get_device_for("controlnet").type != 'mps':
        model = model.to(devices.get_device_for("controlnet"))
    
    assert input_image.ndim == 3
    image_depth = input_image
    with torch.no_grad():

        depth = estimateleres(image_depth, model, 512, 512) # add support for resizing/cropping

        numbytes=2
        depth_min = depth.min()
        depth_max = depth.max()
        max_val = (2**(8*numbytes))-1

        # check output before normalizing and mapping to 16 bit
        if depth_max - depth_min > np.finfo("float").eps:
            out = max_val * (depth - depth_min) / (depth_max - depth_min)
        else:
            out = np.zeros(depth.shape)
        
        # single channel, 16 bit image
        depth_image = out.astype("uint16")
        depth_image = cv2.bitwise_not(depth_image)

        # convert to uint8
        depth_image = cv2.convertScaleAbs(depth_image, alpha=(255.0/65535.0))

        return depth_image, None
