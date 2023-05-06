import torch
from transformers import CLIPProcessor, CLIPVisionModel
from modules import devices
import os
from annotator.annotator_path import clip_vision_path


remote_model_path = "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/pytorch_model.bin"
clip_path = clip_vision_path
print(f'ControlNet ClipVision location: {clip_path}')

clip_proc = None
clip_vision_model = None


def apply_clip(img):
    global clip_proc, clip_vision_model
    
    if clip_vision_model is None:
        modelpath = os.path.join(clip_path, 'pytorch_model.bin')
        if not os.path.exists(modelpath):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(remote_model_path, model_dir=clip_path)

        clip_proc = CLIPProcessor.from_pretrained(clip_path)
        clip_vision_model = CLIPVisionModel.from_pretrained(clip_path)

    with torch.no_grad():
        clip_vision_model = clip_vision_model.to(devices.get_device_for("controlnet"))
        style_for_clip = clip_proc(images=img, return_tensors="pt")['pixel_values']
        style_feat = clip_vision_model(style_for_clip.to(devices.get_device_for("controlnet")))['last_hidden_state']

    return style_feat


def unload_clip_model():
    global clip_proc, clip_vision_model
    if clip_vision_model is not None:
        clip_vision_model.cpu()