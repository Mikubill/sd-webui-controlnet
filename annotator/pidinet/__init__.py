import os
import torch
import numpy as np
from einops import rearrange
from annotator.pidinet.model import pidinet
from modules import extensions, devices
from scripts.utils import load_state_dict

netNetwork = None
remote_model_path = "https://github.com/TencentARC/T2I-Adapter/raw/main/models/table5_pidinet.pth"
modeldir = os.path.join(extensions.extensions_dir, "sd-webui-controlnet", "annotator", "pidinet")

def apply_pidinet(input_image):
    global netNetwork
    if netNetwork is None:
        modelpath = os.path.join(modeldir, "table5_pidinet.pth")
        if not os.path.exists(modelpath):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(remote_model_path, model_dir=modeldir)
        netNetwork = pidinet()
        ckp = load_state_dict(modelpath)
        netNetwork.load_state_dict({k.replace('module.',''):v for k, v in ckp.items()})
        
    netNetwork.to(devices.get_device_for("controlnet")).eval()
    assert input_image.ndim == 3
    input_image = input_image[:, :, ::-1].copy()
    with torch.no_grad():
        image_pidi = torch.from_numpy(input_image).float().to(devices.get_device_for("controlnet"))
        image_pidi = image_pidi / 255.0
        image_pidi = rearrange(image_pidi, 'h w c -> 1 c h w')
        edge = netNetwork(image_pidi)[-1]
        edge = edge>0.5
        edge = (edge * 255.0).clip(0, 255).cpu().numpy().astype(np.uint8)
        
    return edge[0][0] 

def unload_pid_model():
    global netNetwork
    if netNetwork is not None:
        netNetwork.cpu()