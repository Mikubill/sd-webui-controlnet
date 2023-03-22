from .utils import colorize
from PIL import Image
import torch
import os

model_zoedepth = None

def unload_zoedepth():
    global model_zoedepth
    if model_zoedepth is not None:
        model_zoedepth = model_zoedepth.cpu()

def apply_zoedepth(img, res=512):
    global model_zoedepth

    if model_zoedepth is None:
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        if not os.path.exists("DPT_BEiT_L_384.txt"):
            torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)  # Triggers fresh download of MiDaS repo
            open("DPT_BEiT_L_384.txt","w")
        model_zoedepth = torch.hub.load('isl-org/ZoeDepth', "ZoeD_NK", pretrained=True, force_reload=True).to(DEVICE).eval()

    img_zoedepth = model_zoedepth.infer_pil(img)
    img_zoedepth_colored_depth = Image.fromarray(colorize(img_zoedepth)).convert('L')

    return img_zoedepth_colored_depth

def apply_colored_zoedepth(img, res=512):
    global model_zoedepth


    if model_zoedepth is None:
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        if not os.path.exists("DPT_BEiT_L_384.txt"):
            torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)  # Triggers fresh download of MiDaS repo
            open("DPT_BEiT_L_384.txt","w")
        model_zoedepth = torch.hub.load('isl-org/ZoeDepth', "ZoeD_NK", pretrained=True, force_reload=True).to(DEVICE).eval()

    img_zoedepth = model_zoedepth.infer_pil(img)
    img_zoedepth_colored_depth = colorize(img_zoedepth)

    return img_zoedepth_colored_depth
