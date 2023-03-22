from utils import colorize
from PIL import Image
import torch

model_zoedepth = None

def unload_zoedepth():
    global model_zoedepth
    if model_zoedepth is not None:
        model_zoedepth = model_zoedepth.cpu()

def apply_zoedepth(img, res=512):
    global model_zoedepth
    if model_zoedepth is None:
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_zoedepth = torch.hub.load('isl-org/ZoeDepth', "ZoeD_NK", pretrained=True).to(DEVICE).eval()

    img_zoedepth = model_zoedepth.infer_pil((img)
    img_zoedepth_colored_depth = colorize(img_zoedepth)

    return img_zoedepth_colored_depth