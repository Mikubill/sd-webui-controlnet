import numpy as np
from PIL import Image, ImageEnhance

def apply_colorization(img, brightness, contrast):
    pil_img = Image.fromarray(img.astype(np.uint8)).convert('L').convert('RGB')
    if brightness != 0: pil_img = ImageEnhance.Brightness(pil_img).enhance((brightness+100)/100)
    if   contrast != 0: pil_img = ImageEnhance.Contrast(pil_img).enhance((contrast+100)/100)
    return np.array(pil_img)