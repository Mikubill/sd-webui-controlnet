import torch
import shutil
import os
import copy
import modules.scripts as scripts

from torchvision import transforms
from modules import images
from modules.processing import process_images
from modules.shared import opts

import gradio as gr
import cv2
from PIL import Image

def automatic_prompt(image,device):
    """
    If the user wants to use automatic_prompt, necessary load the model. 
    Therefore, it is needed to clone BLIP model.
    
    Need to clone the BLIP repositry:
    from BLIP.models.blip import blip_decoder
    !pip install pytorch_pretrained_bert --upgrade
    !git clone https://github.com/salesforce/BLIP
    """
    
    image_size = 512
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'

    model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device)
    transform = transforms.Compose([
            transforms.Resize((image_size,image_size),interpolation=transforms.functional.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ]) 
    image=transform(image.resize((image_size,image_size))).unsqueeze(0).to(device)  
    with torch.no_grad():
        # beam search
        caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5) 
        # nucleus sampling
        # caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5) 
    return caption[0]

class Script(scripts.Script):
    def title(self):
        return "automatic prompt"

    def show(self,is_img2img):
        return

    def ui(self, is_img2img):
        ctrls_group = ()
        max_models = opts.data.get("control_net_max_models_num", 1)
        with gr.Group():
            with gr.Accordion("ControlNet-AUTOMATIC", open = False):
                with gr.Tabs():
                    for i in range(max_models):
                        with gr.Tab(f"ControlNet-{i}", open=False):
                            ctrls_group += (gr.Checkbox(label="Automatic Mode", value=False), )

        return ctrls_group

    def run(self, p, *args):
        video_num = opts.data.get("control_net_max_models_num", 1)
        video_list = [get_all_frames(video) for video in args[:video_num]]
        duration, = args[video_num:]

        frame_num = get_min_frame_num(video_list)
        if frame_num > 0:
            output_image_list = []
            for frame in range(frame_num):
                copy_p = copy.copy(p)
                copy_p.control_net_input_image = []
                for video in video_list:
                    if video is None:
                        continue
                    copy_p.control_net_input_image.append(video[frame])
                proc = process_images(copy_p)
                img = proc.images[0]
                output_image_list.append(img)
                copy_p.close()
            save_gif(p.outpath_samples, output_image_list, "animation", duration)
            proc.images = [p.outpath_samples + "/animation.gif"]

        else:
            proc = process_images(p)

        return proc