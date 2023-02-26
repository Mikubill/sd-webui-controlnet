import shutil
import os
import copy


import modules.scripts as scripts
from modules import images
from modules.processing import process_images
from modules.shared import opts

import gradio as gr
import cv2
from PIL import Image
import huggingface_hub
import onnxruntime as rt
import numpy as np

# Declare Execution Providers
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

# Download and host the model
model_path = huggingface_hub.hf_hub_download(
    "skytnt/anime-seg", "isnetis.onnx")
rmbg_model = rt.InferenceSession(model_path, providers=providers)

# Function to get mask
def get_mask(img, s=1024):
    #Resize the img to a square shape with dimension s
    #Convert img pixel values from integers 0-255 to float 0-1
    img = (img / 255).astype(np.float32)
    #Get height and width of the image 
    h, w = h0, w0 = img.shape[:-1]
    #IF height is greater than width, set h as s and w as s*width/height
    #ELSE, set w as s and h as s*height/width
    h, w = (s, int(s * w / h)) if h > w else (int(s * h / w), s)
    #Calculate padding for height and width
    ph, pw = s - h, s - w
    #Create a 1024x1024x3 array of 0's   
    img_input = np.zeros([s, s, 3], dtype=np.float32)
    #Resize the original image to (w,h) and then pad with the calculated ph,pw
    img_input[ph // 2:ph // 2 + h, pw //
              2:pw // 2 + w] = cv2.resize(img, (w, h))
    #Change the axes
    img_input = np.transpose(img_input, (2, 0, 1))
    #Add an extra axis (1,0) 
    img_input = img_input[np.newaxis, :]
    #Run the model to get the mask
    mask = rmbg_model.run(None, {'img': img_input})[0][0]
    #Transpose axis
    mask = np.transpose(mask, (1, 2, 0))
    #Crop it to the images original dimensions (h0,w0)
    mask = mask[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w]
    #Resize the mask to original image size (h0,w0) 
    mask = cv2.resize(mask, (w0, h0))[:, :, np.newaxis]
    return mask

# Function to remove background
def rmbg_fn(img):
    #Call get_mask() to get the mask
    mask = get_mask(img)
    #Multiply the image and the mask together to get the output image
    img = (mask * img + 255 * (1 - mask)).astype(np.uint8)
    #Convert mask value back to int 0-255
    mask = (mask * 255).astype(np.uint8)
    #Concatenate the output image and mask
    img = np.concatenate([img, mask], axis=2, dtype=np.uint8)
    #Stacking 3 identical copies of the mask for displaying
    mask = mask.repeat(3, axis=2)
    return mask, img
    
def get_all_frames(video_path):
    if video_path is None:
        return None
    cap = cv2.VideoCapture(video_path)
    frame_list = []
    if not cap.isOpened():
        return
    while True:
        ret, frame = cap.read()
        if ret:
            frame_list.append(frame)
        else:
            return frame_list

def get_min_frame_num(video_list):
    min_frame_num = -1
    for video in video_list:
        if video is None:
            continue
        else:
            frame_num = len(video)
            print(frame_num)
            if min_frame_num < 0:
                min_frame_num = frame_num
            elif frame_num < min_frame_num:
                min_frame_num = frame_num
    return min_frame_num

def remove_background(proc):
    # Seperate the Background from the foreground
    nmask, nimg = rmbg_fn(np.array(proc.images[0]))
    # Change the image back to an image format
    img = Image.fromarray(nimg).convert("RGB") 
    return img

def save_gif(path, image_list, name):
    tmp_dir = path + "/tmp/" 
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.mkdir(tmp_dir)
    path_list = []
    imgs = []
    for i, image in enumerate(image_list):
        images.save_image(image, tmp_dir, f"output_{i}")
        path_list.append(tmp_dir + f"output_{i}-0000.png")
    for i in range(len(path_list)):
        img = Image.open(path_list[i])
        imgs.append(img)

    imgs[0].save(path + f"/{name}.gif", save_all=True, append_images=imgs[1:], optimize=False, duration=50, loop=0)
    


class Script(scripts.Script):  
# The title of the script. This is what will be displayed in the dropdown menu.
    def title(self):
        return "controlnet m2m"

# Determines when the script should be shown in the dropdown menu via the 
# returned value. As an example:
# is_img2img is True if the current tab is img2img, and False if it is txt2img.
# Thus, return is_img2img to only show the script on the img2img tab.
    def show(self, is_img2img):
        return True

# How the script's is displayed in the UI. See https://gradio.app/docs/#components
# for the different UI components you can use and how to create them.
# Most UI components can return a value, such as a boolean for a checkbox.
# The returned values are passed to the run method as parameters.
    def ui(self, is_img2img):
        ctrls_group = ()
        max_models = opts.data.get("control_net_max_models_num", 1)

        with gr.Group():
            with gr.Accordion("ControlNet-M2M", open = False):
                with gr.Tabs():
                    for i in range(max_models):
                        with gr.Tab(f"ControlNet-{i}", open=False):
                            ctrls_group += (gr.Video(format='mp4', source='upload', elem_id = f"video_{i}"), )

        return ctrls_group
  

# This is where the additional processing is implemented. The parameters include
# self, the model object "p" (a StableDiffusionProcessing class, see
# processing.py), and the parameters returned by the ui method.
# Custom functions can be defined here, and additional libraries can be imported 
# to be used in processing. The return value should be a Processed object, which is
# what is returned by the process_images method.
    def run(self, p, *args):
        video_num = opts.data.get("control_net_max_models_num", 1)
        video_list = [get_all_frames(video) for video in args[:video_num]]
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
            save_gif(p.outpath_samples, output_image_list, "animation")
            proc.images = [p.outpath_samples + "/animation.gif"]

        else:
            proc = process_images(p)
        
        return proc