import modules.scripts as scripts
import gradio as gr
import cv2
import shutil
import os
import copy

from modules import images
from modules.processing import process_images, Processed
from modules.processing import Processed
from modules.shared import opts, cmd_opts, state
from PIL import Image


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
        input_video = gr.Video(format='mp4', source='upload')
        return [input_video]
  

# This is where the additional processing is implemented. The parameters include
# self, the model object "p" (a StableDiffusionProcessing class, see
# processing.py), and the parameters returned by the ui method.
# Custom functions can be defined here, and additional libraries can be imported 
# to be used in processing. The return value should be a Processed object, which is
# what is returned by the process_images method.
    def run(self, p, input_video):
        def get_all_frames(video_path):
            cap = cv2.VideoCapture(video_path)
            frame_list = []
            if not cap.isOpened():
                return
            digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
            n = 0
            while True:
                ret, frame = cap.read()
                if ret:
                    frame_list.append(frame)
                else:
                    return frame_list
        def save_gif(path, image_list):
            tmp_dir = path + "/tmp/" 
            os.mkdir(tmp_dir)
            path_list = []
            imgs = []
            for i, image in enumerate(image_list):
                images.save_image(image, tmp_dir, f"output_{i}")
                path_list.append(tmp_dir + f"output_{i}-0000.png")
            for i in range(len(path_list)):
                img = Image.open(path_list[i])
                imgs.append(img)

            imgs[0].save(path + "/animation.gif", save_all=True, append_images=imgs[1:], optimize=False, duration=5, loop=0)
            shutil.rmtree(tmp_dir)

        if input_video is not None:
            output_list = []
            input_list = get_all_frames(input_video)

            for input_image in input_list:
                copy_p = copy.copy(p)
                copy_p.control_net_allow_script_control = True
                copy_p.control_net_input_image = input_image
                proc = process_images(copy_p)
                output_list.append(proc.images[0])
            save_gif(p.outpath_samples, output_list)
            proc.images = [p.outpath_samples + "/animation.gif"]

        else:
            proc = process_images(p)
        
        
        
        
        return proc