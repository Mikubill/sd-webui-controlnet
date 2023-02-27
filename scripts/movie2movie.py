import copy
import os
import shutil

import cv2
import gradio as gr
import modules.scripts as scripts

from modules import images
from modules.processing import process_images
from modules.shared import opts
from PIL import Image


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

def save_gif(path, image_list, name, duration):
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

    imgs[0].save(path + f"/{name}.gif", save_all=True, append_images=imgs[1:], optimize=False, duration=duration, loop=0)
    

class Script(scripts.Script):  
    
    def title(self):
        return "controlnet m2m"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        # How the script's is displayed in the UI. See https://gradio.app/docs/#components
        # for the different UI components you can use and how to create them.
        # Most UI components can return a value, such as a boolean for a checkbox.
        # The returned values are passed to the run method as parameters.
        
        ctrls_group = ()
        max_models = opts.data.get("control_net_max_models_num", 1)

        with gr.Group():
            with gr.Accordion("ControlNet-M2M", open = False):
                with gr.Tabs():
                    for i in range(max_models):
                        with gr.Tab(f"ControlNet-{i}", open=False):
                            ctrls_group += (gr.Video(format='mp4', source='upload', elem_id = f"video_{i}"), )

                duration = gr.Slider(label=f"Duration", value=50.0, minimum=10.0, maximum=200.0, step=10, interactive=True) 
        ctrls_group += (duration,)

        return ctrls_group

    def run(self, p, *args):
        # This is where the additional processing is implemented. The parameters include
        # self, the model object "p" (a StableDiffusionProcessing class, see
        # processing.py), and the parameters returned by the ui method.
        # Custom functions can be defined here, and additional libraries can be imported 
        # to be used in processing. The return value should be a Processed object, which is
        # what is returned by the process_images method.

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