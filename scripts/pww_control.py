from collections import OrderedDict
from typing import Union, Dict, Any, Optional

from modules import shared, devices, script_callbacks, processing, masking, images
import gradio as gr
import numpy as np

import modules.scripts as scripts
from scripts.processor import *
from scripts.adapter import PlugableAdapter
# from scripts.hook import ControlParams, UnetHook
from scripts.hook_pww import ControlParams, UnetHook
from scripts import external_code
from modules.processing import StableDiffusionProcessingImg2Img
from PIL import Image


from scripts.controlnet import image_dict_from_unit, swap_img2img_pipeline, img2img_tab_tracker
from scripts.controlnet import Script as Script_cn
from scripts.hook_pww import ControlParams, UnetHook
from scripts.pww_utils import encode_text_color_inputs, hijack_CrossAttn
import math
import ast

MAX_NUM_COLORS = 8
NUM_PWW_PARAMS = 4


class Script(Script_cn):
    def __init__(self) -> None:
        super().__init__()

    def show(self, is_img2img):
        # if is_img2img:
            # return False
        return scripts.AlwaysVisible
        # return False

    def pww_uigroup(self, is_img2img):

        def create_canvas(h, w):
            return np.zeros(shape=(h, w, 3), dtype=np.uint8) + 255

        def extract_color_textboxes(color_map_image):
            # Get unique colors in color_map_image
            w, h = 15, 3

            colors = unique_colors(Image.fromarray(color_map_image))
            color_blocks = [Image.new("RGB", (w, h), color=color) for color in colors]
            # Append white blocks to color_blocks to fill up to MAX_NUM_COLORS
            num_missing_blocks = MAX_NUM_COLORS - len(color_blocks)
            white_block = Image.new("RGB", (w, h), color=(32, 32, 32))
            color_blocks += [white_block] * num_missing_blocks

            default_prompt = ["obj" for _ in range(len(colors))] + ["" for _ in range(len(colors), MAX_NUM_COLORS)]
            default_strength = ["0.5" for _ in range(len(colors))] + ["" for _ in range(len(colors), MAX_NUM_COLORS)]
            colors.extend([None] * num_missing_blocks)

            return (*color_blocks, *default_prompt, *default_strength, *colors)

        def unique_colors(image, threshold=0.01):
            colors = image.getcolors(image.size[0] * image.size[1])
            total_pixels = image.size[0] * image.size[1]
            unique_colors = []
            for count, color in colors:
                if count / total_pixels > threshold:
                    unique_colors.append(color)
            return unique_colors

        def collect_color_content(*args):
            n = len(args)
            chunk_size = n // 3
            colors, prompts, strengths = [args[i:i+chunk_size] for i in range(0, n, chunk_size)]
            content_collection = []
            for color, prompt, strength in zip(colors, prompts, strengths):
                if color is not None:
                    input_str = '%s:"%s,%s,-1"'%(color, prompt, strength)
                    content_collection.append(input_str)
            if len(content_collection) > 0:
                return "{%s}"%','.join(content_collection)
            else:
                return ""

        canvas_height, canvas_width = (self.img2img_h_slider, self.img2img_w_slider) if is_img2img else (self.txt2img_h_slider, self.txt2img_w_slider)
        segmentation_input_image = gr.Image(source='upload', type='numpy', tool='color-sketch', interactive=True)
        
        color_context = gr.Textbox(label="Color context", value='', interactive=True)
        weight_function = gr.Textbox(label="Weight function scale", value='0.2', interactive=True)
        with gr.Row():
            pww_enabled = gr.Checkbox(label='Enable', value=False)
            pww_create_button = gr.Button(value="Create blank canvas")
            pww_create_button.click(fn=create_canvas, inputs=[canvas_height, canvas_width], outputs=[segmentation_input_image])
            extract_color_boxes_button = gr.Button(value="Extract color content")
            generate_color_boxes_button = gr.Button(value="Generate color content")

        with gr.Accordion('Color context option', open=False):
            prompts = []
            strengths = []
            color_maps = []
            colors = [gr.Textbox(value="", visible=False) for i in range(MAX_NUM_COLORS)]
            for n in range(MAX_NUM_COLORS):
                with gr.Row():
                    color_maps.append(gr.Image(image=create_canvas(15,3), interactive=False, type='numpy'))
                    with gr.Column():
                        prompts.append(gr.Textbox(label="Prompt", interactive=True))
                    with gr.Column():
                        strengths.append(gr.Textbox(label="Strength", interactive=True))

        extract_color_boxes_button.click(fn=extract_color_textboxes, inputs=[segmentation_input_image], outputs=[*color_maps, *prompts, *strengths, *colors])
        generate_color_boxes_button.click(fn=collect_color_content, inputs=[*colors, *prompts, *strengths], outputs=[color_context])
        ctrls = (pww_enabled, color_context, weight_function, segmentation_input_image)
        return ctrls
    
    def ui(self, is_img2img):
        """this function should create gradio UI elements. See https://gradio.app/docs/#components
        The return value should be an array of all components that are used in processing.
        Values of those returned components will be passed to run() and process() functions.
        """
        self.infotext_fields = []
        controls = ()
        max_models = shared.opts.data.get("control_net_max_models_num", 1)
        with gr.Group():
            with gr.Accordion("ControlNet", open = False, elem_id="controlnet"):
                if max_models > 1:
                    with gr.Tabs():
                        for i in range(max_models):
                            with gr.Tab(f"Control Model - {i}"):
                                controls += (self.uigroup(f"ControlNet-{i}", is_img2img),)
                else:
                    with gr.Column():
                        controls += (self.uigroup(f"ControlNet", is_img2img),)
            with gr.Accordion('PaintWithWord', open=False):
                controls += self.pww_uigroup(is_img2img)            
        return controls

    def process(self, p, *args):
        """
        This function is called before processing begins for AlwaysVisible scripts.
        You can modify the processing object (p) here, inject hooks, etc.
        args contains all values returned by components from ui()
        """
        # Parse PwW arguments
        pww_args = args[-NUM_PWW_PARAMS:]
        args = args[:-NUM_PWW_PARAMS] 

        unet = p.sd_model.model.diffusion_model
        if self.latest_network is not None:
            # always restore (~0.05s)
            self.latest_network.restore(unet)

        params_group = external_code.get_all_units_from(args)
        enabled_units = []
        if len(params_group) == 0:
            # fill a null group
            remote_unit = self.parse_remote_call(p, self.get_default_ui_unit(), 0)
            if remote_unit.enabled:
                params_group.append(remote_unit)

        for idx, unit in enumerate(params_group):
            unit = self.parse_remote_call(p, unit, idx)
            if not unit.enabled:
                continue

            enabled_units.append(unit)
            if len(params_group) != 1:
                prefix = f"ControlNet-{idx}"
            else:
                prefix = "ControlNet"

            p.extra_generation_params.update({
                f"{prefix} Enabled": True,
                f"{prefix} Module": unit.module,
                f"{prefix} Model": unit.model,
                f"{prefix} Weight": unit.weight,
                f"{prefix} Guidance Start": unit.guidance_start,
                f"{prefix} Guidance End": unit.guidance_end,
            })

        if len(params_group) == 0 or len(enabled_units) == 0:
           self.latest_network = None
           return 

        detected_maps = []
        forward_params = []
        hook_lowvram = False
        
        # cache stuff
        if self.latest_model_hash != p.sd_model.sd_model_hash:
            self.clear_control_model_cache()

        # unload unused preproc
        module_list = [unit.module for unit in enabled_units]
        for key in self.unloadable:
            if key not in module_list:
                self.unloadable.get(unit.module, lambda:None)()

        self.latest_model_hash = p.sd_model.sd_model_hash
        for idx, unit in enumerate(enabled_units):
            p_input_image = self.get_remote_call(p, "control_net_input_image", None, idx)
            image = image_dict_from_unit(unit)
            if image is not None:
                while len(image['mask'].shape) < 3:
                    image['mask'] = image['mask'][..., np.newaxis]

            resize_mode = external_code.resize_mode_from_value(unit.resize_mode)
            invert_image = unit.invert_image

            if unit.low_vram:
                hook_lowvram = True
                
            model_net = self.load_control_model(p, unet, unit.model, unit.low_vram)
            model_net.reset()

            is_img2img = img2img_tab_tracker.submit_button == 'img2img_generate'
            is_img2img_batch_tab = is_img2img and img2img_tab_tracker.submit_img2img_tab == 'img2img_batch_tab'
            if is_img2img_batch_tab and getattr(p, "image_control", None) is not None:
                input_image = HWC3(np.asarray(p.image_control))
            elif p_input_image is not None:
                input_image = HWC3(np.asarray(p_input_image))
            elif image is not None:
                # Need to check the image for API compatibility
                if isinstance(image['image'], str):
                    from modules.api.api import decode_base64_to_image
                    input_image = HWC3(np.asarray(decode_base64_to_image(image['image'])))
                else:
                    input_image = HWC3(image['image'])

                # Adding 'mask' check for API compatibility
                if 'mask' in image and not ((image['mask'][:, :, 0]==0).all() or (image['mask'][:, :, 0]==255).all()):
                    print("using mask as input")
                    input_image = HWC3(image['mask'][:, :, 0])
                    invert_image = True
            else:
                # use img2img init_image as default
                input_image = getattr(p, "init_images", [None])[0] 
                if input_image is None:
                    raise ValueError('controlnet is enabled but no input image is given')
                input_image = HWC3(np.asarray(input_image))

            if issubclass(type(p), StableDiffusionProcessingImg2Img) and p.inpaint_full_res == True and p.image_mask is not None:
                input_image = Image.fromarray(input_image)
                mask = p.image_mask.convert('L')
                crop_region = masking.get_crop_region(np.array(mask), p.inpaint_full_res_padding)
                crop_region = masking.expand_crop_region(crop_region, p.width, p.height, mask.width, mask.height)

                input_image = input_image.crop(crop_region)
                input_image = images.resize_image(2, input_image, p.width, p.height)
                input_image = HWC3(np.asarray(input_image))

            if invert_image:
                detected_map = np.zeros_like(input_image, dtype=np.uint8)
                detected_map[np.min(input_image, axis=2) < 127] = 255
                input_image = detected_map

            print(f"Loading preprocessor: {unit.module}")
            preprocessor = self.preprocessor[unit.module]
            h, w, bsz = p.height, p.width, p.batch_size
            if unit.processor_res > 64:
                detected_map, is_image = preprocessor(input_image, res=unit.processor_res, thr_a=unit.threshold_a, thr_b=unit.threshold_b)
            else:
                detected_map, is_image = preprocessor(input_image)
            
            if is_image:
                control, detected_map = self.detectmap_proc(detected_map, unit.module, unit.rgbbgr_mode, resize_mode, h, w)
                detected_maps.append((detected_map, unit.module))
            else:
                control = detected_map  

            forward_param = ControlParams(
                control_model=model_net,
                hint_cond=control,
                guess_mode=unit.guess_mode,
                weight=unit.weight,
                guidance_stopped=False,
                start_guidance_percent=unit.guidance_start,
                stop_guidance_percent=unit.guidance_end,
                advanced_weighting=None,
                is_adapter=isinstance(model_net, PlugableAdapter),
                is_extra_cond=getattr(model_net, "target", "") == "scripts.adapter.StyleAdapter"
            )
            forward_params.append(forward_param)

            del model_net

        self.latest_network = UnetHook(lowvram=hook_lowvram)    
        self.latest_network.hook(unet)
        self.latest_network.notify(forward_params, p.sampler_name in ["DDIM", "PLMS", "UniPC"])
        self.detected_map = detected_maps

        if len(enabled_units) > 0 and shared.opts.data.get("control_net_skip_img2img_processing") and hasattr(p, "init_images"):
            swap_img2img_pipeline(p)
        
        # Retreive PwW arguments
        pww_enabled, color_context, weight_function_scale, color_map_image = pww_args
        if pww_enabled:
            color_map_image = Image.fromarray(color_map_image).resize((p.width, p.height))
            color_context = ast.literal_eval(color_context)
            pww_cross_attention_weight = encode_text_color_inputs(p, color_map_image, color_context)
            pww_cross_attention_weight.update({
                "WEIGHT_FUNCTION": lambda w, sigma, qk: float(weight_function_scale) * w * math.log(sigma + 1.0) * qk.max()})
            self.latest_network.p = p
            self.latest_network.pww_cross_attention_weight.update(pww_cross_attention_weight)
            hijack_CrossAttn(p)
