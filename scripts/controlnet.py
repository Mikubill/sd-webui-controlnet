import gc
import inspect
import os
from collections import OrderedDict
from copy import copy
from typing import Union, Dict, Optional, List
import importlib

import numpy
import torch

import modules.scripts as scripts
from modules import shared, devices, script_callbacks, processing, masking, images
import gradio as gr
import numpy as np

from einops import rearrange
from scripts import global_state, hook, external_code, xyz_grid_support, batch_hijack
importlib.reload(global_state)
importlib.reload(hook)
importlib.reload(external_code)
importlib.reload(xyz_grid_support)
importlib.reload(batch_hijack)
from scripts.cldm import PlugableControlModel
from scripts.processor import *
from scripts.adapter import PlugableAdapter
from scripts.utils import load_state_dict
from scripts.hook import ControlParams, UnetHook
from modules.processing import StableDiffusionProcessingImg2Img, StableDiffusionProcessingTxt2Img
from modules.images import save_image
from PIL import Image
from torchvision.transforms import Resize, InterpolationMode, CenterCrop, Compose

gradio_compat = True
try:
    from distutils.version import LooseVersion
    from importlib_metadata import version
    if LooseVersion(version("gradio")) < LooseVersion("3.10"):
        gradio_compat = False
except ImportError:
    pass

# svgsupports
svgsupport = False
try:
    import io
    import base64
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPM
    svgsupport = True
except ImportError:
    pass

refresh_symbol = '\U0001f504'       # 🔄
switch_values_symbol = '\U000021C5' # ⇅
camera_symbol = '\U0001F4F7'        # 📷
reverse_symbol = '\U000021C4'       # ⇄
tossup_symbol = '\u2934'

webcam_enabled = False
webcam_mirrored = False
global_batch_input_dir = gr.Textbox(
    label='Controlnet input directory',
    placeholder='Leave empty to use input directory',
    **shared.hide_dirs,
    elem_id='controlnet_batch_input_dir')
img2img_batch_input_dir = None
img2img_batch_input_dir_callbacks = []
img2img_batch_output_dir = None
img2img_batch_output_dir_callbacks = []
generate_buttons = {}

class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)

    def get_block_name(self):
        return "button"


def find_closest_lora_model_name(search: str):
    if not search:
        return None
    if search in global_state.cn_models:
        return search
    search = search.lower()
    if search in global_state.cn_models_names:
        return global_state.cn_models_names.get(search)
    applicable = [name for name in global_state.cn_models_names.keys()
                  if search in name.lower()]
    if not applicable:
        return None
    applicable = sorted(applicable, key=lambda name: len(name))
    return global_state.cn_models_names[applicable[0]]


def swap_img2img_pipeline(p: processing.StableDiffusionProcessingImg2Img):
    p.__class__ = processing.StableDiffusionProcessingTxt2Img
    dummy = processing.StableDiffusionProcessingTxt2Img()
    for k,v in dummy.__dict__.items():
        if hasattr(p, k):
            continue
        setattr(p, k, v)


global_state.update_cn_models()


def image_dict_from_any(image) -> Optional[Dict[str, np.ndarray]]:
    if image is None:
        return None

    if isinstance(image, (tuple, list)):
        image = {'image': image[0], 'mask': image[1]}
    elif not isinstance(image, dict):
        image = {'image': image, 'mask': None}
    else:  # type(image) is dict
        # copy to enable modifying the dict and prevent response serialization error
        image = dict(image)

    if isinstance(image['image'], str):
        if os.path.exists(image['image']):
            image['image'] = numpy.array(Image.open(image['image'])).astype('uint8')
        else:
            image['image'] = external_code.to_base64_nparray(image['image'])

    if isinstance(image['mask'], str):
        if os.path.exists(image['mask']):
            image['mask'] = numpy.array(Image.open(image['mask'])).astype('uint8')
        else:
            image['mask'] = external_code.to_base64_nparray(image['mask'])
    elif image['mask'] is None:
        image['mask'] = np.zeros_like(image['image'], dtype=np.uint8)

    return image


class UiControlNetUnit(external_code.ControlNetUnit):
    def __init__(
        self,
        input_mode: batch_hijack.InputMode = batch_hijack.InputMode.SIMPLE,
        batch_images: Optional[Union[str, List[external_code.InputImage]]] = None,
        output_dir: str = '',
        loopback: bool = False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.is_ui = True
        self.input_mode = input_mode
        self.batch_images = batch_images
        self.output_dir = output_dir
        self.loopback = loopback


class Script(scripts.Script):
    model_cache = OrderedDict()

    def __init__(self) -> None:
        super().__init__()
        self.latest_network = None
        self.preprocessor_keys = global_state.module_names
        self.preprocessor = global_state.cn_preprocessor_modules
        self.unloadable = global_state.cn_preprocessor_unloadable
        self.input_image = None
        self.latest_model_hash = ""
        self.txt2img_w_slider = gr.Slider()
        self.txt2img_h_slider = gr.Slider()
        self.img2img_w_slider = gr.Slider()
        self.img2img_h_slider = gr.Slider()
        self.enabled_units = []
        self.detected_map = []
        batch_hijack.instance.process_batch_callbacks.append(self.batch_tab_process)
        batch_hijack.instance.process_batch_each_callbacks.append(self.batch_tab_process_each)
        batch_hijack.instance.postprocess_batch_each_callbacks.insert(0, self.batch_tab_postprocess_each)
        batch_hijack.instance.postprocess_batch_callbacks.insert(0, self.batch_tab_postprocess)

    def title(self):
        return "ControlNet"

    def show(self, is_img2img):
        # if is_img2img:
            # return False
        return scripts.AlwaysVisible

    def after_component(self, component, **kwargs):
        if component.elem_id == "txt2img_width":
            self.txt2img_w_slider = component
            return self.txt2img_w_slider
        if component.elem_id == "txt2img_height":
            self.txt2img_h_slider = component
            return self.txt2img_h_slider
        if component.elem_id == "img2img_width":
            self.img2img_w_slider = component
            return self.img2img_w_slider
        if component.elem_id == "img2img_height":
            self.img2img_h_slider = component
            return self.img2img_h_slider
        
    def get_module_basename(self, module):
        for k, v in self.preprocessor_keys.items():
            if v == module or k == module:
                return k
        
    def get_threshold_block(self, proc):
        pass

    def get_default_ui_unit(self, is_ui=True):
        cls = UiControlNetUnit if is_ui else external_code.ControlNetUnit
        return cls(
            enabled=False,
            module="none",
            model="None",
            guess_mode=False,
        )

    def uigroup(self, tabname, is_img2img, elem_id_tabname):
        infotext_fields = []
        default_unit = self.get_default_ui_unit()
        with gr.Row():
            with gr.Tabs():
                with gr.Tab(label='Upload') as upload_tab:
                    upload_image = gr.Image(source='upload', mirror_webcam=False, type='numpy', tool='sketch', elem_id=f'{elem_id_tabname}_{tabname}_input_image')
                    generated_image = gr.Image(label="Annotator result", visible=False, elem_id=f'{elem_id_tabname}_{tabname}_generated_image')

                with gr.Tab(label='Batch') as batch_tab:
                    batch_image_dir = gr.Textbox(label='Input directory', placeholder='Leave empty to use img2img batch controlnet input directory', elem_id=f'{elem_id_tabname}_{tabname}_batch_image_dir')

        with gr.Row():
            gr.HTML(value='<p>Invert colors if your image has white background.<br >Change your brush width to make it thinner if you want to draw something.<br ></p>')
            webcam_enable = ToolButton(value=camera_symbol)
            webcam_mirror = ToolButton(value=reverse_symbol)
            send_dimen_button = ToolButton(value=tossup_symbol)

        with gr.Row():
            enabled = gr.Checkbox(label='Enable', value=default_unit.enabled)
            scribble_mode = gr.Checkbox(label='Invert Input Color', value=default_unit.invert_image)
            rgbbgr_mode = gr.Checkbox(label='RGB to BGR', value=default_unit.rgbbgr_mode)
            lowvram = gr.Checkbox(label='Low VRAM', value=default_unit.low_vram)
            guess_mode = gr.Checkbox(label='Guess Mode', value=default_unit.guess_mode)
            loopback = gr.Checkbox(label='Loopback', value=default_unit.loopback)

        # infotext_fields.append((enabled, "ControlNet Enabled"))
        
        def send_dimensions(image):
            def closesteight(num):
                rem = num % 8
                if rem <= 4:
                    return round(num - rem)
                else:
                    return round(num + (8 - rem))
            if(image):
                interm = np.asarray(image.get('image'))
                return closesteight(interm.shape[1]), closesteight(interm.shape[0])
            else:
                return gr.Slider.update(), gr.Slider.update()
                        
        def webcam_toggle():
            global webcam_enabled
            webcam_enabled = not webcam_enabled
            return {"value": None, "source": "webcam" if webcam_enabled else "upload", "__type__": "update"}
                
        def webcam_mirror_toggle():
            global webcam_mirrored
            webcam_mirrored = not webcam_mirrored
            return {"mirror_webcam": webcam_mirrored, "__type__": "update"}
            
        webcam_enable.click(fn=webcam_toggle, inputs=None, outputs=upload_image)
        webcam_mirror.click(fn=webcam_mirror_toggle, inputs=None, outputs=upload_image)

        def refresh_all_models(*inputs):
            global_state.update_cn_models()
                
            dd = inputs[0]
            selected = dd if dd in global_state.cn_models else "None"
            return gr.Dropdown.update(value=selected, choices=list(global_state.cn_models.keys()))

        with gr.Row():
            module = gr.Dropdown(list(self.preprocessor_keys.values()), label=f"Preprocessor", value=default_unit.module)
            model = gr.Dropdown(list(global_state.cn_models.keys()), label=f"Model", value=default_unit.model)
            refresh_models = ToolButton(value=refresh_symbol)
            refresh_models.click(refresh_all_models, model, model)
                # ctrls += (refresh_models, )
        with gr.Row():
            weight = gr.Slider(label=f"Weight", value=default_unit.weight, minimum=0.0, maximum=2.0, step=.05)
            guidance_start = gr.Slider(label="Guidance Start (T)", value=default_unit.guidance_start, minimum=0.0, maximum=1.0, interactive=True)
            guidance_end = gr.Slider(label="Guidance End (T)", value=default_unit.guidance_end, minimum=0.0, maximum=1.0, interactive=True)

                # model_dropdowns.append(model)
        def build_sliders(module):
            module = self.get_module_basename(module)
            if module == "canny":
                return [
                    gr.update(label="Annotator resolution", value=512, minimum=64, maximum=2048, step=1, interactive=True),
                    gr.update(label="Canny low threshold", minimum=1, maximum=255, value=100, step=1, interactive=True),
                    gr.update(label="Canny high threshold", minimum=1, maximum=255, value=200, step=1, interactive=True),
                    gr.update(visible=True)
                ]
            elif module == "mlsd": #Hough
                return [
                    gr.update(label="Hough Resolution", minimum=64, maximum=2048, value=512, step=1, interactive=True),
                    gr.update(label="Hough value threshold (MLSD)", minimum=0.01, maximum=2.0, value=0.1, step=0.01, interactive=True),
                    gr.update(label="Hough distance threshold (MLSD)", minimum=0.01, maximum=20.0, value=0.1, step=0.01, interactive=True),
                    gr.update(visible=True)
                ]
            elif module in ["hed", "scribble_hed", "hed_safe"]:
                return [
                    gr.update(label="HED Resolution", minimum=64, maximum=2048, value=512, step=1, interactive=True),
                    gr.update(label="Threshold A", value=64, minimum=64, maximum=1024, interactive=False),
                    gr.update(label="Threshold B", value=64, minimum=64, maximum=1024, interactive=False),
                    gr.update(visible=True)
                ]
            elif module in ["openpose", "openpose_full", "segmentation"]:
                return [
                    gr.update(label="Annotator Resolution", minimum=64, maximum=2048, value=512, step=1, interactive=True),
                    gr.update(label="Threshold A", value=64, minimum=64, maximum=1024, interactive=False),
                    gr.update(label="Threshold B", value=64, minimum=64, maximum=1024, interactive=False),
                    gr.update(visible=True)
                ]
            elif module == "depth":
                return [
                    gr.update(label="Midas Resolution", minimum=64, maximum=2048, value=512, step=1, interactive=True),
                    gr.update(label="Threshold A", value=64, minimum=64, maximum=1024, interactive=False),
                    gr.update(label="Threshold B", value=64, minimum=64, maximum=1024, interactive=False),
                    gr.update(visible=True)
                ]
            elif module in ["depth_leres", "depth_leres_boost"]:
                return [
                    gr.update(label="LeReS Resolution", minimum=64, maximum=2048, value=512, step=1, interactive=True),
                    gr.update(label="Remove Near %", value=0, minimum=0, maximum=100, step=0.1, interactive=True),
                    gr.update(label="Remove Background %", value=0, minimum=0, maximum=100, step=0.1, interactive=True),
                    gr.update(visible=True)
                ]
            elif module == "normal_map":
                return [
                    gr.update(label="Normal Resolution", minimum=64, maximum=2048, value=512, step=1, interactive=True),
                    gr.update(label="Normal background threshold", minimum=0.0, maximum=1.0, value=0.4, step=0.01, interactive=True),
                    gr.update(label="Threshold B", value=64, minimum=64, maximum=1024, interactive=False),
                    gr.update(visible=True)
                ]
            elif module == "binary":
                return [
                    gr.update(label="Annotator resolution", value=512, minimum=64, maximum=2048, step=1, interactive=True),
                    gr.update(label="Binary threshold", minimum=0, maximum=255, value=0, step=1, interactive=True),
                    gr.update(label="Threshold B", value=64, minimum=64, maximum=1024, interactive=False),
                    gr.update(visible=True)
                ]
            elif module == "color":
                return [
                    gr.update(label="Annotator Resolution", value=512, minimum=64, maximum=2048, step=8, interactive=True),
                    gr.update(label="Threshold A", value=64, minimum=64, maximum=1024, interactive=False),
                    gr.update(label="Threshold B", value=64, minimum=64, maximum=1024, interactive=False),
                    gr.update(visible=True)
                ]
            elif module == "none":
                return [
                    gr.update(label="Normal Resolution", value=64, minimum=64, maximum=2048, interactive=False),
                    gr.update(label="Threshold A", value=64, minimum=64, maximum=1024, interactive=False),
                    gr.update(label="Threshold B", value=64, minimum=64, maximum=1024, interactive=False),
                    gr.update(visible=False)
                ]
            else:
                return [
                    gr.update(label="Annotator resolution", value=512, minimum=64, maximum=2048, step=1, interactive=True),
                    gr.update(label="Threshold A", value=64, minimum=64, maximum=1024, interactive=False),
                    gr.update(label="Threshold B", value=64, minimum=64, maximum=1024, interactive=False),
                    gr.update(visible=True)
                ]
                
        # advanced options    
        advanced = gr.Column(visible=False)
        with advanced:
            processor_res = gr.Slider(label="Annotator resolution", value=default_unit.processor_res, minimum=64, maximum=2048, interactive=False)
            threshold_a =  gr.Slider(label="Threshold A", value=default_unit.threshold_a, minimum=64, maximum=1024, interactive=False)
            threshold_b =  gr.Slider(label="Threshold B", value=default_unit.threshold_b, minimum=64, maximum=1024, interactive=False)
            
        if gradio_compat:    
            module.change(build_sliders, inputs=[module], outputs=[processor_res, threshold_a, threshold_b, advanced])
                
        # infotext_fields.extend((module, model, weight))

        def create_canvas(h, w):
            return np.zeros(shape=(h, w, 3), dtype=np.uint8) + 255
            
        def svgPreprocess(inputs):
            if (inputs):
                if (inputs['image'].startswith("data:image/svg+xml;base64,") and svgsupport):
                    svg_data = base64.b64decode(inputs['image'].replace('data:image/svg+xml;base64,',''))
                    drawing = svg2rlg(io.BytesIO(svg_data))
                    png_data = renderPM.drawToString(drawing, fmt='PNG')
                    encoded_string = base64.b64encode(png_data)
                    base64_str = str(encoded_string, "utf-8")
                    base64_str = "data:image/png;base64,"+ base64_str
                    inputs['image'] = base64_str
                return upload_image.orgpreprocess(inputs)
            return None

        resize_mode = gr.Radio(choices=[e.value for e in external_code.ResizeMode], value=default_unit.resize_mode.value, label="Resize Mode")
        with gr.Row():
            with gr.Column():
                canvas_width = gr.Slider(label="Canvas Width", minimum=256, maximum=1024, value=512, step=64)
                canvas_height = gr.Slider(label="Canvas Height", minimum=256, maximum=1024, value=512, step=64)
                    
            if gradio_compat:
                canvas_swap_res = ToolButton(value=switch_values_symbol)
                canvas_swap_res.click(lambda w, h: (h, w), inputs=[canvas_width, canvas_height], outputs=[canvas_width, canvas_height])
                    
        create_button = gr.Button(value="Create blank canvas")
        create_button.click(fn=create_canvas, inputs=[canvas_height, canvas_width], outputs=[upload_image])
        
        def run_annotator(image, module, pres, pthr_a, pthr_b):
            img = HWC3(image['image'])
            if not ((image['mask'][:, :, 0]==0).all() or (image['mask'][:, :, 0]==255).all()):
                img = HWC3(image['mask'][:, :, 0])
                
            module = self.get_module_basename(module)
            preprocessor = self.preprocessor[module]
            result = None
            if pres > 64:
                result, is_image = preprocessor(img, res=pres, thr_a=pthr_a, thr_b=pthr_b)
            else:
                result, is_image = preprocessor(img)
            
            if is_image:
                return gr.update(value=result, visible=True, interactive=False)
        
        with gr.Row():
            annotator_button = gr.Button(value="Preview annotator result")
            annotator_button_hide = gr.Button(value="Hide annotator result")
        
        annotator_button.click(fn=run_annotator, inputs=[upload_image, module, processor_res, threshold_a, threshold_b], outputs=[generated_image])
        annotator_button_hide.click(fn=lambda: gr.update(visible=False), inputs=None, outputs=[generated_image])

        if is_img2img:
            send_dimen_button.click(fn=send_dimensions, inputs=[upload_image], outputs=[self.img2img_w_slider, self.img2img_h_slider])
        else:
            send_dimen_button.click(fn=send_dimensions, inputs=[upload_image], outputs=[self.txt2img_w_slider, self.txt2img_h_slider])

        input_mode = gr.State(batch_hijack.InputMode.SIMPLE)
        input_image = gr.State()
        batch_image_dir_state = gr.State('')
        output_dir_state = gr.State('')
        unit_args = (input_mode, batch_image_dir_state, output_dir_state, loopback, enabled, module, model, weight, input_image, scribble_mode, resize_mode, rgbbgr_mode, lowvram, processor_res, threshold_a, threshold_b, guidance_start, guidance_end, guess_mode)
        self.register_modules(tabname, unit_args)

        upload_image.orgpreprocess=upload_image.preprocess
        upload_image.preprocess=svgPreprocess

        # update unit when any of its properties changes
        unit = gr.State(default_unit)
        for comp in unit_args:
            for events in (
                ('edit', 'clear'),
                ('click',),
                ('change',),
            ):
                if not all(hasattr(comp, event) for event in events): continue
                for event in events:
                    getattr(comp, event)(fn=UiControlNetUnit, inputs=list(unit_args), outputs=unit)

                break

        def index_of_init_parameter(parameter):
            parameters = inspect.getfullargspec(UiControlNetUnit.__init__)[0][1:] + inspect.getfullargspec(external_code.ControlNetUnit.__init__)[0][1:]
            index = parameters.index(parameter)
            return index

        # keep upload_image in sync with input_image
        upload_image_index = index_of_init_parameter('image')
        components = list(unit_args)
        components[upload_image_index] = upload_image
        for event in 'edit', 'clear':
            getattr(upload_image, event)(fn=lambda *args: (args[upload_image_index], UiControlNetUnit(*args)), inputs=components, outputs=[input_image, unit])

        # keep input_mode in sync
        input_mode_index = index_of_init_parameter('input_mode')
        components = list(unit_args)
        for input_tab in (
            (upload_tab, batch_hijack.InputMode.SIMPLE),
            (batch_tab, batch_hijack.InputMode.BATCH)
        ):
            components[input_mode_index] = gr.State(input_tab[1])
            input_tab[0].select(fn=lambda *args: (args[input_mode_index], UiControlNetUnit(*args)), inputs=components, outputs=[input_mode, unit])

        batch_dir_index = index_of_init_parameter('batch_images')
        def determine_batch_dir(batch_dir, fallback_dir, fallback_fallback_dir, *args):
            args = list(args)
            if batch_dir:
                args[batch_dir_index] = batch_dir
            elif fallback_dir:
                args[batch_dir_index] = fallback_dir
            else:
                args[batch_dir_index] = fallback_fallback_dir

            return args[batch_dir_index], UiControlNetUnit(*args)

        # keep batch_dir in sync with global batch input textboxes
        global img2img_batch_input_dir, img2img_batch_input_dir_callbacks
        def subscribe_for_batch_dir():
            global global_batch_input_dir, img2img_batch_input_dir
            batch_dirs = [batch_image_dir, global_batch_input_dir, img2img_batch_input_dir]
            for batch_dir_comp in batch_dirs:
                if not hasattr(batch_dir_comp, 'change'): continue
                batch_dir_comp.change(
                    fn=determine_batch_dir,
                    inputs=batch_dirs + list(unit_args),
                    outputs=[batch_image_dir_state, unit])

        if img2img_batch_input_dir is None:
            # we are too soon, subscribe later when available
            img2img_batch_input_dir_callbacks.append(subscribe_for_batch_dir)
        else:
            subscribe_for_batch_dir()

        # keep output_dir in sync with global batch output textbox
        global img2img_batch_output_dir, img2img_batch_output_dir_callbacks
        def subscribe_for_output_dir():
            output_dir_index = index_of_init_parameter('output_dir')
            def update_output_dir(output_dir, *args):
                args = list(args)
                args[output_dir_index] = output_dir
                return output_dir, UiControlNetUnit(*args)

            global img2img_batch_output_dir
            img2img_batch_output_dir.change(
                fn=update_output_dir,
                inputs=[img2img_batch_output_dir] + list(unit_args),
                outputs=[output_dir_state, unit]
            )

        if img2img_batch_input_dir is None:
            # we are too soon, subscribe later when available
            img2img_batch_output_dir_callbacks.append(subscribe_for_output_dir)
        else:
            subscribe_for_output_dir()

        return unit

    def ui(self, is_img2img):
        """this function should create gradio UI elements. See https://gradio.app/docs/#components
        The return value should be an array of all components that are used in processing.
        Values of those returned components will be passed to run() and process() functions.
        """
        self.infotext_fields = []
        self.paste_field_names = []
        controls = ()
        max_models = shared.opts.data.get("control_net_max_models_num", 1)
        elem_id_tabname = ("img2img" if is_img2img else "txt2img") + "_controlnet"
        with gr.Group(elem_id=elem_id_tabname):
            with gr.Accordion("ControlNet", open = False, elem_id="controlnet"):
                if max_models > 1:
                    with gr.Tabs(elem_id=f"{elem_id_tabname}_tabs"):
                        for i in range(max_models):
                            with gr.Tab(f"Control Model - {i}"):
                                controls += (self.uigroup(f"ControlNet-{i}", is_img2img, elem_id_tabname),)
                else:
                    with gr.Column():
                        controls += (self.uigroup(f"ControlNet", is_img2img, elem_id_tabname),)

        if shared.opts.data.get("control_net_sync_field_args", False):
            for _, field_name in self.infotext_fields:
                self.paste_field_names.append(field_name)

        return controls

    def register_modules(self, tabname, params):
        enabled, module, model, weight = params[:4]
        guidance_start, guidance_end, guess_mode = params[-3:]
        
        self.infotext_fields.extend([
            (enabled, f"{tabname} Enabled"),
            (module, f"{tabname} Preprocessor"),
            (model, f"{tabname} Model"),
            (weight, f"{tabname} Weight"),
            (guidance_start, f"{tabname} Guidance Start"),
            (guidance_end, f"{tabname} Guidance End"),
        ])

    def clear_control_model_cache(self):
        Script.model_cache.clear()
        gc.collect()
        devices.torch_gc()

    def load_control_model(self, p, unet, model, lowvram):
        if model in Script.model_cache:
            print(f"Loading model from cache: {model}")
            return Script.model_cache[model]

        # Remove model from cache to clear space before building another model
        if len(Script.model_cache) > 0 and len(Script.model_cache) >= shared.opts.data.get("control_net_model_cache_size", 2):
            Script.model_cache.popitem(last=False)
            gc.collect()
            devices.torch_gc()

        model_net = self.build_control_model(p, unet, model, lowvram)

        if shared.opts.data.get("control_net_model_cache_size", 2) > 0:
            Script.model_cache[model] = model_net

        return model_net

    def build_control_model(self, p, unet, model, lowvram):
        model_path = global_state.cn_models.get(model, None)
        if model_path is None:
            model = find_closest_lora_model_name(model)
            model_path = global_state.cn_models.get(model, None)

        if model_path is None:
            raise RuntimeError(f"model not found: {model}")

        # trim '"' at start/end
        if model_path.startswith("\"") and model_path.endswith("\""):
            model_path = model_path[1:-1]

        if not os.path.exists(model_path):
            raise ValueError(f"file not found: {model_path}")

        print(f"Loading model: {model}")
        state_dict = load_state_dict(model_path)
        network_module = PlugableControlModel
        network_config = shared.opts.data.get("control_net_model_config", global_state.default_conf)
        if not os.path.isabs(network_config):
            network_config = os.path.join(global_state.script_dir, network_config)

        if any([k.startswith("body.") or k == 'style_embedding' for k, v in state_dict.items()]):
            # adapter model
            network_module = PlugableAdapter
            network_config = shared.opts.data.get("control_net_model_adapter_config", global_state.default_conf_adapter)
            if not os.path.isabs(network_config):
                network_config = os.path.join(global_state.script_dir, network_config)

        override_config = os.path.splitext(model_path)[0] + ".yaml"
        if os.path.exists(override_config):
            network_config = override_config

        network = network_module(
            state_dict=state_dict,
            config_path=network_config,
            lowvram=lowvram,
            base_model=unet,
        )
        network.to(p.sd_model.device, dtype=p.sd_model.dtype)
        print(f"ControlNet model {model} loaded.")
        return network

    @staticmethod
    def get_remote_call(p, attribute, default=None, idx=0, strict=False, force=False):
        if not force and not shared.opts.data.get("control_net_allow_script_control", False):
            return default

        def get_element(obj, strict=False):
            if not isinstance(obj, list):
                return obj if not strict or idx == 0 else None
            elif idx < len(obj):
                return obj[idx]
            else:
                return None

        attribute_value = get_element(getattr(p, attribute, None), strict)
        default_value = get_element(default)
        return attribute_value if attribute_value is not None else default_value

    def parse_remote_call(self, p, unit: external_code.ControlNetUnit, idx):
        selector = self.get_remote_call

        unit.enabled = selector(p, "control_net_enabled", unit.enabled, idx, strict=True)
        unit.module = selector(p, "control_net_module", unit.module, idx)
        unit.model = selector(p, "control_net_model", unit.model, idx)
        unit.weight = selector(p, "control_net_weight", unit.weight, idx)
        unit.image = selector(p, "control_net_image", unit.image, idx)
        unit.scribble_mode = selector(p, "control_net_scribble_mode", unit.invert_image, idx)
        unit.resize_mode = selector(p, "control_net_resize_mode", unit.resize_mode, idx)
        unit.rgbbgr_mode = selector(p, "control_net_rgbbgr_mode", unit.rgbbgr_mode, idx)
        unit.low_vram = selector(p, "control_net_lowvram", unit.low_vram, idx)
        unit.processor_res = selector(p, "control_net_pres", unit.processor_res, idx)
        unit.threshold_a = selector(p, "control_net_pthr_a", unit.threshold_a, idx)
        unit.threshold_b = selector(p, "control_net_pthr_b", unit.threshold_b, idx)
        unit.guidance_start = selector(p, "control_net_guidance_start", unit.guidance_start, idx)
        unit.guidance_end = selector(p, "control_net_guidance_end", unit.guidance_end, idx)
        unit.guidance_end = selector(p, "control_net_guidance_strength", unit.guidance_end, idx)
        unit.guess_mode = selector(p, "control_net_guess_mode", unit.guess_mode, idx)

        return unit

    def detectmap_proc(self, detected_map, module, rgbbgr_mode, resize_mode, h, w):
        detected_map = HWC3(detected_map)
        if module == "normal_map" or rgbbgr_mode:
            control = torch.from_numpy(detected_map[:, :, ::-1].copy()).float().to(devices.get_device_for("controlnet")) / 255.0
        else:
            control = torch.from_numpy(detected_map.copy()).float().to(devices.get_device_for("controlnet")) / 255.0

        control = rearrange(control, 'h w c -> c h w')
        detected_map = rearrange(torch.from_numpy(detected_map), 'h w c -> c h w')

        if resize_mode == external_code.ResizeMode.INNER_FIT:
            h0 = detected_map.shape[1]
            w0 = detected_map.shape[2]
            w1 = w0
            h1 = int(w0/w*h)
            if (h/w > h0/w0):
                h1 = h0
                w1 = int(h0/h*w)
            transform = Compose([
                CenterCrop(size=(h1, w1)),
                Resize(size=(h, w), interpolation=InterpolationMode.BICUBIC)
            ])
            control = transform(control)
            detected_map = transform(detected_map)
        elif resize_mode == external_code.ResizeMode.OUTER_FIT:
            h0 = detected_map.shape[1]
            w0 = detected_map.shape[2]
            h1 = h0
            w1 = int(h0/h*w)
            if (h/w > h0/w0):
                w1 = w0
                h1 = int(w0/w*h)
            transform = Compose([
                CenterCrop(size=(h1, w1)),
                Resize(size=(h, w),interpolation=InterpolationMode.BICUBIC)
            ])
            control = transform(control)
            detected_map = transform(detected_map)
        else:
            control = Resize((h,w), interpolation=InterpolationMode.BICUBIC)(control)
            detected_map = Resize((h,w), interpolation=InterpolationMode.BICUBIC)(detected_map)

        # for log use
        detected_map = rearrange(detected_map, 'c h w -> h w c').numpy().astype(np.uint8)
        return control, detected_map

    def is_ui(self, args):
        return args and all(isinstance(arg, UiControlNetUnit) for arg in args)

    def get_enabled_units(self, p):
        units = external_code.get_all_units_in_processing(p)
        enabled_units = []

        if len(units) == 0:
            # fill a null group
            remote_unit = self.parse_remote_call(p, self.get_default_ui_unit(), 0)
            if remote_unit.enabled:
                units.append(remote_unit)

        for idx, unit in enumerate(units):
            unit = self.parse_remote_call(p, unit, idx)
            if not unit.enabled:
                continue

            enabled_units.append(copy(unit))
            if len(units) != 1:
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

        return enabled_units

    def process(self, p, *args):
        """
        This function is called before processing begins for AlwaysVisible scripts.
        You can modify the processing object (p) here, inject hooks, etc.
        args contains all values returned by components from ui()
        """
        unet = p.sd_model.model.diffusion_model
        if self.latest_network is not None:
            # always restore (~0.05s)
            self.latest_network.restore(unet)

        if not batch_hijack.instance.is_batch:
            self.enabled_units = self.get_enabled_units(p)

        if len(self.enabled_units) == 0:
           self.latest_network = None
           return

        detected_maps = []
        forward_params = []
        hook_lowvram = False

        # cache stuff
        if self.latest_model_hash != p.sd_model.sd_model_hash:
            self.clear_control_model_cache()

        # unload unused preproc
        module_list = [unit.module for unit in self.enabled_units]
        for key in self.unloadable:
            if key not in module_list:
                self.unloadable.get(key, lambda:None)()

        self.latest_model_hash = p.sd_model.sd_model_hash
        for idx, unit in enumerate(self.enabled_units):
            unit.module = self.get_module_basename(unit.module)
            p_input_image = self.get_remote_call(p, "control_net_input_image", None, idx)
            image = image_dict_from_any(unit.image)
            if image is not None:
                while len(image['mask'].shape) < 3:
                    image['mask'] = image['mask'][..., np.newaxis]

            resize_mode = external_code.resize_mode_from_value(unit.resize_mode)
            invert_image = unit.invert_image

            if unit.low_vram:
                hook_lowvram = True

            model_net = self.load_control_model(p, unet, unit.model, unit.low_vram)
            model_net.reset()

            if batch_hijack.instance.is_batch and getattr(p, "image_control", None) is not None:
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

                # scale crop region to the size of our image
                x1, y1, x2, y2 = crop_region
                scale_x, scale_y = mask.width / float(input_image.width), mask.height / float(input_image.height)
                crop_region = int(x1 / scale_x), int(y1 / scale_y), int(x2 / scale_x), int(y2 / scale_y)

                input_image = input_image.crop(crop_region)
                input_image = images.resize_image(2, input_image, p.width, p.height)
                input_image = HWC3(np.asarray(input_image))

            np.random.seed(int(p.seed) % 65535)

            print(f"Loading preprocessor: {unit.module}")
            preprocessor = self.preprocessor[unit.module]
            h, w, bsz = p.height, p.width, p.batch_size
            if unit.processor_res > 64:
                detected_map, is_image = preprocessor(input_image, res=unit.processor_res, thr_a=unit.threshold_a, thr_b=unit.threshold_b)
            else:
                detected_map, is_image = preprocessor(input_image)
                
            if invert_image:
                detected_map = 255 - detected_map.copy() 

            if unit.module == "none" and "style" in unit.model:
                detected_map_bytes = detected_map[:,:,0].tobytes()
                detected_map = np.ndarray((round(input_image.shape[0]/4),input_image.shape[1]),dtype="float32",buffer=detected_map_bytes)
                detected_map = torch.Tensor(detected_map).to(devices.get_device_for("controlnet"))
                is_image = False

            if isinstance(p, StableDiffusionProcessingTxt2Img) and p.enable_hr:
                if p.hr_resize_x == 0 and p.hr_resize_y == 0:
                    hr_y = int(p.height * p.hr_scale)
                    hr_x = int(p.width * p.hr_scale)
                else:
                    hr_y, hr_x = p.hr_resize_y, p.hr_resize_x

                if is_image:
                    hr_control, _ = self.detectmap_proc(detected_map, unit.module, unit.rgbbgr_mode, resize_mode, hr_y, hr_x)
                else:
                    hr_control = detected_map
            else:
                hr_control = None

            if is_image:
                control, detected_map = self.detectmap_proc(detected_map, unit.module, unit.rgbbgr_mode, resize_mode, h, w)
                detected_maps.append((detected_map, unit.module))
            else:
                control = detected_map

                if unit.module == 'clip_vision':
                    fake_detected_map = np.ndarray((detected_map.shape[0]*4, detected_map.shape[1]),dtype="uint8",buffer=detected_map.numpy(force=True).tobytes())
                    detected_maps.append((fake_detected_map, unit.module))


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
                is_extra_cond=getattr(model_net, "target", "") == "scripts.adapter.StyleAdapter",
                global_average_pooling=model_net.config.model.params.get("global_average_pooling", False),
                hr_hint_cond=hr_control
            )
            forward_params.append(forward_param)

            del model_net

        self.latest_network = UnetHook(lowvram=hook_lowvram)
        self.latest_network.hook(unet)
        self.latest_network.notify(forward_params, p.sampler_name in ["DDIM", "PLMS", "UniPC"])
        self.detected_map = detected_maps

        if len(self.enabled_units) > 0 and shared.opts.data.get("control_net_skip_img2img_processing") and hasattr(p, "init_images"):
            swap_img2img_pipeline(p)

    def postprocess(self, p, processed, *args):
        if not batch_hijack.instance.is_batch:
            self.enabled_units.clear()

        if shared.opts.data.get("control_net_detectmap_autosaving", False) and self.latest_network is not None:
            for detect_map, module in self.detected_map:
                detectmap_dir = os.path.join(shared.opts.data.get("control_net_detectedmap_dir", ""), module)
                if not os.path.isabs(detectmap_dir):
                    detectmap_dir = os.path.join(p.outpath_samples, detectmap_dir)
                if module != "none":
                    os.makedirs(detectmap_dir, exist_ok=True)
                    img = Image.fromarray(detect_map)
                    save_image(img, detectmap_dir, module)

        if self.latest_network is None:
            return

        no_detectmap_opt = shared.opts.data.get("control_net_no_detectmap", False)
        if not batch_hijack.instance.is_batch or not no_detectmap_opt and self.detected_map:
            for detect_map, module in self.detected_map:
                if detect_map is None:
                    continue
                processed.images.extend([Image.fromarray(detect_map)])

        self.input_image = None
        self.latest_network.restore(p.sd_model.model.diffusion_model)
        self.latest_network = None
        self.detected_map.clear()

        gc.collect()
        devices.torch_gc()

    def batch_tab_process(self, p, batches, *args, **kwargs):
        self.enabled_units = self.get_enabled_units(p)
        for unit_i, unit in enumerate(self.enabled_units):
            unit.batch_images = iter([batch[unit_i] for batch in batches])

    def batch_tab_process_each(self, p, *args, **kwargs):
        for unit_i, unit in enumerate(self.enabled_units):
            if getattr(unit, 'loopback', False) and batch_hijack.instance.batch_index > 0: continue

            unit.image = next(unit.batch_images)

    def batch_tab_postprocess_each(self, p, processed, *args, **kwargs):
        for unit_i, unit in enumerate(self.enabled_units):
            if getattr(unit, 'loopback', False):
                output_images = getattr(processed, 'images', [])[processed.index_of_first_image:]
                if output_images:
                    unit.image = np.array(output_images[0])
                else:
                    print(f'Warning: No loopback image found for ControlNet model {unit_i}. Using control map from last batch iteration instead')

    def batch_tab_postprocess(self, p, *args, **kwargs):
        self.enabled_units.clear()
        self.input_image = None
        if self.latest_network is None: return

        self.latest_network.restore(shared.sd_model.model.diffusion_model)
        self.latest_network = None
        self.detected_map.clear()


def update_script_args(p, value, arg_idx):
    for s in scripts.scripts_txt2img.alwayson_scripts:
        if isinstance(s, Script):
            args = list(p.script_args)
            # print(f"Changed arg {arg_idx} from {args[s.args_from + arg_idx - 1]} to {value}")
            args[s.args_from + arg_idx] = value
            p.script_args = tuple(args)
            break


def on_ui_settings():
    section = ('control_net', "ControlNet")
    shared.opts.add_option("control_net_model_config", shared.OptionInfo(
        global_state.default_conf, "Config file for Control Net models", section=section))
    shared.opts.add_option("control_net_model_adapter_config", shared.OptionInfo(
        global_state.default_conf_adapter, "Config file for Adapter models", section=section))
    shared.opts.add_option("control_net_detectedmap_dir", shared.OptionInfo(
        global_state.default_detectedmap_dir, "Directory for detected maps auto saving", section=section))
    shared.opts.add_option("control_net_models_path", shared.OptionInfo(
        "", "Extra path to scan for ControlNet models (e.g. training output directory)", section=section))
    shared.opts.add_option("control_net_max_models_num", shared.OptionInfo(
        1, "Multi ControlNet: Max models amount (requires restart)", gr.Slider, {"minimum": 1, "maximum": 10, "step": 1}, section=section))
    shared.opts.add_option("control_net_model_cache_size", shared.OptionInfo(
        1, "Model cache size (requires restart)", gr.Slider, {"minimum": 1, "maximum": 5, "step": 1}, section=section))
    shared.opts.add_option("control_net_control_transfer", shared.OptionInfo(
        False, "Apply transfer control when loading models", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("control_net_no_detectmap", shared.OptionInfo(
        False, "Do not append detectmap to output", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("control_net_detectmap_autosaving", shared.OptionInfo(
        False, "Allow detectmap auto saving", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("control_net_allow_script_control", shared.OptionInfo(
        False, "Allow other script to control this extension", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("control_net_skip_img2img_processing", shared.OptionInfo(
        False, "Skip img2img processing when using img2img initial image", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("control_net_monocular_depth_optim", shared.OptionInfo(
        False, "Enable optimized monocular depth estimation", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("control_net_only_mid_control", shared.OptionInfo(
        False, "Only use mid-control when inference", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("control_net_sync_field_args", shared.OptionInfo(
        False, "Passing ControlNet parameters with \"Send to img2img\"", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("controlnet_show_batch_images_in_ui", shared.OptionInfo(
        False, "Show batch images in gradio gallerie output", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("controlnet_increment_seed_during_batch", shared.OptionInfo(
        True, "Increment seed after each controlnet batch iteration", gr.Checkbox, {"interactive": True}, section=section))
    # shared.opts.add_option("control_net_advanced_weighting", shared.OptionInfo(
    #     False, "Enable advanced weight tuning", gr.Checkbox, {"interactive": False}, section=section))


def on_after_component(component, **_kwargs):
    global img2img_batch_input_dir
    if getattr(component, 'elem_id', None) == 'img2img_batch_input_dir':
        img2img_batch_input_dir = component
        for callback in img2img_batch_input_dir_callbacks:
            callback()
        return

    global img2img_batch_output_dir
    if getattr(component, 'elem_id', None) == 'img2img_batch_output_dir':
        img2img_batch_output_dir = component
        for callback in img2img_batch_output_dir_callbacks:
            callback()
        return

    if getattr(component, 'elem_id', None) == 'img2img_batch_inpaint_mask_dir':
        global_batch_input_dir.render()
        return


batch_hijack.instance.do_hijack()
script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_after_component(on_after_component)
