import gc
import os
import stat
from collections import OrderedDict
from enum import Enum
from typing import Union

import torch

import modules.scripts as scripts
from modules import shared, devices, script_callbacks, processing, masking, images
import gradio as gr
import numpy as np

from einops import rearrange
from scripts.cldm import PlugableControlModel
from scripts.processor import *
from scripts.adapter import PlugableAdapter
from scripts.utils import load_state_dict
from scripts.hook import ControlParams, UnetHook
from modules import sd_models
from modules.paths import models_path
from modules.processing import StableDiffusionProcessingImg2Img
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

CN_MODEL_EXTS = [".pt", ".pth", ".ckpt", ".safetensors"]
cn_models = OrderedDict()      # "My_Lora(abcd1234)" -> C:/path/to/model.safetensors
cn_models_names = {}  # "my_lora" -> "My_Lora(abcd1234)"
cn_models_dir = os.path.join(models_path, "ControlNet")
cn_models_dir_old = os.path.join(scripts.basedir(), "models")

default_conf = os.path.join("models", "cldm_v15.yaml")
default_conf_adapter = os.path.join("models", "sketch_adapter_v14.yaml")
cn_detectedmap_dir = os.path.join("detected_maps")
default_detectedmap_dir = cn_detectedmap_dir
script_dir = scripts.basedir()

os.makedirs(cn_models_dir, exist_ok=True)
os.makedirs(cn_detectedmap_dir, exist_ok=True)

refresh_symbol = '\U0001f504'       # 🔄
switch_values_symbol = '\U000021C5' # ⇅
camera_symbol = '\U0001F4F7'        # 📷
reverse_symbol = '\U000021C4'       # ⇄
tossup_symbol = '\u2934'

webcam_enabled = False
webcam_mirrored = False

PARAM_COUNT = 15


class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)

    def get_block_name(self):
        return "button"
    

def traverse_all_files(curr_path, model_list):
    f_list = [(os.path.join(curr_path, entry.name), entry.stat())
              for entry in os.scandir(curr_path)]
    for f_info in f_list:
        fname, fstat = f_info
        if os.path.splitext(fname)[1] in CN_MODEL_EXTS:
            model_list.append(f_info)
        elif stat.S_ISDIR(fstat.st_mode):
            model_list = traverse_all_files(fname, model_list)
    return model_list


def get_all_models(sort_by, filter_by, path):
    res = OrderedDict()
    fileinfos = traverse_all_files(path, [])
    filter_by = filter_by.strip(" ")
    if len(filter_by) != 0:
        fileinfos = [x for x in fileinfos if filter_by.lower()
                     in os.path.basename(x[0]).lower()]
    if sort_by == "name":
        fileinfos = sorted(fileinfos, key=lambda x: os.path.basename(x[0]))
    elif sort_by == "date":
        fileinfos = sorted(fileinfos, key=lambda x: -x[1].st_mtime)
    elif sort_by == "path name":
        fileinfos = sorted(fileinfos)

    for finfo in fileinfos:
        filename = finfo[0]
        name = os.path.splitext(os.path.basename(filename))[0]
        # Prevent a hypothetical "None.pt" from being listed.
        if name != "None":
            res[name + f" [{sd_models.model_hash(filename)}]"] = filename

    return res


def find_closest_lora_model_name(search: str):
    if not search:
        return None
    if search in cn_models:
        return search
    search = search.lower()
    if search in cn_models_names:
        return cn_models_names.get(search)
    applicable = [name for name in cn_models_names.keys()
                  if search in name.lower()]
    if not applicable:
        return None
    applicable = sorted(applicable, key=lambda name: len(name))
    return cn_models_names[applicable[0]]


def swap_img2img_pipeline(p: processing.StableDiffusionProcessingImg2Img):
    p.__class__ = processing.StableDiffusionProcessingTxt2Img
    dummy = processing.StableDiffusionProcessingTxt2Img()
    for k,v in dummy.__dict__.items():
        if hasattr(p, k):
            continue
        setattr(p, k, v)


def update_cn_models():
    cn_models.clear()
    ext_dirs = (shared.opts.data.get("control_net_models_path", None), getattr(shared.cmd_opts, 'controlnet_dir', None))
    extra_lora_paths = (extra_lora_path for extra_lora_path in ext_dirs
                if extra_lora_path is not None and os.path.exists(extra_lora_path))
    paths = [cn_models_dir, cn_models_dir_old, *extra_lora_paths]

    for path in paths:
        sort_by = shared.opts.data.get(
            "control_net_models_sort_models_by", "name")
        filter_by = shared.opts.data.get("control_net_models_name_filter", "")
        found = get_all_models(sort_by, filter_by, path)
        cn_models.update({**found, **cn_models})

    # insert "None" at the beginning of `cn_models` in-place
    cn_models_copy = OrderedDict(cn_models)
    cn_models.clear()
    cn_models.update({**{"None": None}, **cn_models_copy})

    cn_models_names.clear()
    for name_and_hash, filename in cn_models.items():
        if filename is None:
            continue
        name = os.path.splitext(os.path.basename(filename))[0].lower()
        cn_models_names[name] = name_and_hash


update_cn_models()


class ResizeMode(Enum):
    RESIZE = "Just Resize"
    INNER_FIT = "Scale to Fit (Inner Fit)"
    OUTER_FIT = "Envelope (Outer Fit)"


def resize_mode_from_value(value: Union[str, int, ResizeMode]) -> ResizeMode:
    if isinstance(value, str):
        return ResizeMode(value)
    elif isinstance(value, int):
        return [e for e in ResizeMode][value]
    else:
        return value

class Script(scripts.Script):
    model_cache = OrderedDict()

    def __init__(self) -> None:
        super().__init__()
        self.latest_network = None
        self.preprocessor = {
            "none": lambda x, *args, **kwargs: (x, True),
            "canny": canny,
            "depth": midas,
            "depth_leres": leres,
            "hed": hed,
            "mlsd": mlsd,
            "normal_map": midas_normal,
            "openpose": openpose,
            "openpose_hand": openpose_hand,
            "clip_vision": clip,
            "color": color,
            "pidinet": pidinet,
            "scribble": simple_scribble,
            "fake_scribble": fake_scribble,
            "segmentation": uniformer,
            "binary": binary,
        }
        self.unloadable = {
            "hed": unload_hed,
            "fake_scribble": unload_hed,
            "mlsd": unload_mlsd,
            "clip": unload_clip,
            "depth": unload_midas,
            "depth_leres": unload_leres,
            "normal_map": unload_midas,
            "pidinet": unload_pidinet,
            "openpose": unload_openpose,
            "openpose_hand": unload_openpose,
            "segmentation": unload_uniformer,
        }
        self.input_image = None
        self.latest_model_hash = ""
        self.txt2img_w_slider = gr.Slider()
        self.txt2img_h_slider = gr.Slider()
        self.img2img_w_slider = gr.Slider()
        self.img2img_h_slider = gr.Slider()

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
        
    def get_threshold_block(self, proc):
        pass
    
    def uigroup(self, is_img2img):
        ctrls = ()
        infotext_fields = []
        with gr.Row():
            input_image = gr.Image(source='upload', mirror_webcam=False, type='numpy', tool='sketch')
            generated_image = gr.Image(label="Annotator result", visible=False)

        with gr.Row():
            gr.HTML(value='<p>Invert colors if your image has white background.<br >Change your brush width to make it thinner if you want to draw something.<br ></p>')
            webcam_enable = ToolButton(value=camera_symbol)
            webcam_mirror = ToolButton(value=reverse_symbol)
            send_dimen_button = ToolButton(value=tossup_symbol)

        with gr.Row():
            enabled = gr.Checkbox(label='Enable', value=False)
            scribble_mode = gr.Checkbox(label='Invert Input Color', value=False)
            rgbbgr_mode = gr.Checkbox(label='RGB to BGR', value=False)
            lowvram = gr.Checkbox(label='Low VRAM', value=False)
            guess_mode = gr.Checkbox(label='Guess Mode', value=False)

        ctrls += (enabled,)
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
            
        webcam_enable.click(fn=webcam_toggle, inputs=None, outputs=input_image)
        webcam_mirror.click(fn=webcam_mirror_toggle, inputs=None, outputs=input_image)

        def refresh_all_models(*inputs):
            update_cn_models()
                
            dd = inputs[0]
            selected = dd if dd in cn_models else "None"
            return gr.Dropdown.update(value=selected, choices=list(cn_models.keys()))

        with gr.Row():
            module = gr.Dropdown(list(self.preprocessor.keys()), label=f"Preprocessor", value="none")
            model = gr.Dropdown(list(cn_models.keys()), label=f"Model", value="None")
            refresh_models = ToolButton(value=refresh_symbol)
            refresh_models.click(refresh_all_models, model, model)
                # ctrls += (refresh_models, )
        with gr.Row():
            weight = gr.Slider(label=f"Weight", value=1.0, minimum=0.0, maximum=2.0, step=.05)
            guidance_start = gr.Slider(label="Guidance Start (T)", value=0.0, minimum=0.0, maximum=1.0, interactive=True)
            guidance_end = gr.Slider(label="Guidance End (T)", value=1.0, minimum=0.0, maximum=1.0, interactive=True)

            ctrls += (module, model, weight,)
                # model_dropdowns.append(model)
        def build_sliders(module):
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
            elif module in ["hed", "fake_scribble"]:
                return [
                    gr.update(label="HED Resolution", minimum=64, maximum=2048, value=512, step=1, interactive=True),
                    gr.update(label="Threshold A", value=64, minimum=64, maximum=1024, interactive=False),
                    gr.update(label="Threshold B", value=64, minimum=64, maximum=1024, interactive=False),
                    gr.update(visible=True)
                ]
            elif module in ["openpose", "openpose_hand", "segmentation"]:
                return [
                    gr.update(label="Annotator Resolution", minimum=64, maximum=2048, value=512, step=1, interactive=True),
                    gr.update(label="Threshold A", value=64, minimum=64, maximum=1024, interactive=False),
                    gr.update(label="Threshold B", value=64, minimum=64, maximum=1024, interactive=False),
                    gr.update(visible=True)
                ]
            elif module == "depth":
                return [
                    gr.update(label="Midas Resolution", minimum=64, maximum=2048, value=384, step=1, interactive=True),
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
            processor_res = gr.Slider(label="Annotator resolution", value=64, minimum=64, maximum=2048, interactive=False)
            threshold_a =  gr.Slider(label="Threshold A", value=64, minimum=64, maximum=1024, interactive=False)
            threshold_b =  gr.Slider(label="Threshold B", value=64, minimum=64, maximum=1024, interactive=False)
            
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
                return input_image.orgpreprocess(inputs)
            return None

        resize_mode = gr.Radio(choices=[e.value for e in ResizeMode], value=ResizeMode.INNER_FIT.value, label="Resize Mode")
        with gr.Row():
            with gr.Column():
                canvas_width = gr.Slider(label="Canvas Width", minimum=256, maximum=1024, value=512, step=64)
                canvas_height = gr.Slider(label="Canvas Height", minimum=256, maximum=1024, value=512, step=64)
                    
            if gradio_compat:
                canvas_swap_res = ToolButton(value=switch_values_symbol)
                canvas_swap_res.click(lambda w, h: (h, w), inputs=[canvas_width, canvas_height], outputs=[canvas_width, canvas_height])
                    
        create_button = gr.Button(value="Create blank canvas")
        create_button.click(fn=create_canvas, inputs=[canvas_height, canvas_width], outputs=[input_image])
        
        def run_annotator(image, module, pres, pthr_a, pthr_b):
            img = HWC3(image['image'])
            if not ((image['mask'][:, :, 0]==0).all() or (image['mask'][:, :, 0]==255).all()):
                img = HWC3(image['mask'][:, :, 0])
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
        
        annotator_button.click(fn=run_annotator, inputs=[input_image, module, processor_res, threshold_a, threshold_b], outputs=[generated_image])
        annotator_button_hide.click(fn=lambda: gr.update(visible=False), inputs=None, outputs=[generated_image])

        if is_img2img:
            send_dimen_button.click(fn=send_dimensions, inputs=[input_image], outputs=[self.img2img_w_slider, self.img2img_h_slider])
        else:
            send_dimen_button.click(fn=send_dimensions, inputs=[input_image], outputs=[self.txt2img_w_slider, self.txt2img_h_slider])                                        
        
        ctrls += (input_image, scribble_mode, resize_mode, rgbbgr_mode)
        ctrls += (lowvram,)
        ctrls += (processor_res, threshold_a, threshold_b, guidance_start, guidance_end, guess_mode)
            
        input_image.orgpreprocess=input_image.preprocess
        input_image.preprocess=svgPreprocess
    
        return ctrls
        

    def ui(self, is_img2img):
        """this function should create gradio UI elements. See https://gradio.app/docs/#components
        The return value should be an array of all components that are used in processing.
        Values of those returned components will be passed to run() and process() functions.
        """
        self.infotext_fields = []
        ctrls_group = (
            gr.State(is_img2img),
            gr.State(True),  # is_ui
        )
        max_models = shared.opts.data.get("control_net_max_models_num", 1)
        with gr.Group():
            with gr.Accordion("ControlNet", open = False, elem_id="controlnet"):
                if max_models > 1:
                    with gr.Tabs():
                            for i in range(max_models):
                                with gr.Tab(f"Control Model - {i}"):
                                    ctrls = self.uigroup(is_img2img)
                                    self.register_modules(f"ControlNet-{i}", ctrls)
                                    ctrls_group += ctrls
                else:
                    with gr.Column():
                        ctrls = self.uigroup(is_img2img)
                        self.register_modules(f"ControlNet", ctrls)
                        ctrls_group += ctrls

                return ctrls_group
            
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

        model_path = cn_models.get(model, None)
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
        network_config = shared.opts.data.get("control_net_model_config", default_conf)
        if not os.path.isabs(network_config):
            network_config = os.path.join(script_dir, network_config)

        if any([k.startswith("body.") or k == 'style_embedding' for k, v in state_dict.items()]):
            # adapter model     
            network_module = PlugableAdapter
            network_config = shared.opts.data.get("control_net_model_adapter_config", default_conf_adapter)
            if not os.path.isabs(network_config):
                network_config = os.path.join(script_dir, network_config)
            
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

        def get_element(obj, idx, strict=False):
            if not isinstance(obj, list):
                return obj if not strict or idx == 0 else None
            elif idx < len(obj):
                return obj[idx]
            else:
                return None

        attribute_value = get_element(getattr(p, attribute, None), idx, strict)
        default_value = get_element(default, idx)
        return attribute_value if attribute_value is not None else default_value

    def parse_remote_call(self, p, params, idx):
        if params is None:
            params = [None] * PARAM_COUNT

        enabled, module, model, weight, image, scribble_mode, \
            resize_mode, rgbbgr_mode, lowvram, pres, pthr_a, pthr_b, guidance_start, guidance_end, guess_mode = params

        selector = self.get_remote_call

        enabled = selector(p, "control_net_enabled", enabled, idx, strict=True)
        module = selector(p, "control_net_module", module, idx)
        model = selector(p, "control_net_model", model, idx)
        weight = selector(p, "control_net_weight", weight, idx)
        image = selector(p, "control_net_image", image, idx)
        scribble_mode = selector(p, "control_net_scribble_mode", scribble_mode, idx)
        resize_mode = selector(p, "control_net_resize_mode", resize_mode, idx)
        rgbbgr_mode = selector(p, "control_net_rgbbgr_mode", rgbbgr_mode, idx)
        lowvram = selector(p, "control_net_lowvram", lowvram, idx)
        pres = selector(p, "control_net_pres", pres, idx)
        pthr_a = selector(p, "control_net_pthr_a", pthr_a, idx)
        pthr_b = selector(p, "control_net_pthr_b", pthr_b, idx)
        guidance_strength = selector(p, "control_net_guidance_strength", 1.0, idx)
        guidance_start = selector(p, "control_net_guidance_start", guidance_start, idx)
        guidance_end = selector(p, "control_net_guidance_end", guidance_end, idx)
        guess_mode = selector(p, "control_net_guess_mode", guess_mode, idx)
        if guidance_strength < 1.0:
            # for backward compatible
            guidance_end = guidance_strength

        input_image = selector(p, "control_net_input_image", None, idx)

        return (enabled, module, model, weight, image, scribble_mode, \
            resize_mode, rgbbgr_mode, lowvram, pres, pthr_a, pthr_b, guidance_start, guidance_end, guess_mode), input_image
        
    def detectmap_proc(self, detected_map, module, rgbbgr_mode, resize_mode, h, w):
        detected_map = HWC3(detected_map)
        if module == "normal_map" or rgbbgr_mode:
            control = torch.from_numpy(detected_map[:, :, ::-1].copy()).float().to(devices.get_device_for("controlnet")) / 255.0
        else:
            control = torch.from_numpy(detected_map.copy()).float().to(devices.get_device_for("controlnet")) / 255.0
            
        control = rearrange(control, 'h w c -> c h w')
        detected_map = rearrange(torch.from_numpy(detected_map), 'h w c -> c h w')

        if resize_mode == ResizeMode.INNER_FIT:
            transform = Compose([
                Resize(h if h<w else w, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(size=(h, w)),
            ])
            control = transform(control)
            detected_map = transform(detected_map)
        elif resize_mode == ResizeMode.OUTER_FIT:
            transform = Compose([
                Resize(h if h>w else w, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(size=(h, w))
            ]) 
            control = transform(control)
            detected_map = transform(detected_map)
        else:
            control = Resize((h,w), interpolation=InterpolationMode.BICUBIC)(control)
            detected_map = Resize((h,w), interpolation=InterpolationMode.BICUBIC)(detected_map)
            
        # for log use
        detected_map = rearrange(detected_map, 'c h w -> h w c').numpy().astype(np.uint8)
        return control, detected_map

    def process(self, p, is_img2img=False, is_ui=False, *args):
        """
        This function is called before processing begins for AlwaysVisible scripts.
        You can modify the processing object (p) here, inject hooks, etc.
        args contains all values returned by components from ui()
        """
        unet = p.sd_model.model.diffusion_model
        if self.latest_network is not None:
            # always restore (~0.05s)
            self.latest_network.restore(unet)

        control_groups = []
        params_group = [args[i:i + PARAM_COUNT] for i in range(0, len(args), PARAM_COUNT)]
        if len(params_group) == 0:
            # fill a null group
            params, _ = self.parse_remote_call(p, None, 0)
            if params[0]: # enabled
                params_group.append(params)

        for idx, params in enumerate(params_group):
            params, _ = self.parse_remote_call(p, params, idx)
            enabled, module, model, weight, image, scribble_mode, \
                resize_mode, rgbbgr_mode, lowvram, pres, pthr_a, pthr_b, guidance_start, guidance_end, guess_mode = params

            if not enabled:
                continue
            control_groups.append((module, model, params))
            if len(params_group) != 1:
                prefix = f"ControlNet-{idx}"
            else:
                prefix = "ControlNet"
            p.extra_generation_params.update({
                f"{prefix} Enabled": True,
                f"{prefix} Module": module,
                f"{prefix} Model": model,
                f"{prefix} Weight": weight,
                f"{prefix} Guidance Start": guidance_start,
                f"{prefix} Guidance End": guidance_end,
            })

        if len(params_group) == 0 or len(control_groups) == 0:
           self.latest_network = None
           return 

        detected_maps = []
        forward_params = []
        hook_lowvram = False
        
        # cache stuff
        if self.latest_model_hash != p.sd_model.sd_model_hash:
            self.clear_control_model_cache()

        # unload unused preproc
        module_list = [mod[0] for mod in control_groups]
        for key in self.unloadable:
            if key not in module_list:
                self.unloadable.get(module, lambda:None)()
            
        self.latest_model_hash = p.sd_model.sd_model_hash
        for idx, contents in enumerate(control_groups):
            module, model, params = contents
            _, input_image = self.parse_remote_call(p, params, idx)
            enabled, module, model, weight, image, scribble_mode, \
                resize_mode, rgbbgr_mode, lowvram, pres, pthr_a, pthr_b, guidance_start, guidance_end, guess_mode = params

            resize_mode = resize_mode_from_value(resize_mode)

            if lowvram:
                hook_lowvram = True
                
            model_net = self.load_control_model(p, unet, model, lowvram)
            model_net.reset()

            is_img2img_batch_tab = is_img2img and img2img_tab_tracker.submit_img2img_tab == 'img2img_batch_tab'
            if is_img2img_batch_tab and hasattr(p, "image_control") and p.image_control is not None:
                input_image = HWC3(np.asarray(p.image_control)) 
            elif input_image is not None:
                input_image = HWC3(np.asarray(input_image))
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
                    scribble_mode = True
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
                    
            if scribble_mode:
                detected_map = np.zeros_like(input_image, dtype=np.uint8)
                detected_map[np.min(input_image, axis=2) < 127] = 255
                input_image = detected_map
            
            print(f"Loading preprocessor: {module}")
            preprocessor = self.preprocessor[module]
            h, w, bsz = p.height, p.width, p.batch_size
            if pres > 64:
                detected_map, is_image = preprocessor(input_image, res=pres, thr_a=pthr_a, thr_b=pthr_b)
            else:
                detected_map, is_image = preprocessor(input_image)
            
            if is_image:
                control, detected_map = self.detectmap_proc(detected_map, module, rgbbgr_mode, resize_mode, h, w)
                detected_maps.append((detected_map, module))
            else:
                control = detected_map  

            forward_param = ControlParams(
                control_model=model_net,
                hint_cond=control,
                guess_mode=guess_mode,
                weight=weight,
                guidance_stopped=False,
                start_guidance_percent=guidance_start,
                stop_guidance_percent=guidance_end,
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
            
        if len(control_groups) > 0 and shared.opts.data.get("control_net_skip_img2img_processing") and hasattr(p, "init_images"):
            swap_img2img_pipeline(p)

    def postprocess(self, p, processed, is_img2img=False, is_ui=False, *args):
        if shared.opts.data.get("control_net_detectmap_autosaving", False) and self.latest_network is not None:
            for detect_map, module in self.detected_map:
                detectmap_dir = os.path.join(shared.opts.data.get("control_net_detectedmap_dir", False), module)
                if not os.path.isabs(detectmap_dir):
                    detectmap_dir = os.path.join(p.outpath_samples, detectmap_dir)
                if module != "none":
                    os.makedirs(detectmap_dir, exist_ok=True)
                    img = Image.fromarray(detect_map)
                    save_image(img, detectmap_dir, module)

        is_img2img_batch_tab = is_ui and is_img2img and img2img_tab_tracker.submit_img2img_tab == 'img2img_batch_tab'
        if self.latest_network is None or is_img2img_batch_tab:
            return

        no_detectmap_opt = shared.opts.data.get("control_net_no_detectmap", False)
        if not no_detectmap_opt and hasattr(self, "detected_map") and self.detected_map is not None:
            for detect_map, module in self.detected_map:
                if detect_map is None:
                    continue
                if module in ["canny", "mlsd", "scribble", "fake_scribble", "pidinet", "binary"]:
                    detect_map = 255-detect_map
                processed.images.extend([Image.fromarray(detect_map)])

        self.input_image = None
        self.latest_network.restore(p.sd_model.model.diffusion_model)
        self.latest_network = None

        gc.collect()
        devices.torch_gc()

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
        default_conf, "Config file for Control Net models", section=section))
    shared.opts.add_option("control_net_model_adapter_config", shared.OptionInfo(
        default_conf_adapter, "Config file for Adapter models", section=section))
    shared.opts.add_option("control_net_detectedmap_dir", shared.OptionInfo(
        default_detectedmap_dir, "Directory for detected maps auto saving", section=section))
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
    shared.opts.add_option("control_net_only_midctrl_hires", shared.OptionInfo(
        True, "Use mid-control on highres pass (second pass)", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("control_net_allow_script_control", shared.OptionInfo(
        False, "Allow other script to control this extension", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("control_net_skip_img2img_processing", shared.OptionInfo(
        False, "Skip img2img processing when using img2img initial image", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("control_net_monocular_depth_optim", shared.OptionInfo(
        False, "Enable optimized monocular depth estimation", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("control_net_only_mid_control", shared.OptionInfo(
        False, "Only use mid-control when inference", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("control_net_cfg_based_guidance", shared.OptionInfo(
        False, "Enable CFG-Based guidance", gr.Checkbox, {"interactive": True}, section=section))
    # shared.opts.add_option("control_net_advanced_weighting", shared.OptionInfo(
    #     False, "Enable advanced weight tuning", gr.Checkbox, {"interactive": False}, section=section))
    
    
class Img2ImgTabTracker:
    def __init__(self):
        self.img2img_tabs = set()
        self.active_img2img_tab = 'img2img_img2img_tab'
        self.submit_img2img_tab = None

    def save_submit_img2img_tab(self):
        self.submit_img2img_tab = self.active_img2img_tab

    def set_active_img2img_tab(self, tab):
        self.active_img2img_tab = tab.elem_id

    def on_after_component_callback(self, component, **_kwargs):
        if type(component) is gr.State:
            return

        if type(component) is gr.Button and component.elem_id == 'img2img_generate':
            component.click(fn=self.save_submit_img2img_tab, inputs=[], outputs=[])
            return

        tab = getattr(component, 'parent', None)
        is_tab = type(tab) is gr.Tab and getattr(tab, 'elem_id', None) is not None
        is_img2img_tab = is_tab and getattr(tab, 'parent', None) is not None and getattr(tab.parent, 'elem_id', None) == 'mode_img2img'
        if is_img2img_tab and tab.elem_id not in self.img2img_tabs:
            tab.select(fn=self.set_active_img2img_tab, inputs=gr.State(tab), outputs=[])
            self.img2img_tabs.add(tab.elem_id)
            return


img2img_tab_tracker = Img2ImgTabTracker()
script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_after_component(img2img_tab_tracker.on_after_component_callback)
