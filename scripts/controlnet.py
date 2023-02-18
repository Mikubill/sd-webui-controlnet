import os
import stat
from collections import OrderedDict

import torch

import modules.scripts as scripts
from modules import shared, devices, script_callbacks, processing
import gradio as gr
import numpy as np

from einops import rearrange
from scripts.cldm import PlugableControlModel
from scripts.processor import *
from scripts.adapter import PlugableAdapter
from scripts.utils import load_state_dict
from modules import sd_models
from modules.processing import StableDiffusionProcessingImg2Img
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
cn_models = {}      # "My_Lora(abcd1234)" -> C:/path/to/model.safetensors
cn_models_names = {}  # "my_lora" -> "My_Lora(abcd1234)"
cn_models_dir = os.path.join(scripts.basedir(), "models")
os.makedirs(cn_models_dir, exist_ok=True)
default_conf_adapter = os.path.join(cn_models_dir, "sketch_adapter_v14.yaml")
default_conf = os.path.join(cn_models_dir, "cldm_v15.yaml")
refresh_symbol = '\U0001f504'  # 🔄
switch_values_symbol = '\U000021C5' # ⇅


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
    global cn_models, cn_models_names
    res = OrderedDict()
    ext_dirs = (shared.opts.data.get("control_net_models_path", None), getattr(shared.cmd_opts, 'controlnet_dir', None))
    extra_lora_paths = (extra_lora_path for extra_lora_path in ext_dirs
                if extra_lora_path is not None and os.path.exists(extra_lora_path))
    paths = [cn_models_dir, *extra_lora_paths]

    for path in paths:
        sort_by = shared.opts.data.get(
            "control_net_models_sort_models_by", "name")
        filter_by = shared.opts.data.get("control_net_models_name_filter", "")
        found = get_all_models(sort_by, filter_by, path)
        res = {**found, **res}

    cn_models = OrderedDict(**{"None": None}, **res)
    cn_models_names = {}
    for name_and_hash, filename in cn_models.items():
        if filename == None:
            continue
        name = os.path.splitext(os.path.basename(filename))[0].lower()
        cn_models_names[name] = name_and_hash


update_cn_models()


class Script(scripts.Script):
    def __init__(self) -> None:
        super().__init__()
        self.latest_params = (None, None)
        self.latest_network = None
        self.preprocessor = {
            "none": lambda x, *args, **kwargs: x,
            "canny": canny,
            "depth": midas,
            "depth_leres": leres,
            "hed": hed,
            "mlsd": mlsd,
            "normal_map": midas_normal,
            "openpose": openpose,
            "openpose_hand": openpose_hand,
            "pidinet": pidinet,
            "scribble": simple_scribble,
            "fake_scribble": fake_scribble,
            "segmentation": uniformer,
        }
        self.unloadable = {
            "hed": unload_hed,
            "fake_scribble": unload_hed,
            "mlsd": unload_mlsd,
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

    def title(self):
        return "ControlNet for generating"

    def show(self, is_img2img):
        # if is_img2img:
            # return False
        return scripts.AlwaysVisible
    
    def get_threshold_block(self, proc):
        pass

    def ui(self, is_img2img):
        """this function should create gradio UI elements. See https://gradio.app/docs/#components
        The return value should be an array of all components that are used in processing.
        Values of those returned components will be passed to run() and process() functions.
        """
        ctrls = ()
        model_dropdowns = []
        self.infotext_fields = []
        with gr.Group():
            with gr.Accordion('ControlNet', open=False):
                input_image = gr.Image(source='upload', type='numpy', tool='sketch')
                gr.HTML(value='<p>Enable scribble mode if your image has white background.<br >Change your brush width to make it thinner if you want to draw something.<br ></p>')

                with gr.Row():
                    enabled = gr.Checkbox(label='Enable', value=False)
                    scribble_mode = gr.Checkbox(label='Scribble Mode (Invert colors)', value=False)
                    rgbbgr_mode = gr.Checkbox(label='RGB to BGR', value=False)
                    lowvram = gr.Checkbox(label='Low VRAM', value=False)
                    
                ctrls += (enabled,)
                self.infotext_fields.append((enabled, "ControlNet Enabled"))
                
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
                    guidance_strength =  gr.Slider(label="Guidance strength (T)", value=1.0, minimum=0.0, maximum=1.0, interactive=True)

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
                    elif module == "depth_leres":
                        return [
                            gr.update(label="LeReS Resolution", minimum=64, maximum=2048, value=384, step=1, interactive=True),
                            gr.update(label="Threshold A", value=64, minimum=64, maximum=1024, interactive=False),
                            gr.update(label="Threshold B", value=64, minimum=64, maximum=1024, interactive=False),
                            gr.update(visible=True)
                        ]
                    elif module == "normal_map":
                        return [
                            gr.update(label="Normal Resolution", minimum=64, maximum=2048, value=512, step=1, interactive=True),
                            gr.update(label="Normal background threshold", minimum=0.0, maximum=1.0, value=0.4, step=0.01, interactive=True),
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
                    
                self.infotext_fields.extend([
                    (module, f"ControlNet Preprocessor"),
                    (model, f"ControlNet Model"),
                    (weight, f"ControlNet Weight"),
                ])

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

                resize_mode = gr.Radio(choices=["Envelope (Outer Fit)", "Scale to Fit (Inner Fit)", "Just Resize"], value="Scale to Fit (Inner Fit)", label="Resize Mode")
                with gr.Row():
                    with gr.Column():
                        canvas_width = gr.Slider(label="Canvas Width", minimum=256, maximum=1024, value=512, step=64)
                        canvas_height = gr.Slider(label="Canvas Height", minimum=256, maximum=1024, value=512, step=64)
                        
                    if gradio_compat:
                        canvas_swap_res = ToolButton(value=switch_values_symbol)
                        
                create_button = gr.Button(value="Create blank canvas")            
                create_button.click(fn=create_canvas, inputs=[canvas_height, canvas_width], outputs=[input_image])
                
                if gradio_compat:
                    canvas_swap_res.click(lambda w, h: (h, w), inputs=[canvas_width, canvas_height], outputs=[canvas_width, canvas_height])
                    
                ctrls += (input_image, scribble_mode, resize_mode, rgbbgr_mode)
                ctrls += (lowvram,)
                ctrls += (processor_res, threshold_a, threshold_b, guidance_strength)
                
                input_image.orgpreprocess=input_image.preprocess
                input_image.preprocess=svgPreprocess

        return ctrls

    def set_infotext_fields(self, p, params, weight):
        module, model = params
        if model == "None" or model == "none":
            return
        p.extra_generation_params.update({
            "ControlNet Enabled": True,
            f"ControlNet Module": module,
            f"ControlNet Model": model,
            f"ControlNet Weight": weight,
        })

    def process(self, p, *args):
        """
        This function is called before processing begins for AlwaysVisible scripts.
        You can modify the processing object (p) here, inject hooks, etc.
        args contains all values returned by components from ui()
        """
        unet = p.sd_model.model.diffusion_model

        def restore_networks():
            if self.latest_network is not None:
                print("restoring last networks")
                self.input_image = None
                self.latest_network.restore(unet)
                self.latest_network = None

            last_module = self.latest_params[0]
            if last_module is not None:
                self.unloadable.get(last_module, lambda:None)()

        enabled, module, model, weight, image, scribble_mode, \
            resize_mode, rgbbgr_mode, lowvram, pres, pthr_a, pthr_b, guidance_strength = args
        
        # Other scripts can control this extension now
        if shared.opts.data.get("control_net_allow_script_control", False):
            enabled = getattr(p, 'control_net_enabled', enabled)
            module = getattr(p, 'control_net_module', module)
            model = getattr(p, 'control_net_model', model)
            weight = getattr(p, 'control_net_weight', weight)
            image = getattr(p, 'control_net_image', image)
            scribble_mode = getattr(p, 'control_net_scribble_mode', scribble_mode)
            resize_mode = getattr(p, 'control_net_resize_mode', resize_mode)
            rgbbgr_mode = getattr(p, 'control_net_rgbbgr_mode', rgbbgr_mode)
            lowvram = getattr(p, 'control_net_lowvram', lowvram)
            pres = getattr(p, 'control_net_pres', pres)
            pthr_a = getattr(p, 'control_net_pthr_a', pthr_a)
            pthr_b = getattr(p, 'control_net_pthr_b', pthr_b)
            guidance_strength = getattr(p, 'control_net_guidance_strength', guidance_strength)

            input_image = getattr(p, 'control_net_input_image', None)
        else:
            input_image = None

        if not enabled:
            restore_networks()
            return

        models_changed = self.latest_params[1] != model \
            or self.latest_model_hash != p.sd_model.sd_model_hash or self.latest_network == None \
            or (self.latest_network is not None and self.latest_network.lowvram != lowvram)

        self.latest_params = (module, model)
        self.latest_model_hash = p.sd_model.sd_model_hash
        if models_changed:
            restore_networks()
            model_path = cn_models.get(model, None)

            if model_path is None:
                raise RuntimeError(f"model not found: {model}")

            # trim '"' at start/end
            if model_path.startswith("\"") and model_path.endswith("\""):
                model_path = model_path[1:-1]

            if not os.path.exists(model_path):
                raise ValueError(f"file not found: {model_path}")

            print(f"Loading preprocessor: {module}, model: {model}")
            state_dict = load_state_dict(model_path)
            network_module = PlugableControlModel
            network_config = shared.opts.data.get("control_net_model_config", default_conf)
            if any([k.startswith("body.") for k, v in state_dict.items()]):
                # adapter model     
                network_module = PlugableAdapter
                network_config = shared.opts.data.get("control_net_model_adapter_config", default_conf_adapter)

            network = network_module(
                state_dict=state_dict, 
                config_path=network_config, 
                weight=weight, 
                lowvram=lowvram,
                base_model=unet,
            )
            network.to(p.sd_model.device, dtype=p.sd_model.dtype)
            network.hook(unet, p.sd_model)

            print(f"ControlNet model {model} loaded.")
            self.latest_network = network
          
        if input_image is not None:
            input_image = HWC3(np.asarray(input_image))
        elif image is not None:
            input_image = HWC3(image['image'])
            if not ((image['mask'][:, :, 0]==0).all() or (image['mask'][:, :, 0]==255).all()):
                print("using mask as input")
                input_image = HWC3(image['mask'][:, :, 0])
                scribble_mode = True
        else:
            # use img2img init_image as default
            input_image = getattr(p, "init_images", [None])[0]
            if input_image is None:
                raise ValueError('controlnet is enabled but no input image is given')
            input_image = HWC3(np.asarray(input_image))
                
        if scribble_mode:
            detected_map = np.zeros_like(input_image, dtype=np.uint8)
            detected_map[np.min(input_image, axis=2) < 127] = 255
            input_image = detected_map
                
        preprocessor = self.preprocessor[self.latest_params[0]]
        h, w, bsz = p.height, p.width, p.batch_size
        if pres > 64:
            detected_map = preprocessor(input_image, res=pres, thr_a=pthr_a, thr_b=pthr_b)
        else:
            detected_map = preprocessor(input_image)
        detected_map = HWC3(detected_map)
        
        if module == "normal_map" or rgbbgr_mode:
            control = torch.from_numpy(detected_map[:, :, ::-1].copy()).float().to(devices.get_device_for("controlnet")) / 255.0
        else:
            control = torch.from_numpy(detected_map.copy()).float().to(devices.get_device_for("controlnet")) / 255.0
        
        control = rearrange(control, 'h w c -> c h w')
        detected_map = rearrange(torch.from_numpy(detected_map), 'h w c -> c h w')
        if resize_mode == "Scale to Fit (Inner Fit)":
            transform = Compose([
                Resize(h if h<w else w, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(size=(h, w))
            ]) 
            control = transform(control)
            detected_map = transform(detected_map)
        elif resize_mode == "Envelope (Outer Fit)":
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
        self.detected_map = rearrange(detected_map, 'c h w -> h w c').numpy().astype(np.uint8)
            
        # control = torch.stack([control for _ in range(bsz)], dim=0)
        self.latest_network.notify(control, weight, guidance_strength)
        self.set_infotext_fields(p, self.latest_params, weight)

        if shared.opts.data.get("control_net_skip_img2img_processing") and hasattr(p, "init_images"):
            swap_img2img_pipeline(p)

    def postprocess(self, p, processed, *args):
        is_img2img = issubclass(type(p), StableDiffusionProcessingImg2Img)
        is_img2img_batch_tab = is_img2img and img2img_tab_tracker.submit_img2img_tab == 'img2img_batch_tab'
        no_detectmap_opt = shared.opts.data.get("control_net_no_detectmap", False)
        if self.latest_network is None or no_detectmap_opt or is_img2img_batch_tab:
            return
        if hasattr(self, "detected_map") and self.detected_map is not None:
            result =  self.detected_map
            if self.latest_params[0] in ["canny", "mlsd", "scribble", "fake_scribble"]:
                result = 255-result
            processed.images.extend([result])

def update_script_args(p, value, arg_idx):
    for s in scripts.scripts_txt2img.alwayson_scripts:
        if isinstance(s, Script):
            args = list(p.script_args)
            # print(f"Changed arg {arg_idx} from {args[s.args_from + arg_idx - 1]} to {value}")
            args[s.args_from + arg_idx] = value
            p.script_args = tuple(args)
            break


# def confirm_models(p, xs):
#     for x in xs:
#         if x in ["", "None"]:
#             continue
#         if not find_closest_lora_model_name(x):
#             raise RuntimeError(f"Unknown ControlNet model: {x}")

def on_ui_settings():
    section = ('control_net', "ControlNet")
    shared.opts.add_option("control_net_model_config", shared.OptionInfo(
        default_conf, "Config file for Control Net models", section=section))
    shared.opts.add_option("control_net_model_adapter_config", shared.OptionInfo(
        default_conf_adapter, "Config file for Adapter models", section=section))
    shared.opts.add_option("control_net_models_path", shared.OptionInfo(
        "", "Extra path to scan for ControlNet models (e.g. training output directory)", section=section))

    shared.opts.add_option("control_net_control_transfer", shared.OptionInfo(
        False, "Apply transfer control when loading models", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("control_net_no_detectmap", shared.OptionInfo(
        False, "Do not append detectmap to output", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("control_net_only_midctrl_hires", shared.OptionInfo(
        True, "Use mid-control on highres pass (second pass)", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("control_net_allow_script_control", shared.OptionInfo(
        False, "Allow other script to control this extension", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("control_net_skip_img2img_processing", shared.OptionInfo(
        False, "Skip img2img processing when using img2img initial image", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("control_net_only_mid_control", shared.OptionInfo(
        False, "Only use mid-control when inference", gr.Checkbox, {"interactive": True}, section=section))
    

    # control_net_skip_hires


script_callbacks.on_ui_settings(on_ui_settings)


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
        
        if not hasattr(component, "parent"):
            return

        tab = component.parent
        is_tab = type(tab) is gr.Tab and tab.elem_id is not None
        is_img2img_tab = is_tab and tab.parent is not None and tab.parent.elem_id == 'mode_img2img'
        if is_img2img_tab and tab.elem_id not in self.img2img_tabs:
            tab.select(fn=self.set_active_img2img_tab, inputs=gr.State(tab), outputs=[])
            self.img2img_tabs.add(tab.elem_id)
            return


img2img_tab_tracker = Img2ImgTabTracker()
script_callbacks.on_after_component(img2img_tab_tracker.on_after_component_callback)





from typing import Any, List, Dict, Set, Union
from fastapi import FastAPI, Body, HTTPException, Request, Response
import base64
import io
from io import BytesIO
from PIL import PngImagePlugin,Image
import piexif
import piexif.helper

from modules.api.models import *
from modules.processing import StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img, process_images

from modules import sd_samplers
from modules.shared import opts, cmd_opts
import modules.shared as shared


def validate_sampler_name(name):
    config = sd_samplers.all_samplers_map.get(name, None)
    if config is None:
        raise HTTPException(status_code=404, detail="Sampler not found")

    return name

def decode_base64_to_image(encoding):
    if encoding.startswith("data:image/"):
        encoding = encoding.split(";")[1].split(",")[1]
    try:
        image = Image.open(BytesIO(base64.b64decode(encoding)))
        return image
    except Exception as err:
        raise HTTPException(status_code=500, detail="Invalid encoded image")   

def encode_pil_to_base64(image):
    with io.BytesIO() as output_bytes:

        if opts.samples_format.lower() == 'png':
            use_metadata = False
            metadata = PngImagePlugin.PngInfo()
            for key, value in image.info.items():
                if isinstance(key, str) and isinstance(value, str):
                    metadata.add_text(key, value)
                    use_metadata = True
            image.save(output_bytes, format="PNG", pnginfo=(metadata if use_metadata else None), quality=opts.jpeg_quality)

        elif opts.samples_format.lower() in ("jpg", "jpeg", "webp"):
            parameters = image.info.get('parameters', None)
            exif_bytes = piexif.dump({
                "Exif": { piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(parameters or "", encoding="unicode") }
            })
            if opts.samples_format.lower() in ("jpg", "jpeg"):
                image.save(output_bytes, format="JPEG", exif = exif_bytes, quality=opts.jpeg_quality)
            else:
                image.save(output_bytes, format="WEBP", exif = exif_bytes, quality=opts.jpeg_quality)

        else:
            raise HTTPException(status_code=500, detail="Invalid image format")

        bytes_data = output_bytes.getvalue()

    return base64.b64encode(bytes_data)

def encode_np_to_base64(image):
    pil_img = Image.fromarray(image)
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG")
    return base64.b64encode(pil_img.getvalue()).decode("utf-8")

def controlnet_api(_: gr.Blocks, app: FastAPI):

    @app.post("/controlnet/txt2img")
    async def txt2img(
        prompt: str = Body("", title='Prompt'),
        negative_prompt: str = Body("", title='Negative Prompt'),
        controlnet_input_image: List[str] = Body([], title='ControlNet Input Image'),
        controlnet_mask: List[str] = Body([], title='ControlNet Input Mask'),
        controlnet_module: str = Body("", title='Controlnet Module'),
        controlnet_model: str = Body("", title='Controlnet Model'),
        controlnet_weight: float = Body(1.0, title='Controlnet Weight'),
        controlnet_resize_mode: str = Body("Scale to Fit (Inner Fit)", title='Controlnet Resize Mode'),
        controlnet_lowvram: bool = Body(True, title='Controlnet Low VRAM'),
        controlnet_processor_res: int = Body(512, title='Controlnet Processor Res'),
        controlnet_threshold_a: int = Body(64, title='Controlnet Threshold a'),
        controlnet_threshold_b: int = Body(64, title='Controlnet Threshold b'),
        seed: int = Body(-1, title="Seed"),
        subseed: int = Body(-1, title="Subseed"),
        subseed_strength: int = Body(-1, title="Subseed Strength"),
        sampler_index: str = Body("", title='Sampler Name'),
        batch_size: int = Body(1, title="Batch Size"),
        n_iter: int = Body(1, title="Iteration"),
        steps: int = Body(20, title="Steps"),
        cfg_scale: float = Body(7, title="CFG"),
        width: int = Body(512, title="width"),
        height: int = Body(512, title="height"),
        restore_faces: bool = Body(True, title="Restore Faces"),
        override_settings: Dict[str, Any] = Body(None, title="Override Settings"),
        override_settings_restore_afterwards: bool = Body(True, title="Restore Override Settings Afterwards"),    
        ):

        p = StableDiffusionProcessingTxt2Img(
            sd_model=shared.sd_model,
            outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
            outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
            prompt=prompt,
            styles=[],
            negative_prompt=negative_prompt,
            seed=seed,
            subseed=subseed,
            subseed_strength=subseed_strength,
            seed_resize_from_h=-1,
            seed_resize_from_w=-1,
            seed_enable_extras=False,
            sampler_name=sampler_index,
            batch_size=batch_size,
            n_iter=n_iter,
            steps=steps,
            cfg_scale=cfg_scale,
            width=width,
            height=height,
            restore_faces=restore_faces,
            tiling=False,
            enable_hr=False,
            denoising_strength=None,
            hr_scale=2,
            hr_upscaler=None,
            hr_second_pass_steps=0,
            hr_resize_x=0,
            hr_resize_y=0,
            override_settings=override_settings,
            do_not_save_samples=True,
            do_not_save_grid=True,
        )

        cn_image = Image.open(io.BytesIO(base64.b64decode(controlnet_input_image[0])))        
        cn_image_np = np.array(cn_image).astype('uint8')

        if(controlnet_mask == []):
            cn_mask_np = np.zeros(shape=(512, 512, 3)).astype('uint8')
        else:
            cn_mask = Image.open(io.BytesIO(base64.b64decode(controlnet_mask[0])))        
            cn_mask_np = np.array(cn_mask).astype('uint8')
     
        cn_args = {
            "enabled": True,
            "module": controlnet_module,
            "model": controlnet_model,
            "weight": controlnet_weight,
            "input_image": {'image': cn_image_np, 'mask': cn_mask_np},
            "scribble_mode": False,
            "resize_mode": controlnet_resize_mode,
            "rgbbgr_mode": False,
            "lowvram": controlnet_lowvram,
            "processor_res": controlnet_processor_res,
            "threshold_a": controlnet_threshold_a,
            "threshold_b": controlnet_threshold_b,
        }

        p.scripts = scripts.scripts_txt2img
        p.script_args = (
            0, # todo: why
            cn_args["enabled"],
            cn_args["module"],
            cn_args["model"],
            cn_args["weight"],
            cn_args["input_image"],
            cn_args["scribble_mode"],
            cn_args["resize_mode"],
            cn_args["rgbbgr_mode"],
            cn_args["lowvram"],
            cn_args["processor_res"],
            cn_args["threshold_a"],
            cn_args["threshold_b"],
            0, False, False, False, False, '', 1, '', 0, '', 0, '', True, False, False, False # todo: extend to include wither alwaysvisible scripts
        )

        print(p.script_args)

        if cmd_opts.enable_console_prompts:
            print(f"\ntxt2img: {prompt}", file=shared.progress_print_out)

        shared.state.begin()

        processed = scripts.scripts_txt2img.run(p, *(p.script_args))
        
        if processed is None: # fall back
           processed = process_images(p)            

        p.close()

        shared.state.end()

        generation_info_js = processed.js()
        if opts.samples_log_stdout:
            print(generation_info_js)

        if opts.do_not_show_images:
            processed.images = []

        b64images = list(map(encode_pil_to_base64, processed.images[0:1]))
        
        return {"images": b64images, "info": processed.js()}


    @app.post("/controlnet/img2img")
    async def img2img(
        init_images: List[str] = Body([], title='Init Images'),
        mask: str = Body(None, title='Mask'),
        mask_blur: int = Body(30, title='Mask Blur'),
        inpainting_fill: int = Body(0, title='Inpainting Fill'),
        inpaint_full_res: bool = Body(True, title='Inpainting Full Resolution'),
        inpaint_full_res_padding: int = Body(1, title='Inpainting Full Resolution Padding'),
        inpainting_mask_invert: int = Body(1, title='Mask Invert'),
        resize_mode: int = Body(0, title='Resize Mode'),
        denoising_strength: float = Body(0.7, title='Denoising Strength'),
        prompt: str = Body("", title='Prompt'),
        negative_prompt: str = Body("", title='Negative Prompt'),
        controlnet_input_image: List[str] = Body([], title='ControlNet Input Image'),
        controlnet_mask: List[str] = Body([], title='ControlNet Input Mask'),
        controlnet_module: str = Body("", title='Controlnet Module'),
        controlnet_model: str = Body("", title='Controlnet Model'),
        controlnet_weight: float = Body(1.0, title='Controlnet Weight'),
        controlnet_resize_mode: str = Body("Scale to Fit (Inner Fit)", title='Controlnet Resize Mode'),
        controlnet_lowvram: bool = Body(True, title='Controlnet Low VRAM'),
        controlnet_processor_res: int = Body(512, title='Controlnet Processor Res'),
        controlnet_threshold_a: int = Body(64, title='Controlnet Threshold a'),
        controlnet_threshold_b: int = Body(64, title='Controlnet Threshold b'),
        seed: int = Body(-1, title="Seed"),
        subseed: int = Body(-1, title="Subseed"),
        subseed_strength: int = Body(-1, title="Subseed Strength"),
        sampler_index: str = Body("", title='Sampler Name'),
        batch_size: int = Body(1, title="Batch Size"),
        n_iter: int = Body(1, title="Iteration"),
        steps: int = Body(20, title="Steps"),
        cfg_scale: float = Body(7, title="CFG"),
        width: int = Body(512, title="width"),
        height: int = Body(512, title="height"),
        restore_faces: bool = Body(True, title="Restore Faces"),
        include_init_images: bool = Body(True, title="Include Init Images"),
        override_settings: Dict[str, Any] = Body(None, title="Override Settings"),
        override_settings_restore_afterwards: bool = Body(True, title="Restore Override Settings Afterwards"),    
        ):

        if mask:
            mask = decode_base64_to_image(mask)

        p = StableDiffusionProcessingImg2Img(
            sd_model=shared.sd_model,
            outpath_samples=opts.outdir_samples or opts.outdir_img2img_samples,
            outpath_grids=opts.outdir_grids or opts.outdir_img2img_grids,
            prompt=prompt,
            negative_prompt=negative_prompt,
            init_images=[decode_base64_to_image(x) for x in init_images],
            styles=[],
            seed=seed,
            subseed=subseed,
            subseed_strength=subseed_strength,
            seed_resize_from_h=-1,
            seed_resize_from_w=-1,
            seed_enable_extras=False,
            sampler_name=sampler_index,
            batch_size=batch_size,
            n_iter=n_iter,
            steps=steps,
            cfg_scale=cfg_scale,
            width=width,
            height=height,
            restore_faces=restore_faces,
            tiling=False,
            mask=mask,
            mask_blur=mask_blur,
            inpainting_fill=inpainting_fill,
            resize_mode=resize_mode,
            denoising_strength=denoising_strength,
            inpaint_full_res=inpaint_full_res,
            inpaint_full_res_padding=inpaint_full_res_padding,
            inpainting_mask_invert=inpainting_mask_invert,
            override_settings=override_settings,
            do_not_save_samples=True,
            do_not_save_grid=True,
        )

        cn_image = Image.open(io.BytesIO(base64.b64decode(controlnet_input_image[0])))        
        cn_image_np = np.array(cn_image).astype('uint8')

        if(controlnet_mask == []):
            cn_mask_np = np.zeros(shape=(512, 512, 3)).astype('uint8')
        else:
            cn_mask = Image.open(io.BytesIO(base64.b64decode(controlnet_mask[0])))        
            cn_mask_np = np.array(cn_mask).astype('uint8')
     
        cn_args = {
            "enabled": True,
            "module": controlnet_module,
            "model": controlnet_model,
            "weight": controlnet_weight,
            "input_image": {'image': cn_image_np, 'mask': cn_mask_np},
            "scribble_mode": False,
            "resize_mode": controlnet_resize_mode,
            "rgbbgr_mode": False,
            "lowvram": controlnet_lowvram,
            "processor_res": controlnet_processor_res,
            "threshold_a": controlnet_threshold_a,
            "threshold_b": controlnet_threshold_b,
        }

        p.scripts = scripts.scripts_txt2img
        p.script_args = (
            0, # todo: why
            cn_args["enabled"],
            cn_args["module"],
            cn_args["model"],
            cn_args["weight"],
            cn_args["input_image"],
            cn_args["scribble_mode"],
            cn_args["resize_mode"],
            cn_args["rgbbgr_mode"],
            cn_args["lowvram"],
            cn_args["processor_res"],
            cn_args["threshold_a"],
            cn_args["threshold_b"],
            0, False, False, False, False, '', 1, '', 0, '', 0, '', True, False, False, False # default args
        )

        if shared.cmd_opts.enable_console_prompts:
            print(f"\nimg2img: {prompt}", file=shared.progress_print_out)

        p.extra_generation_params["Mask blur"] = mask_blur

        shared.state.begin()

        processed = scripts.scripts_img2img.run(p, *(p.script_args)) # todo: extend to include wither alwaysvisible scripts
        
        if processed is None: # fall back
           processed = process_images(p)            

        p.close()

        shared.state.end()

        generation_info_js = processed.js()
        if opts.samples_log_stdout:
            print(generation_info_js)

        if opts.do_not_show_images:
            processed.images = []

        b64images = list(map(encode_pil_to_base64, processed.images[0:1]))
        
        return {"images": b64images, "info": processed.js()}


try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(controlnet_api)
except:
    pass
