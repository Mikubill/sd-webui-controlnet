import os
import stat
from collections import OrderedDict

import torch

import modules.scripts as scripts
from modules import shared, devices, script_callbacks
import gradio as gr

import numpy as np
from einops import rearrange
from modules import sd_models
from torchvision.transforms import Resize, InterpolationMode, CenterCrop, Compose
from scripts.cldm import PlugableControlModel
from scripts.processor import *
from modules.ui_components import ToolButton

CN_MODEL_EXTS = [".pt", ".pth", ".ckpt", ".safetensors"]
cn_models = {}      # "My_Lora(abcd1234)" -> C:/path/to/model.safetensors
cn_models_names = {}  # "my_lora" -> "My_Lora(abcd1234)"
cn_models_dir = os.path.join(scripts.basedir(), "models")
os.makedirs(cn_models_dir, exist_ok=True)
default_conf = os.path.join(cn_models_dir, "cldm_v15.yaml")
refresh_symbol = '\U0001f504'  # 🔄

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
            "hed": hed,
            "mlsd": mlsd,
            "normal_map": midas_normal,
            "openpose": openpose,
            "openpose_hand": openpose_hand,
            "scribble": simple_scribble,
            "fake_scribble": fake_scribble,
            "segmentation": uniformer,
        }
        self.unloadable = {
            "hed": unload_hed,
            "fake_scribble": unload_hed,
            "mlsd": unload_mlsd,
            "depth": unload_midas,
            "normal_map": unload_midas,
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
                    weight = gr.Slider(label=f"Weight", value=1.0, minimum=0.0, maximum=2.0, step=.05)

                    ctrls += (module, model, weight,)
                    # model_dropdowns.append(model)
                    
                def build_sliders(module):
                    if module == "canny":
                        return [
                            gr.Slider.update(label="Annotator resolution", value=512, minimum=64, maximum=1024, step=1, interactive=True),
                            gr.Slider.update(label="Canny low threshold", minimum=1, maximum=255, value=100, step=1, interactive=True),
                            gr.Slider.update(label="Canny high threshold", minimum=1, maximum=255, value=200, step=1, interactive=True),
                        ]
                    elif module == "mlsd": #Hough
                        return [
                            gr.Slider.update(label="Hough Resolution", minimum=128, maximum=1024, value=512, step=1, interactive=True),
                            gr.Slider.update(label="Hough value threshold (MLSD)", minimum=0.01, maximum=2.0, value=0.1, step=0.01, interactive=True),
                            gr.Slider.update(label="Hough distance threshold (MLSD)", minimum=0.01, maximum=20.0, value=0.1, step=0.01, interactive=True)
                        ]
                    elif module in ["hed", "fake_scribble"]:
                        return [
                            gr.Slider.update(label="HED Resolution", minimum=128, maximum=1024, value=512, step=1, interactive=True),
                            gr.Slider.update(label="Threshold A", value=64, minimum=64, maximum=1024, interactive=False),
                            gr.Slider.update(label="Threshold B", value=64, minimum=64, maximum=1024, interactive=False),
                        ]
                    elif module in ["openpose", "openpose_hand", "segmentation"]:
                        return [
                            gr.Slider.update(label="Annotator Resolution", minimum=128, maximum=1024, value=512, step=1, interactive=True),
                            gr.Slider.update(label="Threshold A", value=64, minimum=64, maximum=1024, interactive=False),
                            gr.Slider.update(label="Threshold B", value=64, minimum=64, maximum=1024, interactive=False),
                        ]
                    elif module == "depth":
                        return [
                            gr.Slider.update(label="Midas Resolution", minimum=128, maximum=1024, value=384, step=1, interactive=True),
                            gr.Slider.update(label="Threshold A", value=64, minimum=64, maximum=1024, interactive=False),
                            gr.Slider.update(label="Threshold B", value=64, minimum=64, maximum=1024, interactive=False),
                        ]
                    elif module == "normal_map":
                        return [
                            gr.Slider.update(label="Normal Resolution", minimum=128, maximum=1024, value=512, step=1, interactive=True),
                            gr.Slider.update(label="Normal background threshold", minimum=0.0, maximum=1.0, value=0.4, step=0.01, interactive=True),
                            gr.Slider.update(label="Threshold B", value=64, minimum=64, maximum=1024, interactive=False),
                        ]
                    else:
                        return [
                            gr.Slider.update(label="Annotator resolution", value=512, minimum=64, maximum=1024, step=1, interactive=True),
                            gr.Slider.update(label="Threshold A", value=64, minimum=64, maximum=1024, interactive=False),
                            gr.Slider.update(label="Threshold B", value=64, minimum=64, maximum=1024, interactive=False),
                        ]
                    
                # advanced options    
                with gr.Column():
                    processor_res = gr.Slider(label="Annotator resolution", value=64, minimum=64, maximum=1024, interactive=False)
                    threshold_a =  gr.Slider(label="Threshold A", value=64, minimum=64, maximum=1024, interactive=False)
                    threshold_b =  gr.Slider(label="Threshold B", value=64, minimum=64, maximum=1024, interactive=False)
                    
                module.change(build_sliders, inputs=[module], outputs=[processor_res, threshold_a, threshold_b])
                    
                self.infotext_fields.extend([
                    (module, f"ControlNet Preprocessor"),
                    (model, f"ControlNet Model"),
                    (weight, f"ControlNet Weight"),
                ])

                def create_canvas(h, w): 
                    return np.zeros(shape=(h, w, 3), dtype=np.uint8) + 255
                
                resize_mode = gr.Radio(choices=["Envelope (Outer Fit)", "Scale to Fit (Inner Fit)", "Just Resize"], value="Scale to Fit (Inner Fit)", label="Resize Mode")
                with gr.Row():
                    canvas_width = gr.Slider(label="Canvas Width", minimum=256, maximum=1024, value=512, step=64)
                    canvas_height = gr.Slider(label="Canvas Height", minimum=256, maximum=1024, value=512, step=64)
                with gr.Row():
                    create_button = gr.Button(value="Create blank canvas")              
                create_button.click(fn=create_canvas, inputs=[canvas_height, canvas_width], outputs=[input_image])
                ctrls += (input_image, scribble_mode, resize_mode, rgbbgr_mode)
                ctrls += (lowvram,)
                ctrls += (processor_res, threshold_a, threshold_b)
                
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
            resize_mode, rgbbgr_mode, lowvram, pres, pthr_a, pthr_b = args
        
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
            network = PlugableControlModel(
                model_path=model_path, 
                config_path=shared.opts.data.get("control_net_model_config", default_conf), 
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
        detected_map = preprocessor(input_image, res=pres, thr_a=pthr_a, thr_b=pthr_b)
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
        self.latest_network.notify(control, weight)
        self.set_infotext_fields(p, self.latest_params, weight)
        
    def postprocess(self, p, processed, *args):
        if self.latest_network is None or shared.opts.data.get("control_net_no_detectmap", False):
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
    shared.opts.add_option("control_net_models_path", shared.OptionInfo(
        "", "Extra path to scan for ControlNet models (e.g. training output directory)", section=section))

    shared.opts.add_option("control_net_control_transfer", shared.OptionInfo(
        False, "Apply transfer control when loading models", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("control_net_no_detectmap", shared.OptionInfo(
        False, "Do not append detectmap to output", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("control_net_only_midctrl_hires", shared.OptionInfo(
        True, "Use mid-layer control on highres pass (second pass)", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("control_net_allow_script_control", shared.OptionInfo(
        False, "Allow other script to control this extension", gr.Checkbox, {"interactive": True}, section=section))

    # control_net_skip_hires


script_callbacks.on_ui_settings(on_ui_settings)
