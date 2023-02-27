import numpy as np
from fastapi import FastAPI, Body
from PIL import Image
import copy
import contextlib
import pydantic
import sys

import gradio as gr

from modules import ui
from modules.api.models import *
from modules.api import api
from modules.processing import StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img

import modules.scripts as scripts

from scripts.controlnet import update_cn_models, cn_models_names
from scripts.processor import *

def to_base64_nparray(encoding: str):
    return np.array(api.decode_base64_to_image(encoding)).astype('uint8')

def encode_to_base64(image):
    if type(image) is str:
        return image
    elif type(image) is Image.Image:
        return api.encode_pil_to_base64(image)
    elif type(image) is np.ndarray:
        return encode_np_to_base64(image)
    else:
        return ""

def encode_np_to_base64(image):
    pil = Image.fromarray(image)
    return api.encode_pil_to_base64(pil)

cn_root_field_prefix = 'controlnet_'
cn_fields = {
    "input_image": (str, Field(default="", title='ControlNet Input Image')),
    "mask": (str, Field(default="", title='ControlNet Input Mask')),
    "module": (str, Field(default="none", title='Controlnet Module')),
    "model": (str, Field(default="None", title='Controlnet Model')),
    "weight": (float, Field(default=1.0, title='Controlnet Weight')),
    "resize_mode": (str, Field(default="Scale to Fit (Inner Fit)", title='Controlnet Resize Mode')),
    "lowvram": (bool, Field(default=False, title='Controlnet Low VRAM')),
    "processor_res": (int, Field(default=64, title='Controlnet Processor Res')),
    "threshold_a": (float, Field(default=64, title='Controlnet Threshold a')),
    "threshold_b": (float, Field(default=64, title='Controlnet Threshold b')),
    "guidance": (float, Field(default=1.0, title='ControlNet Guidance Strength')),
    "guidance_start": (float, Field(0.0, title='ControlNet Guidance Start')),
    "guidance_end": (float, Field(1.0, title='ControlNet Guidance End')),
    "guessmode": (bool, Field(default=True, title="Guess Mode")),
}

def get_deprecated_cn_field(field_name: str, field):
    field_type, field = field
    field = copy.copy(field)
    field.default = None
    field.extra['deprecated'] = True
    if field_name in ('input_image', 'mask'):
        field_type = List[field_type]
    return f'{cn_root_field_prefix}{field_name}', (field_type, field)

def get_deprecated_field_default(field_name: str):
    if field_name in ('input_image', 'mask'):
        return []
    return cn_fields[field_name][-1].default

ControlNetUnitRequest = pydantic.create_model('ControlNetUnitRequest', **cn_fields)

def create_controlnet_request_model(p_api_class):
    class RequestModel(p_api_class):
        class Config(p_api_class.__config__):
            @staticmethod
            def schema_extra(schema: dict, _):
                props = {}
                for k, v in schema.get('properties', {}).items():
                    if not v.get('deprecated', False):
                        props[k] = v
                    if v.get('docs_default', None) is not None:
                        v['default'] = v['docs_default']
                if props:
                    schema['properties'] = props

    additional_fields = {
        'controlnet_units': (List[ControlNetUnitRequest], Field(default=[], docs_default=[ControlNetUnitRequest()], description="ControlNet Processing Units")),
        **dict(get_deprecated_cn_field(k, v) for k, v in cn_fields.items())
    }

    return pydantic.create_model(
        f'ControlNet{p_api_class.__name__}',
        __base__=RequestModel,
        **additional_fields)

ControlNetTxt2ImgRequest = create_controlnet_request_model(StableDiffusionTxt2ImgProcessingAPI)
ControlNetImg2ImgRequest = create_controlnet_request_model(StableDiffusionImg2ImgProcessingAPI)

class ApiHijack(api.Api):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_api_route("/controlnet/txt2img", self.controlnet_txt2img, methods=["POST"], response_model=TextToImageResponse)
        self.add_api_route("/controlnet/img2img", self.controlnet_img2img, methods=["POST"], response_model=ImageToImageResponse)

    def controlnet_txt2img(self, txt2img_request: ControlNetTxt2ImgRequest):
        return self.controlnet_any2img(
            any2img_request=txt2img_request,
            original_callback=ApiHijack.text2imgapi,
            p_class=StableDiffusionProcessingTxt2Img,
            script_runner=scripts.scripts_txt2img,
        )

    def controlnet_img2img(self, img2img_request: ControlNetImg2ImgRequest):
        return self.controlnet_any2img(
            any2img_request=img2img_request,
            original_callback=ApiHijack.img2imgapi,
            p_class=StableDiffusionProcessingImg2Img,
            script_runner=scripts.scripts_img2img,
        )

    def controlnet_any2img(self, any2img_request, original_callback, p_class, script_runner):
        any2img_request = nest_deprecated_cn_fields(any2img_request)
        script_runner = create_cn_script_runner(script_runner, any2img_request.controlnet_units)
        delattr(any2img_request, 'controlnet_units')
        with self.queue_lock:
            self_copy = copy.copy(self)
            self_copy.queue_lock = contextlib.nullcontext()
            with OverrideInit(p_class, scripts=script_runner):
                return original_callback(self_copy, any2img_request)

api.Api = ApiHijack

class OverrideInit:
    def __init__(self, cls, **kwargs):
        self.cls = cls
        self.kwargs = kwargs
        self.original_init = None

    def __enter__(self):
        def init_hijack(p, *args, **kwargs):
            self.original_init(p, *args, **kwargs)
            for k, v in self.kwargs.items():
                setattr(p, k, v)

        self.original_init = self.cls.__init__
        self.cls.__init__ = init_hijack

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cls.__init__ = self.original_init

def nest_deprecated_cn_fields(any2img_request):
    deprecated_cn_fields = {k: v for k, v in vars(any2img_request).items()
                            if k.startswith(cn_root_field_prefix) and k != 'controlnet_units'}

    any2img_request = copy.copy(any2img_request)
    for k in deprecated_cn_fields.keys():
        delattr(any2img_request, k)

    if all(v is None for v in deprecated_cn_fields.values()):
        return any2img_request

    warn_deprecated_cn_params()
    deprecated_cn_fields = {k[len(cn_root_field_prefix):]: v for k, v in deprecated_cn_fields.items()}
    for k, v in deprecated_cn_fields.items():
        if v is None:
            deprecated_cn_fields[k] = get_deprecated_field_default(k)

    for k in ('input_image', 'mask'):
        deprecated_cn_fields[k] = deprecated_cn_fields[k][0] if deprecated_cn_fields[k] else ""

    any2img_request.controlnet_units.insert(0, ControlNetUnitRequest(**deprecated_cn_fields))
    return any2img_request

def create_cn_script_runner(script_runner, control_unit_requests: List[ControlNetUnitRequest]):
    if not script_runner.scripts:
        script_runner.initialize_scripts(False)
        ui.create_ui()

    cn_script_runner = copy.copy(script_runner)

    cn_script_args = [False]  # is_img2img
    for control_unit_request in control_unit_requests:
        cn_script_args += create_cn_unit_args(control_unit_request)

    script_titles = [script.title().lower() for script in script_runner.alwayson_scripts]
    cn_script_id = script_titles.index('controlnet')
    cn_script = copy.copy(script_runner.alwayson_scripts[cn_script_id])
    cn_script.args_from = 0
    cn_script.args_to = len(cn_script_args)

    def make_script_runner_f_hijack(fixed_original_f):
        def script_runner_f_hijack(p, *args, **kwargs):
            original_script_args = p.script_args
            p.script_args = cn_script_args
            fixed_original_f(p, *args, **kwargs)
            p.script_args = original_script_args

        return script_runner_f_hijack

    for k in ('process', 'process_batch', 'postprocess', 'postprocess_batch', 'postprocess_image'):
        original_f = getattr(cn_script_runner, k, None)
        if original_f is None:
            continue

        setattr(cn_script_runner, k, make_script_runner_f_hijack(original_f))

    cn_script_runner.alwayson_scripts = [cn_script]
    return cn_script_runner

def create_cn_unit_args(unit_request: ControlNetUnitRequest):
    input_image = to_base64_nparray(unit_request.input_image) if unit_request.input_image else None
    mask = None
    if input_image is not None:
        if unit_request.mask:
            mask = to_base64_nparray(unit_request.mask)
        else:
            mask = input_image * 0

        if len(mask.shape) == 2:
            mask = mask[..., np.newaxis]

    if unit_request.guidance < 1.0:
        unit_request.guidance_end = unit_request.guidance

    return (
        True,  # enabled
        unit_request.module,
        unit_request.model,
        unit_request.weight,
        {
            "image": input_image,
            "mask": mask,
        } if input_image is not None else None,  # input_image
        False,  # scribble_mode
        unit_request.resize_mode,
        False,  # rgbbgr_mode
        unit_request.lowvram,
        unit_request.processor_res,
        unit_request.threshold_a,
        unit_request.threshold_b,
        unit_request.guidance_start,
        unit_request.guidance_end,
        unit_request.guessmode
    )

def warn_deprecated_cn_params():
    warning_prefix = '[ControlNet] warning: '
    print(f"{warning_prefix}using deprecated '{cn_root_field_prefix}*' request params", file=sys.stderr)
    print(f"{warning_prefix}consider using the 'control_units' request param instead", file=sys.stderr)

def controlnet_api(_: gr.Blocks, app: FastAPI):
    @app.get("/controlnet/model_list")
    async def model_list():
        update_cn_models()
        print(list(cn_models_names.values()))
        return {"model_list": list(cn_models_names.values())}

    @app.post("/controlnet/detect")
    async def detect(
        controlnet_module: str = Body("None", title='Controlnet Module'),
        controlnet_input_images: List[str] = Body([], title='Controlnet Input Images'),
        controlnet_processor_res: int = Body(512, title='Controlnet Processor Resolution'),
        controlnet_threshold_a: float = Body(64, title='Controlnet Threshold a'),
        controlnet_threshold_b: float = Body(64, title='Controlnet Threshold b')
    ):

        available_modules = [
            "canny", 
            "depth", 
            "depth_leres", 
            "fake_scribble", 
            "hed", 
            "mlsd", 
            "normal_map", 
            "openpose", 
            "segmentation"
        ]

        if controlnet_module not in available_modules:
            return {"images": [], "info": "Module not available"}
        if len(controlnet_input_images) == 0:
            return {"images": [], "info": "No image selected"}
        
        print(f"Detecting {str(len(controlnet_input_images))} images with the {controlnet_module} module.")

        results = []

        for input_image in controlnet_input_images:
            img = to_base64_nparray(input_image)

            if controlnet_module == "canny":
                results.append(canny(img, controlnet_processor_res, controlnet_threshold_a, controlnet_threshold_b))
            elif controlnet_module == "hed":
                results.append(hed(img, controlnet_processor_res))
            elif controlnet_module == "mlsd":
                results.append(mlsd(img, controlnet_processor_res, controlnet_threshold_a, controlnet_threshold_b))
            elif controlnet_module == "depth":
                results.append(midas(img, controlnet_processor_res, np.pi * 2.0))
            elif controlnet_module == "normal_map":
                results.append(midas_normal(img, controlnet_processor_res, np.pi * 2.0, controlnet_threshold_a))
            elif controlnet_module == "depth_leres":
                results.append(leres(img, controlnet_processor_res, np.pi * 2.0, controlnet_threshold_a, controlnet_threshold_b))
            elif controlnet_module == "openpose":
                results.append(openpose(img, controlnet_processor_res, False))
            elif controlnet_module == "fake_scribble":
                results.append(fake_scribble(img, controlnet_processor_res))
            elif controlnet_module == "segmentation":
                results.append(uniformer(img, controlnet_processor_res))

        if controlnet_module == "hed":
            unload_hed()
        elif controlnet_module == "mlsd":
            unload_mlsd()
        elif controlnet_module == "depth" or controlnet_module == "normal_map":
            unload_midas()
        elif controlnet_module == "depth_leres":
            unload_leres()
        elif controlnet_module == "openpose":
            unload_openpose()
        elif controlnet_module == "segmentation":
            unload_uniformer()

        results64 = list(map(encode_to_base64, results))
        return {"images": results64, "info": "Success"}

try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(controlnet_api)
except:
    pass
