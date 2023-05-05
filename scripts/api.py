import numpy as np
from fastapi import FastAPI, Body
from fastapi.exceptions import HTTPException
from PIL import Image

import gradio as gr

from modules.api.models import *
from modules.api import api

from scripts import external_code, global_state
from scripts.processor import preprocessor_sliders_config

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

def controlnet_api(_: gr.Blocks, app: FastAPI):
    @app.get("/controlnet/version")
    async def version():
        return {"version": external_code.get_api_version()}

    @app.get("/controlnet/model_list")
    async def model_list():
        up_to_date_model_list = external_code.get_models(update=True)
        print(up_to_date_model_list)
        return {"model_list": up_to_date_model_list}

    @app.get("/controlnet/module_list")
    async def module_list(alias_names: bool = False):
        _module_list = external_code.get_modules(False)
        _module_list_alias = external_code.get_modules(True)
        _module_detail = {}
        
        _output_list = _module_list if not alias_names else _module_list_alias
        for index, module in enumerate(_output_list):
            if _module_list[index] in preprocessor_sliders_config:
                _module_detail[module] = {
                    "sliders": preprocessor_sliders_config[_module_list[index]]
                }
            else:
                _module_detail[module] = {
                    "sliders": []
                }
        
        return {
            "module_list": _output_list,
            "module_detail": _module_detail
        }

    @app.post("/controlnet/detect")
    async def detect(
        controlnet_module: str = Body("none", title='Controlnet Module'),
        controlnet_input_images: List[str] = Body([], title='Controlnet Input Images'),
        controlnet_processor_res: int = Body(512, title='Controlnet Processor Resolution'),
        controlnet_threshold_a: float = Body(64, title='Controlnet Threshold a'),
        controlnet_threshold_b: float = Body(64, title='Controlnet Threshold b')
    ):
        controlnet_module = global_state.reverse_preprocessor_aliases.get(controlnet_module, controlnet_module)

        if controlnet_module not in global_state.cn_preprocessor_modules:
            raise HTTPException(
                status_code=422, detail="Module not available")

        if len(controlnet_input_images) == 0:
            raise HTTPException(
                status_code=422, detail="No image selected")

        print(f"Detecting {str(len(controlnet_input_images))} images with the {controlnet_module} module.")

        results = []

        processor_module = global_state.cn_preprocessor_modules[controlnet_module]

        for input_image in controlnet_input_images:
            img = external_code.to_base64_nparray(input_image)
            results.append(processor_module(img, res=controlnet_processor_res, thr_a=controlnet_threshold_a, thr_b=controlnet_threshold_b)[0])

        global_state.cn_preprocessor_unloadable.get(controlnet_module, lambda: None)()
        results64 = list(map(encode_to_base64, results))
        return {"images": results64, "info": "Success"}

try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(controlnet_api)
except:
    pass
