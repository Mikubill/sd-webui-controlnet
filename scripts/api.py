from typing import List, Optional
import base64
import io
import torch
import numpy as np
from fastapi import FastAPI, Body
from fastapi.exceptions import HTTPException
from pydantic import BaseModel

from PIL import Image

import gradio as gr

from modules.api.models import *  # noqa:F403
from modules.api import api

from scripts import external_code, global_state
from scripts.logging import logger
from scripts.external_code import ControlNetUnit
from scripts.supported_preprocessor import Preprocessor
from annotator.openpose import draw_poses, decode_json_as_poses
from annotator.openpose.animalpose import draw_animalposes


def encode_to_base64(image):
    if isinstance(image, str):
        return image
    elif isinstance(image, Image.Image):
        return api.encode_pil_to_base64(image)
    elif isinstance(image, np.ndarray):
        return encode_np_to_base64(image)
    else:
        return ""


def encode_np_to_base64(image):
    pil = Image.fromarray(image)
    return api.encode_pil_to_base64(pil)


def encode_tensor_to_base64(obj: torch.Tensor) -> str:
    """Serialize the tensor data to base64 string."""
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    buffer.seek(0)  # Rewind the buffer
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def controlnet_api(_: gr.Blocks, app: FastAPI):
    @app.get("/controlnet/version")
    async def version():
        return {"version": external_code.get_api_version()}

    @app.get("/controlnet/model_list")
    async def model_list(update: bool = True):
        up_to_date_model_list = external_code.get_models(update=update)
        logger.debug(up_to_date_model_list)
        return {"model_list": up_to_date_model_list}

    @app.get("/controlnet/module_list")
    async def module_list(alias_names: bool = False):
        _module_list = external_code.get_modules(alias_names)
        logger.debug(_module_list)

        return {
            "module_list": _module_list,
            "module_detail": external_code.get_modules_detail(alias_names),
        }

    @app.get("/controlnet/control_types")
    async def control_types():
        def format_control_type(
            filtered_preprocessor_list,
            filtered_model_list,
            default_option,
            default_model,
        ):
            return {
                "module_list": filtered_preprocessor_list,
                "model_list": filtered_model_list,
                "default_option": default_option,
                "default_model": default_model,
            }

        return {
            "control_types": {
                control_type: format_control_type(
                    *global_state.select_control_type(control_type)
                )
                for control_type in Preprocessor.get_all_preprocessor_tags()
            }
        }

    @app.get("/controlnet/settings")
    async def settings():
        max_models_num = external_code.get_max_models_num()
        return {"control_net_unit_count": max_models_num}

    @app.post("/controlnet/detect")
    async def detect(
        controlnet_module: str = Body("none", title="Controlnet Module"),
        controlnet_input_images: List[str] = Body([], title="Controlnet Input Images"),
        controlnet_processor_res: int = Body(
            -1, title="Controlnet Processor Resolution"
        ),
        controlnet_threshold_a: float = Body(-1, title="Controlnet Threshold a"),
        controlnet_threshold_b: float = Body(-1, title="Controlnet Threshold b"),
        controlnet_masks: List[str] = Body([], title="Controlnet Masks"),
        low_vram: bool = Body(False, title="Low vram"),
    ):
        preprocessor = Preprocessor.get_preprocessor(controlnet_module)

        if preprocessor is None:
            raise HTTPException(status_code=422, detail="Module not available")

        if controlnet_module in (
            "clip_vision",
            "revision_clipvision",
            "revision_ignore_prompt",
            "ip-adapter-auto",
        ):
            raise HTTPException(status_code=422, detail="Module not supported")

        if len(controlnet_input_images) == 0:
            raise HTTPException(status_code=422, detail="No image selected")

        if preprocessor.requires_mask and len(controlnet_masks) != len(
            controlnet_input_images
        ):
            raise HTTPException(
                status_code=422,
                detail=f"Preprocessor {controlnet_module} requires `controlnet_masks` param.",
            )

        logger.info(
            f"Detecting {str(len(controlnet_input_images))} images with the {controlnet_module} module."
        )

        unit = ControlNetUnit(
            enabled=True,
            module=preprocessor.label,
            processor_res=controlnet_processor_res,
            threshold_a=controlnet_threshold_a,
            threshold_b=controlnet_threshold_b,
        )

        tensors = []
        images = []
        poses = []

        for i, input_image in enumerate(controlnet_input_images):
            img = external_code.to_base64_nparray(input_image)
            # Has mask.
            if i < len(controlnet_masks):
                if preprocessor.accepts_mask:
                    mask = external_code.to_base64_nparray(controlnet_masks[i])[
                        :, :, :1
                    ]
                    img = np.concatenate([img, mask], axis=2)
                else:
                    logger.warn(
                        f"Preprocessor {controlnet_module} does not accept mask. Mask ignored"
                    )

            class JsonAcceptor:
                def __init__(self) -> None:
                    self.value = None

                def accept(self, json_dict: dict) -> None:
                    self.value = json_dict

            json_acceptor = JsonAcceptor()
            result = preprocessor.cached_call(
                img,
                resolution=unit.processor_res,
                slider_1=unit.threshold_a,
                slider_2=unit.threshold_b,
                json_pose_callback=json_acceptor.accept,
                low_vram=low_vram,
            )
            if preprocessor.returns_image:
                images.append(encode_to_base64(result.display_images[0]))
            else:
                tensors.append(encode_tensor_to_base64(result.value))

            if "openpose" in controlnet_module:
                assert json_acceptor.value is not None
                poses.append(json_acceptor.value)

        preprocessor.unload()

        res = {"info": "Success"}
        if poses:
            res["poses"] = poses
        if images:
            res["images"] = images
        if tensors:
            res["tensor"] = tensors

        return res

    class Person(BaseModel):
        pose_keypoints_2d: List[float]
        hand_right_keypoints_2d: Optional[List[float]]
        hand_left_keypoints_2d: Optional[List[float]]
        face_keypoints_2d: Optional[List[float]]

    class PoseData(BaseModel):
        people: List[Person]
        canvas_width: int
        canvas_height: int

    @app.post("/controlnet/render_openpose_json")
    async def render_openpose_json(
        pose_data: List[PoseData] = Body([], title="Pose json files to render.")
    ):
        if not pose_data:
            return {"info": "No pose data detected."}
        else:

            def draw(poses, animals, H, W):
                if poses:
                    assert len(animals) == 0
                    return draw_poses(poses, H, W)
                else:
                    return draw_animalposes(animals, H, W)

            return {
                "images": [
                    encode_to_base64(draw(*decode_json_as_poses(pose.dict())))
                    for pose in pose_data
                ],
                "info": "Success",
            }


try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(controlnet_api)
except Exception:
    logger.warn("Unable to mount ControlNet API.")
