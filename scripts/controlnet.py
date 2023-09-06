import gc
import os
import logging
import re
from collections import OrderedDict
from copy import copy
from typing import Dict, Optional, Tuple
import modules.scripts as scripts
from modules import shared, devices, script_callbacks, processing, masking, images
import gradio as gr
import time


from einops import rearrange
from scripts import global_state, hook, external_code, processor, batch_hijack, controlnet_version, utils
from scripts.controlnet_lora import bind_control_lora, unbind_control_lora
from scripts.processor import *
from scripts.adapter import Adapter, StyleAdapter, Adapter_light
from scripts.controlnet_lllite import PlugableControlLLLite, clear_all_lllite
from scripts.controlmodel_ipadapter import PlugableIPAdapter, clear_all_ip_adapter
from scripts.utils import load_state_dict, get_unique_axis0
from scripts.hook import ControlParams, UnetHook, ControlModelType, HackedImageRNG
from scripts.controlnet_ui.controlnet_ui_group import ControlNetUiGroup, UiControlNetUnit
from scripts.logging import logger
from modules.processing import StableDiffusionProcessingImg2Img, StableDiffusionProcessingTxt2Img
from modules.images import save_image
from scripts.infotext import Infotext

import cv2
import numpy as np
import torch

from pathlib import Path
from PIL import Image, ImageFilter, ImageOps
from scripts.lvminthin import lvmin_thin, nake_nms
from scripts.processor import model_free_preprocessors
from scripts.controlnet_model_guess import build_model_by_guess


gradio_compat = True
try:
    from distutils.version import LooseVersion
    from importlib_metadata import version
    if LooseVersion(version("gradio")) < LooseVersion("3.10"):
        gradio_compat = False
except ImportError:
    pass


# Gradio 3.32 bug fix
import tempfile
gradio_tempfile_path = os.path.join(tempfile.gettempdir(), 'gradio')
os.makedirs(gradio_tempfile_path, exist_ok=True)


def clear_all_secondary_control_models():
    clear_all_lllite()
    clear_all_ip_adapter()


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
            image['image'] = np.array(Image.open(image['image'])).astype('uint8')
        elif image['image']:
            image['image'] = external_code.to_base64_nparray(image['image'])
        else:
            image['image'] = None            

    # If there is no image, return image with None image and None mask
    if image['image'] is None:
        image['mask'] = None
        return image

    if isinstance(image['mask'], str):
        if os.path.exists(image['mask']):
            image['mask'] = np.array(Image.open(image['mask'])).astype('uint8')
        elif image['mask']:
            image['mask'] = external_code.to_base64_nparray(image['mask'])
        else:
            image['mask'] = np.zeros_like(image['image'], dtype=np.uint8)
    elif image['mask'] is None:
        image['mask'] = np.zeros_like(image['image'], dtype=np.uint8)

    return image


def image_has_mask(input_image: np.ndarray) -> bool:
    """
    Determine if an image has an alpha channel (mask) that is not empty.

    The function checks if the input image has three dimensions (height, width, channels), 
    and if the third dimension (channel dimension) is of size 4 (presumably RGB + alpha). 
    Then it checks if the maximum value in the alpha channel is greater than 127. This is 
    presumably to check if there is any non-transparent (or semi-transparent) pixel in the 
    image. A pixel is considered non-transparent if its alpha value is above 127.

    Args:
        input_image (np.ndarray): A 3D numpy array representing an image. The dimensions 
        should represent [height, width, channels].

    Returns:
        bool: True if the image has a non-empty alpha channel, False otherwise.
    """    
    return (
        input_image.ndim == 3 and 
        input_image.shape[2] == 4 and 
        np.max(input_image[:, :, 3]) > 127
    )


def prepare_mask(
    mask: Image.Image, p: processing.StableDiffusionProcessing
) -> Image.Image:
    """
    Prepare an image mask for the inpainting process.

    This function takes as input a PIL Image object and an instance of the 
    StableDiffusionProcessing class, and performs the following steps to prepare the mask:

    1. Convert the mask to grayscale (mode "L").
    2. If the 'inpainting_mask_invert' attribute of the processing instance is True,
       invert the mask colors.
    3. If the 'mask_blur' attribute of the processing instance is greater than 0,
       apply a Gaussian blur to the mask with a radius equal to 'mask_blur'.

    Args:
        mask (Image.Image): The input mask as a PIL Image object.
        p (processing.StableDiffusionProcessing): An instance of the StableDiffusionProcessing class 
                                                   containing the processing parameters.

    Returns:
        mask (Image.Image): The prepared mask as a PIL Image object.
    """
    mask = mask.convert("L")
    if getattr(p, "inpainting_mask_invert", False):
        mask = ImageOps.invert(mask)
    
    if hasattr(p, 'mask_blur_x'):
        if getattr(p, "mask_blur_x", 0) > 0:
            np_mask = np.array(mask)
            kernel_size = 2 * int(2.5 * p.mask_blur_x + 0.5) + 1
            np_mask = cv2.GaussianBlur(np_mask, (kernel_size, 1), p.mask_blur_x)
            mask = Image.fromarray(np_mask)
        if getattr(p, "mask_blur_y", 0) > 0:
            np_mask = np.array(mask)
            kernel_size = 2 * int(2.5 * p.mask_blur_y + 0.5) + 1
            np_mask = cv2.GaussianBlur(np_mask, (1, kernel_size), p.mask_blur_y)
            mask = Image.fromarray(np_mask)
    else:
        if getattr(p, "mask_blur", 0) > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(p.mask_blur))
    
    return mask


def set_numpy_seed(p: processing.StableDiffusionProcessing) -> Optional[int]:
    """
    Set the random seed for NumPy based on the provided parameters.

    Args:
        p (processing.StableDiffusionProcessing): The instance of the StableDiffusionProcessing class.

    Returns:
        Optional[int]: The computed random seed if successful, or None if an exception occurs.

    This function sets the random seed for NumPy using the seed and subseed values from the given instance of
    StableDiffusionProcessing. If either seed or subseed is -1, it uses the first value from `all_seeds`.
    Otherwise, it takes the maximum of the provided seed value and 0.

    The final random seed is computed by adding the seed and subseed values, applying a bitwise AND operation
    with 0xFFFFFFFF to ensure it fits within a 32-bit integer.
    """
    try:
        tmp_seed = int(p.all_seeds[0] if p.seed == -1 else max(int(p.seed), 0))
        tmp_subseed = int(p.all_seeds[0] if p.subseed == -1 else max(int(p.subseed), 0))
        seed = (tmp_seed + tmp_subseed) & 0xFFFFFFFF
        np.random.seed(seed)
        return seed
    except Exception as e:
        logger.warning(e)
        logger.warning('Warning: Failed to use consistent random seed.')
        return None


class Script(scripts.Script, metaclass=(
    utils.TimeMeta if logger.level == logging.DEBUG else type)):

    model_cache = OrderedDict()

    def __init__(self) -> None:
        super().__init__()
        self.latest_network = None
        self.preprocessor = global_state.cache_preprocessors(global_state.cn_preprocessor_modules)
        self.unloadable = global_state.cn_preprocessor_unloadable
        self.input_image = None
        self.latest_model_hash = ""
        self.enabled_units = []
        self.detected_map = []
        self.post_processors = []
        self.noise_modifier = None
        batch_hijack.instance.process_batch_callbacks.append(self.batch_tab_process)
        batch_hijack.instance.process_batch_each_callbacks.append(self.batch_tab_process_each)
        batch_hijack.instance.postprocess_batch_each_callbacks.insert(0, self.batch_tab_postprocess_each)
        batch_hijack.instance.postprocess_batch_callbacks.insert(0, self.batch_tab_postprocess)

    def title(self):
        return "ControlNet"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    @staticmethod
    def get_default_ui_unit(is_ui=True):
        cls = UiControlNetUnit if is_ui else external_code.ControlNetUnit
        return cls(
            enabled=False,
            module="none",
            model="None"
        )

    def uigroup(self, tabname: str, is_img2img: bool, elem_id_tabname: str) -> Tuple[ControlNetUiGroup, gr.State]:
        group = ControlNetUiGroup(
            gradio_compat,
            Script.get_default_ui_unit(),
            self.preprocessor,
        )
        group.render(tabname, elem_id_tabname, is_img2img)
        group.register_callbacks(is_img2img)
        return group, group.render_and_register_unit(tabname, is_img2img)

    def ui(self, is_img2img):
        """this function should create gradio UI elements. See https://gradio.app/docs/#components
        The return value should be an array of all components that are used in processing.
        Values of those returned components will be passed to run() and process() functions.
        """
        infotext = Infotext()
        
        controls = ()
        max_models = shared.opts.data.get("control_net_unit_count", 3)
        elem_id_tabname = ("img2img" if is_img2img else "txt2img") + "_controlnet"
        with gr.Group(elem_id=elem_id_tabname):
            with gr.Accordion(f"ControlNet {controlnet_version.version_flag}", open = False, elem_id="controlnet"):
                if max_models > 1:
                    with gr.Tabs(elem_id=f"{elem_id_tabname}_tabs"):
                        for i in range(max_models):
                            with gr.Tab(f"ControlNet Unit {i}", 
                                        elem_classes=['cnet-unit-tab']):
                                group, state = self.uigroup(f"ControlNet-{i}", is_img2img, elem_id_tabname)
                                infotext.register_unit(i, group)
                                controls += (state,)
                else:
                    with gr.Column():
                        group, state = self.uigroup(f"ControlNet", is_img2img, elem_id_tabname)
                        infotext.register_unit(0, group)
                        controls += (state,)

        if shared.opts.data.get("control_net_sync_field_args", True):
            self.infotext_fields = infotext.infotext_fields
            self.paste_field_names = infotext.paste_field_names

        return controls
    
    @staticmethod
    def clear_control_model_cache():
        Script.model_cache.clear()
        gc.collect()
        devices.torch_gc()

    @staticmethod
    def load_control_model(p, unet, model):
        if model in Script.model_cache:
            logger.info(f"Loading model from cache: {model}")
            return Script.model_cache[model]

        # Remove model from cache to clear space before building another model
        if len(Script.model_cache) > 0 and len(Script.model_cache) >= shared.opts.data.get("control_net_model_cache_size", 2):
            Script.model_cache.popitem(last=False)
            gc.collect()
            devices.torch_gc()

        model_net = Script.build_control_model(p, unet, model)

        if shared.opts.data.get("control_net_model_cache_size", 2) > 0:
            Script.model_cache[model] = model_net

        return model_net

    @staticmethod
    def build_control_model(p, unet, model):
        if model is None or model == 'None':
            raise RuntimeError(f"You have not selected any ControlNet Model.")

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

        logger.info(f"Loading model: {model}")
        state_dict = load_state_dict(model_path)
        network = build_model_by_guess(state_dict, unet, model_path)
        network.to('cpu', dtype=p.sd_model.dtype)
        logger.info(f"ControlNet model {model} loaded.")
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

    @staticmethod
    def parse_remote_call(p, unit: external_code.ControlNetUnit, idx):
        selector = Script.get_remote_call

        unit.enabled = selector(p, "control_net_enabled", unit.enabled, idx, strict=True)
        unit.module = selector(p, "control_net_module", unit.module, idx)
        unit.model = selector(p, "control_net_model", unit.model, idx)
        unit.weight = selector(p, "control_net_weight", unit.weight, idx)
        unit.image = selector(p, "control_net_image", unit.image, idx)
        unit.resize_mode = selector(p, "control_net_resize_mode", unit.resize_mode, idx)
        unit.low_vram = selector(p, "control_net_lowvram", unit.low_vram, idx)
        unit.processor_res = selector(p, "control_net_pres", unit.processor_res, idx)
        unit.threshold_a = selector(p, "control_net_pthr_a", unit.threshold_a, idx)
        unit.threshold_b = selector(p, "control_net_pthr_b", unit.threshold_b, idx)
        unit.guidance_start = selector(p, "control_net_guidance_start", unit.guidance_start, idx)
        unit.guidance_end = selector(p, "control_net_guidance_end", unit.guidance_end, idx)
        # Backward compatibility. See https://github.com/Mikubill/sd-webui-controlnet/issues/1740
        # for more details.
        unit.guidance_end = selector(p, "control_net_guidance_strength", unit.guidance_end, idx)
        unit.control_mode = selector(p, "control_net_control_mode", unit.control_mode, idx)
        unit.pixel_perfect = selector(p, "control_net_pixel_perfect", unit.pixel_perfect, idx)

        return unit

    @staticmethod
    def detectmap_proc(detected_map, module, resize_mode, h, w):

        if 'inpaint' in module:
            detected_map = detected_map.astype(np.float32)
        else:
            detected_map = HWC3(detected_map)

        def safe_numpy(x):
            # A very safe method to make sure that Apple/Mac works
            y = x

            # below is very boring but do not change these. If you change these Apple or Mac may fail.
            y = y.copy()
            y = np.ascontiguousarray(y)
            y = y.copy()
            return y

        def get_pytorch_control(x):
            # A very safe method to make sure that Apple/Mac works
            y = x

            # below is very boring but do not change these. If you change these Apple or Mac may fail.
            y = torch.from_numpy(y)
            y = y.float() / 255.0
            y = rearrange(y, 'h w c -> 1 c h w')
            y = y.clone()
            y = y.to(devices.get_device_for("controlnet"))
            y = y.clone()
            return y

        def high_quality_resize(x, size):
            # Written by lvmin
            # Super high-quality control map up-scaling, considering binary, seg, and one-pixel edges

            inpaint_mask = None
            if x.ndim == 3 and x.shape[2] == 4:
                inpaint_mask = x[:, :, 3]
                x = x[:, :, 0:3]

            if x.shape[0] != size[1] or x.shape[1] != size[0]:
                new_size_is_smaller = (size[0] * size[1]) < (x.shape[0] * x.shape[1])
                new_size_is_bigger = (size[0] * size[1]) > (x.shape[0] * x.shape[1])
                unique_color_count = len(get_unique_axis0(x.reshape(-1, x.shape[2])))
                is_one_pixel_edge = False
                is_binary = False
                if unique_color_count == 2:
                    is_binary = np.min(x) < 16 and np.max(x) > 240
                    if is_binary:
                        xc = x
                        xc = cv2.erode(xc, np.ones(shape=(3, 3), dtype=np.uint8), iterations=1)
                        xc = cv2.dilate(xc, np.ones(shape=(3, 3), dtype=np.uint8), iterations=1)
                        one_pixel_edge_count = np.where(xc < x)[0].shape[0]
                        all_edge_count = np.where(x > 127)[0].shape[0]
                        is_one_pixel_edge = one_pixel_edge_count * 2 > all_edge_count

                if 2 < unique_color_count < 200:
                    interpolation = cv2.INTER_NEAREST
                elif new_size_is_smaller:
                    interpolation = cv2.INTER_AREA
                else:
                    interpolation = cv2.INTER_CUBIC  # Must be CUBIC because we now use nms. NEVER CHANGE THIS

                y = cv2.resize(x, size, interpolation=interpolation)
                if inpaint_mask is not None:
                    inpaint_mask = cv2.resize(inpaint_mask, size, interpolation=interpolation)

                if is_binary:
                    y = np.mean(y.astype(np.float32), axis=2).clip(0, 255).astype(np.uint8)
                    if is_one_pixel_edge:
                        y = nake_nms(y)
                        _, y = cv2.threshold(y, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        y = lvmin_thin(y, prunings=new_size_is_bigger)
                    else:
                        _, y = cv2.threshold(y, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    y = np.stack([y] * 3, axis=2)
            else:
                y = x

            if inpaint_mask is not None:
                inpaint_mask = (inpaint_mask > 127).astype(np.float32) * 255.0
                inpaint_mask = inpaint_mask[:, :, None].clip(0, 255).astype(np.uint8)
                y = np.concatenate([y, inpaint_mask], axis=2)

            return y

        if resize_mode == external_code.ResizeMode.RESIZE:
            detected_map = high_quality_resize(detected_map, (w, h))
            detected_map = safe_numpy(detected_map)
            return get_pytorch_control(detected_map), detected_map

        old_h, old_w, _ = detected_map.shape
        old_w = float(old_w)
        old_h = float(old_h)
        k0 = float(h) / old_h
        k1 = float(w) / old_w

        safeint = lambda x: int(np.round(x))

        if resize_mode == external_code.ResizeMode.OUTER_FIT:
            k = min(k0, k1)
            borders = np.concatenate([detected_map[0, :, :], detected_map[-1, :, :], detected_map[:, 0, :], detected_map[:, -1, :]], axis=0)
            high_quality_border_color = np.median(borders, axis=0).astype(detected_map.dtype)
            if len(high_quality_border_color) == 4:
                # Inpaint hijack
                high_quality_border_color[3] = 255
            high_quality_background = np.tile(high_quality_border_color[None, None], [h, w, 1])
            detected_map = high_quality_resize(detected_map, (safeint(old_w * k), safeint(old_h * k)))
            new_h, new_w, _ = detected_map.shape
            pad_h = max(0, (h - new_h) // 2)
            pad_w = max(0, (w - new_w) // 2)
            high_quality_background[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = detected_map
            detected_map = high_quality_background
            detected_map = safe_numpy(detected_map)
            return get_pytorch_control(detected_map), detected_map
        else:
            k = max(k0, k1)
            detected_map = high_quality_resize(detected_map, (safeint(old_w * k), safeint(old_h * k)))
            new_h, new_w, _ = detected_map.shape
            pad_h = max(0, (new_h - h) // 2)
            pad_w = max(0, (new_w - w) // 2)
            detected_map = detected_map[pad_h:pad_h+h, pad_w:pad_w+w]
            detected_map = safe_numpy(detected_map)
            return get_pytorch_control(detected_map), detected_map

    @staticmethod
    def get_enabled_units(p):
        units = external_code.get_all_units_in_processing(p)
        if len(units) == 0:
            # fill a null group
            remote_unit = Script.parse_remote_call(p, Script.get_default_ui_unit(), 0)
            if remote_unit.enabled:
                units.append(remote_unit)
        
        enabled_units = [
            copy(local_unit)
            for idx, unit in enumerate(units)
            for local_unit in (Script.parse_remote_call(p, unit, idx),)
            if local_unit.enabled
        ]
        Infotext.write_infotext(enabled_units, p)
        return enabled_units

    @staticmethod
    def choose_input_image(
            p: processing.StableDiffusionProcessing, 
            unit: external_code.ControlNetUnit,
            idx: int
        ) -> Tuple[np.ndarray, bool]:
        """ Choose input image from following sources with descending priority:
         - p.image_control: [Deprecated] Lagacy way to pass image to controlnet.
         - p.control_net_input_image: [Deprecated] Lagacy way to pass image to controlnet.
         - unit.image: 
           - ControlNet tab input image.
           - Input image from API call.
         - p.init_images: A1111 img2img tab input image.

        Returns:
            - The input image in ndarray form.
            - Whether input image is from A1111.
        """
        image_from_a1111 = False

        p_input_image = Script.get_remote_call(p, "control_net_input_image", None, idx)
        image = image_dict_from_any(unit.image)

        if batch_hijack.instance.is_batch and getattr(p, "image_control", None) is not None:
            logger.warning("Warn: Using legacy field 'p.image_control'.")
            input_image = HWC3(np.asarray(p.image_control))
        elif p_input_image is not None:
            logger.warning("Warn: Using legacy field 'p.controlnet_input_image'")
            if isinstance(p_input_image, dict) and "mask" in p_input_image and "image" in p_input_image:
                color = HWC3(np.asarray(p_input_image['image']))
                alpha = np.asarray(p_input_image['mask'])[..., None]
                input_image = np.concatenate([color, alpha], axis=2)
            else:
                input_image = HWC3(np.asarray(p_input_image))
        elif image is not None:
            while len(image['mask'].shape) < 3:
                image['mask'] = image['mask'][..., np.newaxis]

            # Need to check the image for API compatibility
            if isinstance(image['image'], str):
                from modules.api.api import decode_base64_to_image
                input_image = HWC3(np.asarray(decode_base64_to_image(image['image'])))
            else:
                input_image = HWC3(image['image'])

            have_mask = 'mask' in image and not (
                (image['mask'][:, :, 0] <= 5).all() or 
                (image['mask'][:, :, 0] >= 250).all()
            )

            if 'inpaint' in unit.module:
                logger.info("using inpaint as input")
                color = HWC3(image['image'])
                if have_mask:
                    alpha = image['mask'][:, :, 0:1]
                else:
                    alpha = np.zeros_like(color)[:, :, 0:1]
                input_image = np.concatenate([color, alpha], axis=2)
            else:
                if have_mask and not shared.opts.data.get("controlnet_ignore_noninpaint_mask", False):
                    logger.info("using mask as input")
                    input_image = HWC3(image['mask'][:, :, 0])
                    unit.module = 'none'  # Always use black bg and white line
        else:
            # use img2img init_image as default
            input_image = getattr(p, "init_images", [None])[0]
            if input_image is None:
                if batch_hijack.instance.is_batch:
                    shared.state.interrupted = True
                raise ValueError('controlnet is enabled but no input image is given')

            input_image = HWC3(np.asarray(input_image))
            image_from_a1111 = True
        
        assert isinstance(input_image, np.ndarray)
        return input_image, image_from_a1111
    
    @staticmethod
    def bound_check_params(unit: external_code.ControlNetUnit) -> None:
        """
        Checks and corrects negative parameters in ControlNetUnit 'unit'.
        Parameters 'processor_res', 'threshold_a', 'threshold_b' are reset to 
        their default values if negative.
        
        Args:
            unit (external_code.ControlNetUnit): The ControlNetUnit instance to check.
        """
        cfg = preprocessor_sliders_config.get(
            global_state.get_module_basename(unit.module), [])
        defaults = {
            param: cfg_default['value']
            for param, cfg_default in zip(
                ("processor_res", 'threshold_a', 'threshold_b'), cfg)
            if cfg_default is not None
        }
        for param, default_value in defaults.items():
            value = getattr(unit, param)
            if value < 0:
                setattr(unit, param, default_value)
                logger.warning(f'[{unit.module}.{param}] Invalid value({value}), using default value {default_value}.')

    def controlnet_main_entry(self, p):
        sd_ldm = p.sd_model
        unet = sd_ldm.model.diffusion_model
        self.noise_modifier = None

        setattr(p, 'controlnet_control_loras', [])

        if self.latest_network is not None:
            # always restore (~0.05s)
            self.latest_network.restore()

        # always clear (~0.05s)
        clear_all_secondary_control_models()

        if not batch_hijack.instance.is_batch:
            self.enabled_units = Script.get_enabled_units(p)

        if len(self.enabled_units) == 0:
           self.latest_network = None
           return

        detected_maps = []
        forward_params = []
        post_processors = []

        # cache stuff
        if self.latest_model_hash != p.sd_model.sd_model_hash:
            Script.clear_control_model_cache()

        for idx, unit in enumerate(self.enabled_units):
            unit.module = global_state.get_module_basename(unit.module)

        # unload unused preproc
        module_list = [unit.module for unit in self.enabled_units]
        for key in self.unloadable:
            if key not in module_list:
                self.unloadable.get(key, lambda:None)()

        self.latest_model_hash = p.sd_model.sd_model_hash
        for idx, unit in enumerate(self.enabled_units):
            Script.bound_check_params(unit)

            resize_mode = external_code.resize_mode_from_value(unit.resize_mode)
            control_mode = external_code.control_mode_from_value(unit.control_mode)

            if unit.module in model_free_preprocessors:
                model_net = None
            else:
                model_net = Script.load_control_model(p, unet, unit.model)
                model_net.reset()

                if getattr(model_net, 'is_control_lora', False):
                    control_lora = model_net.control_model
                    bind_control_lora(unet, control_lora)
                    p.controlnet_control_loras.append(control_lora)

            input_image, image_from_a1111 = Script.choose_input_image(p, unit, idx)
            if image_from_a1111:
                a1111_i2i_resize_mode = getattr(p, "resize_mode", None)
                if a1111_i2i_resize_mode is not None:
                    resize_mode = external_code.resize_mode_from_value(a1111_i2i_resize_mode)
            
            a1111_mask_image : Optional[Image.Image] = getattr(p, "image_mask", None)
            if 'inpaint' in unit.module and not image_has_mask(input_image) and a1111_mask_image is not None:
                a1111_mask = np.array(prepare_mask(a1111_mask_image, p))
                if a1111_mask.ndim == 2:
                    if a1111_mask.shape[0] == input_image.shape[0]:
                        if a1111_mask.shape[1] == input_image.shape[1]:
                            input_image = np.concatenate([input_image[:, :, 0:3], a1111_mask[:, :, None]], axis=2)
                            a1111_i2i_resize_mode = getattr(p, "resize_mode", None)
                            if a1111_i2i_resize_mode is not None:
                                resize_mode = external_code.resize_mode_from_value(a1111_i2i_resize_mode)

            if 'reference' not in unit.module and issubclass(type(p), StableDiffusionProcessingImg2Img) \
                    and p.inpaint_full_res and a1111_mask_image is not None:
                logger.debug("A1111 inpaint mask START")
                input_image = [input_image[:, :, i] for i in range(input_image.shape[2])]
                input_image = [Image.fromarray(x) for x in input_image]

                mask = prepare_mask(a1111_mask_image, p)

                crop_region = masking.get_crop_region(np.array(mask), p.inpaint_full_res_padding)
                crop_region = masking.expand_crop_region(crop_region, p.width, p.height, mask.width, mask.height)

                input_image = [
                    images.resize_image(resize_mode.int_value(), i, mask.width, mask.height) 
                    for i in input_image
                ]

                input_image = [x.crop(crop_region) for x in input_image]
                input_image = [
                    images.resize_image(external_code.ResizeMode.OUTER_FIT.int_value(), x, p.width, p.height) 
                    for x in input_image
                ]

                input_image = [np.asarray(x)[:, :, 0] for x in input_image]
                input_image = np.stack(input_image, axis=2)
                logger.debug("A1111 inpaint mask END")

            if 'inpaint_only' == unit.module and issubclass(type(p), StableDiffusionProcessingImg2Img) and p.image_mask is not None:
                logger.warning('A1111 inpaint and ControlNet inpaint duplicated. ControlNet support enabled.')
                unit.module = 'inpaint'

            # safe numpy
            logger.debug("Safe numpy convertion START")
            input_image = np.ascontiguousarray(input_image.copy()).copy()
            logger.debug("Safe numpy convertion END")

            logger.info(f"Loading preprocessor: {unit.module}")
            preprocessor = self.preprocessor[unit.module]

            high_res_fix = isinstance(p, StableDiffusionProcessingTxt2Img) and getattr(p, 'enable_hr', False)

            h = (p.height // 8) * 8
            w = (p.width // 8) * 8

            if high_res_fix:
                if p.hr_resize_x == 0 and p.hr_resize_y == 0:
                    hr_y = int(p.height * p.hr_scale)
                    hr_x = int(p.width * p.hr_scale)
                else:
                    hr_y, hr_x = p.hr_resize_y, p.hr_resize_x
                hr_y = (hr_y // 8) * 8
                hr_x = (hr_x // 8) * 8
            else:
                hr_y = h
                hr_x = w

            if unit.module == 'inpaint_only+lama' and resize_mode == external_code.ResizeMode.OUTER_FIT:
                # inpaint_only+lama is special and required outpaint fix
                _, input_image = Script.detectmap_proc(input_image, unit.module, resize_mode, hr_y, hr_x)

            control_model_type = ControlModelType.ControlNet
            global_average_pooling = False

            if 'reference' in unit.module:
                control_model_type = ControlModelType.AttentionInjection
            elif 'revision' in unit.module:
                control_model_type = ControlModelType.ReVision
            elif hasattr(model_net, 'control_model') and (isinstance(model_net.control_model, Adapter) or isinstance(model_net.control_model, Adapter_light)):
                control_model_type = ControlModelType.T2I_Adapter
            elif hasattr(model_net, 'control_model') and isinstance(model_net.control_model, StyleAdapter):
                control_model_type = ControlModelType.T2I_StyleAdapter
            elif isinstance(model_net, PlugableIPAdapter):
                control_model_type = ControlModelType.IPAdapter
            elif isinstance(model_net, PlugableControlLLLite):
                control_model_type = ControlModelType.Controlllite

            if control_model_type is ControlModelType.ControlNet:
                global_average_pooling = model_net.control_model.global_average_pooling

            preprocessor_resolution = unit.processor_res
            if unit.pixel_perfect:
                preprocessor_resolution = external_code.pixel_perfect_resolution(
                    input_image,
                    target_H=h,
                    target_W=w,
                    resize_mode=resize_mode
                )

            logger.info(f'preprocessor resolution = {preprocessor_resolution}')
            # Preprocessor result may depend on numpy random operations, use the
            # random seed in `StableDiffusionProcessing` to make the 
            # preprocessor result reproducable.
            # Currently following preprocessors use numpy random:
            # - shuffle
            seed = set_numpy_seed(p)
            logger.debug(f"Use numpy seed {seed}.")
            detected_map, is_image = preprocessor(
                input_image, 
                res=preprocessor_resolution, 
                thr_a=unit.threshold_a,
                thr_b=unit.threshold_b,
            )

            if high_res_fix:
                if is_image:
                    hr_control, hr_detected_map = Script.detectmap_proc(detected_map, unit.module, resize_mode, hr_y, hr_x)
                    detected_maps.append((hr_detected_map, unit.module))
                else:
                    hr_control = detected_map
            else:
                hr_control = None

            if is_image:
                control, detected_map = Script.detectmap_proc(detected_map, unit.module, resize_mode, h, w)
                detected_maps.append((detected_map, unit.module))
            else:
                control = detected_map
                detected_maps.append((input_image, unit.module))

            if control_model_type == ControlModelType.T2I_StyleAdapter:
                control = control['last_hidden_state']

            if control_model_type == ControlModelType.ReVision:
                control = control['image_embeds']

            preprocessor_dict = dict(
                name=unit.module,
                preprocessor_resolution=preprocessor_resolution,
                threshold_a=unit.threshold_a,
                threshold_b=unit.threshold_b
            )

            forward_param = ControlParams(
                control_model=model_net,
                preprocessor=preprocessor_dict,
                hint_cond=control,
                weight=unit.weight,
                guidance_stopped=False,
                start_guidance_percent=unit.guidance_start,
                stop_guidance_percent=unit.guidance_end,
                advanced_weighting=None,
                control_model_type=control_model_type,
                global_average_pooling=global_average_pooling,
                hr_hint_cond=hr_control,
                soft_injection=control_mode != external_code.ControlMode.BALANCED,
                cfg_injection=control_mode == external_code.ControlMode.CONTROL,
            )
            forward_params.append(forward_param)

            if 'inpaint_only' in unit.module:
                final_inpaint_feed = hr_control if hr_control is not None else control
                final_inpaint_feed = final_inpaint_feed.detach().cpu().numpy()
                final_inpaint_feed = np.ascontiguousarray(final_inpaint_feed).copy()
                final_inpaint_mask = final_inpaint_feed[0, 3, :, :].astype(np.float32)
                final_inpaint_raw = final_inpaint_feed[0, :3].astype(np.float32)
                sigma = shared.opts.data.get("control_net_inpaint_blur_sigma", 7)
                final_inpaint_mask = cv2.dilate(final_inpaint_mask, np.ones((sigma, sigma), dtype=np.uint8))
                final_inpaint_mask = cv2.blur(final_inpaint_mask, (sigma, sigma))[None]
                _, Hmask, Wmask = final_inpaint_mask.shape
                final_inpaint_raw = torch.from_numpy(np.ascontiguousarray(final_inpaint_raw).copy())
                final_inpaint_mask = torch.from_numpy(np.ascontiguousarray(final_inpaint_mask).copy())

                def inpaint_only_post_processing(x):
                    _, H, W = x.shape
                    if Hmask != H or Wmask != W:
                        logger.error('Error: ControlNet find post-processing resolution mismatch. This could be related to other extensions hacked processing.')
                        return x
                    r = final_inpaint_raw.to(x.dtype).to(x.device)
                    m = final_inpaint_mask.to(x.dtype).to(x.device)
                    y = m * x.clip(0, 1) + (1 - m) * r
                    y = y.clip(0, 1)
                    return y

                post_processors.append(inpaint_only_post_processing)

            if 'recolor' in unit.module:
                final_feed = hr_control if hr_control is not None else control
                final_feed = final_feed.detach().cpu().numpy()
                final_feed = np.ascontiguousarray(final_feed).copy()
                final_feed = final_feed[0, 0, :, :].astype(np.float32)
                final_feed = (final_feed * 255).clip(0, 255).astype(np.uint8)
                Hfeed, Wfeed = final_feed.shape

                if 'luminance' in unit.module:

                    def recolor_luminance_post_processing(x):
                        C, H, W = x.shape
                        if Hfeed != H or Wfeed != W or C != 3:
                            logger.error('Error: ControlNet find post-processing resolution mismatch. This could be related to other extensions hacked processing.')
                            return x
                        h = x.detach().cpu().numpy().transpose((1, 2, 0))
                        h = (h * 255).clip(0, 255).astype(np.uint8)
                        h = cv2.cvtColor(h, cv2.COLOR_RGB2LAB)
                        h[:, :, 0] = final_feed
                        h = cv2.cvtColor(h, cv2.COLOR_LAB2RGB)
                        h = (h.astype(np.float32) / 255.0).transpose((2, 0, 1))
                        y = torch.from_numpy(h).clip(0, 1).to(x)
                        return y

                    post_processors.append(recolor_luminance_post_processing)

                if 'intensity' in unit.module:

                    def recolor_intensity_post_processing(x):
                        C, H, W = x.shape
                        if Hfeed != H or Wfeed != W or C != 3:
                            logger.error('Error: ControlNet find post-processing resolution mismatch. This could be related to other extensions hacked processing.')
                            return x
                        h = x.detach().cpu().numpy().transpose((1, 2, 0))
                        h = (h * 255).clip(0, 255).astype(np.uint8)
                        h = cv2.cvtColor(h, cv2.COLOR_RGB2HSV)
                        h[:, :, 2] = final_feed
                        h = cv2.cvtColor(h, cv2.COLOR_HSV2RGB)
                        h = (h.astype(np.float32) / 255.0).transpose((2, 0, 1))
                        y = torch.from_numpy(h).clip(0, 1).to(x)
                        return y

                    post_processors.append(recolor_intensity_post_processing)

            if '+lama' in unit.module:
                forward_param.used_hint_cond_latent = hook.UnetHook.call_vae_using_process(p, control)
                self.noise_modifier = forward_param.used_hint_cond_latent

            del model_net

        is_low_vram = any(unit.low_vram for unit in self.enabled_units)

        self.latest_network = UnetHook(lowvram=is_low_vram)
        self.latest_network.hook(model=unet, sd_ldm=sd_ldm, control_params=forward_params, process=p)

        for param in forward_params:
            if param.control_model_type == ControlModelType.IPAdapter:
                param.control_model.hook(
                    model=unet,
                    clip_vision_output=param.hint_cond,
                    weight=param.weight,
                    dtype=torch.float32,
                    start=param.start_guidance_percent,
                    end=param.stop_guidance_percent
                )
            if param.control_model_type == ControlModelType.Controlllite:
                param.control_model.hook(
                    model=unet,
                    cond=param.hint_cond,
                    weight=param.weight,
                    start=param.start_guidance_percent,
                    end=param.stop_guidance_percent
                )

        self.detected_map = detected_maps
        self.post_processors = post_processors

    def controlnet_hack(self, p):
        t = time.time()
        self.controlnet_main_entry(p)
        if len(self.enabled_units) > 0:
            logger.info(f'ControlNet Hooked - Time = {time.time() - t}')
        return

    @staticmethod
    def process_has_sdxl_refiner(p):
        return getattr(p, 'refiner_checkpoint', None) is not None

    def process(self, p, *args, **kwargs):
        if not self.process_has_sdxl_refiner(p):
            self.controlnet_hack(p)
        return

    def before_process_batch(self, p, *args, **kwargs):
        if self.noise_modifier is not None:
            p.rng = HackedImageRNG(rng=p.rng,
                                   noise_modifier=self.noise_modifier,
                                   sd_model=p.sd_model)
        self.noise_modifier = None
        if self.process_has_sdxl_refiner(p):
            self.controlnet_hack(p)
        return

    def postprocess_batch(self, p, *args, **kwargs):
        images = kwargs.get('images', [])
        for post_processor in self.post_processors:
            for i in range(len(images)):
                images[i] = post_processor(images[i])
        return

    def postprocess(self, p, processed, *args):
        clear_all_secondary_control_models()

        self.noise_modifier = None

        for control_lora in getattr(p, 'controlnet_control_loras', []):
            unbind_control_lora(control_lora)
        p.controlnet_control_loras = []

        self.post_processors = []
        setattr(p, 'controlnet_vae_cache', None)

        processor_params_flag = (', '.join(getattr(processed, 'extra_generation_params', []))).lower()
        self.post_processors = []

        if not batch_hijack.instance.is_batch:
            self.enabled_units.clear()

        if shared.opts.data.get("control_net_detectmap_autosaving", False) and self.latest_network is not None:
            for detect_map, module in self.detected_map:
                detectmap_dir = os.path.join(shared.opts.data.get("control_net_detectedmap_dir", ""), module)
                if not os.path.isabs(detectmap_dir):
                    detectmap_dir = os.path.join(p.outpath_samples, detectmap_dir)
                if module != "none":
                    os.makedirs(detectmap_dir, exist_ok=True)
                    img = Image.fromarray(np.ascontiguousarray(detect_map.clip(0, 255).astype(np.uint8)).copy())
                    save_image(img, detectmap_dir, module)

        if self.latest_network is None:
            return

        if not batch_hijack.instance.is_batch:
            if not shared.opts.data.get("control_net_no_detectmap", False):
                if 'sd upscale' not in processor_params_flag:
                    if self.detected_map is not None:
                        for detect_map, module in self.detected_map:
                            if detect_map is None:
                                continue
                            detect_map = np.ascontiguousarray(detect_map.copy()).copy()
                            detect_map = external_code.visualize_inpaint_mask(detect_map)
                            processed.images.extend([
                                Image.fromarray(
                                    detect_map.clip(0, 255).astype(np.uint8)
                                )
                            ])

        self.input_image = None
        self.latest_network.restore()
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
                    logger.warning(f'Warning: No loopback image found for controlnet unit {unit_i}. Using control map from last batch iteration instead')

    def batch_tab_postprocess(self, p, *args, **kwargs):
        self.enabled_units.clear()
        self.input_image = None
        if self.latest_network is None: return

        self.latest_network.restore()
        self.latest_network = None
        self.detected_map.clear()


def on_ui_settings():
    section = ('control_net', "ControlNet")
    shared.opts.add_option("control_net_detectedmap_dir", shared.OptionInfo(
        global_state.default_detectedmap_dir, "Directory for detected maps auto saving", section=section))
    shared.opts.add_option("control_net_models_path", shared.OptionInfo(
        "", "Extra path to scan for ControlNet models (e.g. training output directory)", section=section))
    shared.opts.add_option("control_net_modules_path", shared.OptionInfo(
        "", "Path to directory containing annotator model directories (requires restart, overrides corresponding command line flag)", section=section))
    shared.opts.add_option("control_net_unit_count", shared.OptionInfo(
        3, "Multi-ControlNet: ControlNet unit number (requires restart)", gr.Slider, {"minimum": 1, "maximum": 10, "step": 1}, section=section))
    shared.opts.add_option("control_net_model_cache_size", shared.OptionInfo(
        1, "Model cache size (requires restart)", gr.Slider, {"minimum": 1, "maximum": 5, "step": 1}, section=section))
    shared.opts.add_option("control_net_inpaint_blur_sigma", shared.OptionInfo(
        7, "ControlNet inpainting Gaussian blur sigma", gr.Slider, {"minimum": 0, "maximum": 64, "step": 1}, section=section))
    shared.opts.add_option("control_net_no_high_res_fix", shared.OptionInfo(
        False, "Do not apply ControlNet during highres fix", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("control_net_no_detectmap", shared.OptionInfo(
        False, "Do not append detectmap to output", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("control_net_detectmap_autosaving", shared.OptionInfo(
        False, "Allow detectmap auto saving", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("control_net_allow_script_control", shared.OptionInfo(
        False, "Allow other script to control this extension", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("control_net_sync_field_args", shared.OptionInfo(
        True, "Paste ControlNet parameters in infotext", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("controlnet_show_batch_images_in_ui", shared.OptionInfo(
        False, "Show batch images in gradio gallery output", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("controlnet_increment_seed_during_batch", shared.OptionInfo(
        False, "Increment seed after each controlnet batch iteration", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("controlnet_disable_control_type", shared.OptionInfo(
        False, "Disable control type selection", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("controlnet_disable_openpose_edit", shared.OptionInfo(
        False, "Disable openpose edit", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("controlnet_ignore_noninpaint_mask", shared.OptionInfo(
        False, "Ignore mask on ControlNet input image if control type is not inpaint", 
        gr.Checkbox, {"interactive": True}, section=section))


batch_hijack.instance.do_hijack()
script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_infotext_pasted(Infotext.on_infotext_pasted)
script_callbacks.on_after_component(ControlNetUiGroup.on_after_component)
