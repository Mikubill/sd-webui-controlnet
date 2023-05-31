import gc
import os
from collections import OrderedDict
from copy import copy
from typing import Union, Dict, Optional, List
import importlib
import modules.scripts as scripts
from modules import shared, devices, script_callbacks, processing, masking, images
import gradio as gr

from einops import rearrange
from scripts import global_state, hook, external_code, processor, batch_hijack, controlnet_version
importlib.reload(processor)
importlib.reload(global_state)
importlib.reload(hook)
importlib.reload(external_code)
importlib.reload(batch_hijack)
from scripts.cldm import PlugableControlModel
from scripts.processor import *
from scripts.adapter import PlugableAdapter
from scripts.utils import load_state_dict
from scripts.hook import ControlParams, UnetHook, ControlModelType
from modules.processing import StableDiffusionProcessingImg2Img, StableDiffusionProcessingTxt2Img
from modules.images import save_image
from modules.ui_components import FormRow

import cv2
import numpy as np
import torch
import base64

from pathlib import Path
from PIL import Image, ImageFilter, ImageOps
from scripts.lvminthin import lvmin_thin, nake_nms
from scripts.processor import preprocessor_sliders_config, flag_preprocessor_resolution, model_free_preprocessors, preprocessor_filters

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
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPM
    svgsupport = True
except ImportError:
    pass

# Note: Change symbol hints mapping in `javascript/hints.js` when you change the symbol values.
refresh_symbol = '\U0001f504'       # 🔄
switch_values_symbol = '\U000021C5' # ⇅
camera_symbol = '\U0001F4F7'        # 📷
reverse_symbol = '\U000021C4'       # ⇄
tossup_symbol = '\u2934'
trigger_symbol = '\U0001F4A5'  # 💥
open_symbol = '\U0001F4DD'  # 📝

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

txt2img_submit_button = None
img2img_submit_button = None


class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", elem_classes=['cnet-toolbutton'], **kwargs)

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


def update_json_download_link(json_string: str, file_name: str) -> Dict:
    base64_encoded_json = base64.b64encode(json_string.encode('utf-8')).decode('utf-8')
    data_uri = f'data:application/json;base64,{base64_encoded_json}'
    style = """ 
    position: absolute;
    right: var(--size-2);
    bottom: calc(var(--size-2) * 4);
    font-size: x-small;
    font-weight: bold;
    padding: 2px;

    box-shadow: var(--shadow-drop);
    border: 1px solid var(--button-secondary-border-color);
    border-radius: var(--radius-sm);
    background: var(--background-fill-primary);
    height: var(--size-5);
    color: var(--block-label-text-color);
    """
    hint = "Download the pose as .json file"
    html = f"""<a href='{data_uri}' download='{file_name}' style="{style}" title="{hint}">
                Json</a>"""
    return gr.update(
        value=html,
        visible=(json_string != '')
    )


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
        self.preprocessor = global_state.cache_preprocessors(global_state.cn_preprocessor_modules)
        self.unloadable = global_state.cn_preprocessor_unloadable
        self.input_image = None
        self.latest_model_hash = ""
        self.txt2img_w_slider = gr.Slider()
        self.txt2img_h_slider = gr.Slider()
        self.img2img_w_slider = gr.Slider()
        self.img2img_h_slider = gr.Slider()
        self.enabled_units = []
        self.detected_map = []
        self.post_processors = []
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
        if module is None:
            module = 'none'

        return global_state.reverse_preprocessor_aliases.get(module, module)

    def get_threshold_block(self, proc):
        pass

    def get_default_ui_unit(self, is_ui=True):
        cls = UiControlNetUnit if is_ui else external_code.ControlNetUnit
        return cls(
            enabled=False,
            module="none",
            model="None"
        )

    def uigroup(self, tabname, is_img2img, elem_id_tabname):
        infotext_fields = []
        default_unit = self.get_default_ui_unit()
        with gr.Tabs():
            with gr.Tab(label='Single Image') as upload_tab:
                with gr.Row().style(equal_height=True):
                    input_image = gr.Image(source='upload', brush_radius=20, mirror_webcam=False, type='numpy', tool='sketch', elem_id=f'{elem_id_tabname}_{tabname}_input_image')
                    # Gradio's magic number. Only 242 works.
                    with gr.Group(visible=False) as generated_image_group:
                        generated_image = gr.Image(label="Preprocessor Preview", elem_id=f'{elem_id_tabname}_{tabname}_generated_image').style(height=242)
                        download_pose_link = gr.HTML(value='', visible=False)
                        preview_close_button_style = """ 
                            position: absolute;
                            right: var(--size-2);
                            bottom: var(--size-2);
                            font-size: x-small;
                            font-weight: bold;
                            padding: 2px;
                            cursor: pointer;

                            box-shadow: var(--shadow-drop);
                            border: 1px solid var(--button-secondary-border-color);
                            border-radius: var(--radius-sm);
                            background: var(--background-fill-primary);
                            height: var(--size-5);
                            color: var(--block-label-text-color);
                            """
                        preview_check_elem_id = f'{elem_id_tabname}_{tabname}_controlnet_preprocessor_preview_checkbox'
                        preview_close_button_js = f"document.querySelector(\'#{preview_check_elem_id} input[type=\\\'checkbox\\\']\').click();"
                        gr.HTML(value=f'''<a style="{preview_close_button_style}" title="Close Preview" onclick="{preview_close_button_js}">Close</a>''', visible=True)

            with gr.Tab(label='Batch') as batch_tab:
                batch_image_dir = gr.Textbox(label='Input Directory', placeholder='Leave empty to use img2img batch controlnet input directory', elem_id=f'{elem_id_tabname}_{tabname}_batch_image_dir')

        with gr.Accordion(label='Open New Canvas', visible=False) as create_canvas:
            canvas_width = gr.Slider(label="New Canvas Width", minimum=256, maximum=1024, value=512, step=64, elem_id=f'{elem_id_tabname}_{tabname}_controlnet_canvas_width')
            canvas_height = gr.Slider(label="New Canvas Height", minimum=256, maximum=1024, value=512, step=64, elem_id=f'{elem_id_tabname}_{tabname}_controlnet_canvas_height')
            with gr.Row():
                canvas_create_button = gr.Button(value="Create New Canvas", elem_id=f'{elem_id_tabname}_{tabname}_controlnet_canvas_create_button')
                canvas_cancel_button = gr.Button(value="Cancel", elem_id=f'{elem_id_tabname}_{tabname}_controlnet_canvas_cancel_button')

        with gr.Row(elem_classes="controlnet_image_controls"):
            gr.HTML(value='<p>Set the preprocessor to [invert] If your image has white background and black lines.</p>', elem_classes="controlnet_invert_warning")
            open_new_canvas_button = ToolButton(value=open_symbol, elem_id=f'{elem_id_tabname}_{tabname}_controlnet_open_new_canvas_button')
            webcam_enable = ToolButton(value=camera_symbol, elem_id=f'{elem_id_tabname}_{tabname}_controlnet_webcam_enable')
            webcam_mirror = ToolButton(value=reverse_symbol, elem_id=f'{elem_id_tabname}_{tabname}_controlnet_webcam_mirror')
            send_dimen_button = ToolButton(value=tossup_symbol, elem_id=f'{elem_id_tabname}_{tabname}_controlnet_send_dimen_button')

        open_new_canvas_button.click(lambda: gr.Accordion.update(visible=True), inputs=None, outputs=create_canvas)
        canvas_cancel_button.click(lambda: gr.Accordion.update(visible=False), inputs=None, outputs=create_canvas)

        with FormRow(elem_classes=["checkboxes-row","controlnet_main_options"], variant="compact"):
            enabled = gr.Checkbox(label='Enable', value=default_unit.enabled, elem_id=f'{elem_id_tabname}_{tabname}_controlnet_enable_checkbox')
            lowvram = gr.Checkbox(label='Low VRAM', value=default_unit.low_vram, elem_id=f'{elem_id_tabname}_{tabname}_controlnet_low_vram_checkbox')
            pixel_perfect = gr.Checkbox(label='Pixel Perfect', value=default_unit.pixel_perfect, elem_id=f'{elem_id_tabname}_{tabname}_controlnet_pixel_perfect_checkbox')
            preprocessor_preview = gr.Checkbox(label='Allow Preview', value=False, elem_id=preview_check_elem_id)

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
            global_state.update_cn_models()

            dd = inputs[0]
            selected = dd if dd in global_state.cn_models else "None"
            return gr.Dropdown.update(value=selected, choices=list(global_state.cn_models.keys()))

        if shared.opts.data.get("controlnet_disable_control_type", False):
            type_filter = None
        else:
            with gr.Row(elem_classes="controlnet_control_type"):
                type_filter = gr.Radio(list(preprocessor_filters.keys()), label=f"Control Type", value='All', elem_id=f'{elem_id_tabname}_{tabname}_controlnet_type_filter_radio', elem_classes='controlnet_control_type_filter_group')

        with gr.Row(elem_classes="controlnet_preprocessor_model"):
            module = gr.Dropdown(global_state.ui_preprocessor_keys, label=f"Preprocessor", value=default_unit.module, elem_id=f'{elem_id_tabname}_{tabname}_controlnet_preprocessor_dropdown')
            trigger_preprocessor = ToolButton(value=trigger_symbol, visible=True, elem_id=f'{elem_id_tabname}_{tabname}_controlnet_trigger_preprocessor')
            model = gr.Dropdown(list(global_state.cn_models.keys()), label=f"Model", value=default_unit.model, elem_id=f'{elem_id_tabname}_{tabname}_controlnet_model_dropdown')
            refresh_models = ToolButton(value=refresh_symbol, elem_id=f'{elem_id_tabname}_{tabname}_controlnet_refresh_models')
            refresh_models.click(refresh_all_models, model, model)

        with gr.Row(elem_classes="controlnet_weight_steps"):
            weight = gr.Slider(label=f"Control Weight", value=default_unit.weight, minimum=0.0, maximum=2.0, step=.05, elem_id=f'{elem_id_tabname}_{tabname}_controlnet_control_weight_slider', elem_classes='controlnet_control_weight_slider')
            guidance_start = gr.Slider(label="Starting Control Step", value=default_unit.guidance_start, minimum=0.0, maximum=1.0, interactive=True, elem_id=f'{elem_id_tabname}_{tabname}_controlnet_start_control_step_slider', elem_classes='controlnet_start_control_step_slider')
            guidance_end = gr.Slider(label="Ending Control Step", value=default_unit.guidance_end, minimum=0.0, maximum=1.0, interactive=True, elem_id=f'{elem_id_tabname}_{tabname}_controlnet_ending_control_step_slider', elem_classes='controlnet_ending_control_step_slider')

        def build_sliders(module, pp):
            grs = []
            module = self.get_module_basename(module)
            if module not in preprocessor_sliders_config:
                grs += [
                    gr.update(label=flag_preprocessor_resolution, value=512, minimum=64, maximum=2048, step=1, visible=not pp, interactive=not pp),
                    gr.update(visible=False, interactive=False),
                    gr.update(visible=False, interactive=False),
                    gr.update(visible=True)
                ]
            else:
                for slider_config in preprocessor_sliders_config[module]:
                    if isinstance(slider_config, dict):
                        visible = True
                        if slider_config['name'] == flag_preprocessor_resolution:
                            visible = not pp
                        grs.append(gr.update(
                            label=slider_config['name'],
                            value=slider_config['value'],
                            minimum=slider_config['min'],
                            maximum=slider_config['max'],
                            step=slider_config['step'] if 'step' in slider_config else 1,
                            visible=visible,
                            interactive=visible))
                    else:
                        grs.append(gr.update(visible=False, interactive=False))
                while len(grs) < 3:
                    grs.append(gr.update(visible=False, interactive=False))
                grs.append(gr.update(visible=True))
            if module in model_free_preprocessors:
                grs += [gr.update(visible=False, value='None'), gr.update(visible=False)]
            else:
                grs += [gr.update(visible=True), gr.update(visible=True)]
            return grs

        # advanced options
        with gr.Column(visible=False) as advanced:
            processor_res = gr.Slider(label="Preprocessor resolution", value=default_unit.processor_res, minimum=64, maximum=2048, visible=False, interactive=False, elem_id=f'{elem_id_tabname}_{tabname}_controlnet_preprocessor_resolution_slider')
            threshold_a = gr.Slider(label="Threshold A", value=default_unit.threshold_a, minimum=64, maximum=1024, visible=False, interactive=False, elem_id=f'{elem_id_tabname}_{tabname}_controlnet_threshold_A_slider')
            threshold_b = gr.Slider(label="Threshold B", value=default_unit.threshold_b, minimum=64, maximum=1024, visible=False, interactive=False, elem_id=f'{elem_id_tabname}_{tabname}_controlnet_threshold_B_slider')

        if gradio_compat:
            module.change(build_sliders, inputs=[module, pixel_perfect], outputs=[processor_res, threshold_a, threshold_b, advanced, model, refresh_models])
            pixel_perfect.change(build_sliders, inputs=[module, pixel_perfect], outputs=[processor_res, threshold_a, threshold_b, advanced, model, refresh_models])

            if type_filter is not None:
                def filter_selected(k, pp):
                    default_option = preprocessor_filters[k]
                    pattern = k.lower()
                    preprocessor_list = global_state.ui_preprocessor_keys
                    model_list = list(global_state.cn_models.keys())
                    if pattern == 'all':
                        return [gr.Dropdown.update(value='none', choices=preprocessor_list),
                                gr.Dropdown.update(value='None', choices=model_list)] + build_sliders('none', pp)
                    filtered_preprocessor_list = [x for x in preprocessor_list if pattern in x.lower() or x.lower() == 'none']
                    if pattern in ['canny', 'lineart', 'scribble', 'mlsd']:
                        filtered_preprocessor_list += [x for x in preprocessor_list if 'invert' in x.lower()]
                    filtered_model_list = [x for x in model_list if pattern in x.lower() or x.lower() == 'none']
                    if default_option not in filtered_preprocessor_list:
                        default_option = filtered_preprocessor_list[0]
                    if len(filtered_model_list) == 1:
                        default_model = 'None'
                        filtered_model_list = model_list
                    else:
                        default_model = filtered_model_list[1]
                        for x in filtered_model_list:
                            if '11' in x.split('[')[0]:
                                default_model = x
                                break
                    return [gr.Dropdown.update(value=default_option, choices=filtered_preprocessor_list),
                            gr.Dropdown.update(value=default_model, choices=filtered_model_list)] + build_sliders(default_option, pp)

                type_filter.change(filter_selected, inputs=[type_filter, pixel_perfect], outputs=[module, model, processor_res, threshold_a, threshold_b, advanced, model, refresh_models])

        # infotext_fields.extend((module, model, weight))

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

        def run_annotator(image, module, pres, pthr_a, pthr_b, t2i_w, t2i_h, pp, rm):
            if image is None:
                return gr.update(value=None, visible=True), gr.update(), gr.update()

            img = HWC3(image['image'])
            if not ((image['mask'][:, :, 0] == 0).all() or (image['mask'][:, :, 0] == 255).all()):
                img = HWC3(image['mask'][:, :, 0])

            if 'inpaint' in module:
                color = HWC3(image['image'])
                alpha = image['mask'][:, :, 0:1]
                img = np.concatenate([color, alpha], axis=2)

            module = self.get_module_basename(module)
            preprocessor = self.preprocessor[module]

            if pp:
                raw_H, raw_W, _ = img.shape
                target_H, target_W = t2i_h, t2i_w
                rm = str(rm)

                k0 = float(target_H) / float(raw_H)
                k1 = float(target_W) / float(raw_W)

                if rm == external_code.ResizeMode.OUTER_FIT.value:
                    estimation = min(k0, k1) * float(min(raw_H, raw_W))
                else:
                    estimation = max(k0, k1) * float(min(raw_H, raw_W))

                pres = int(np.round(estimation))
                print(f'Pixel Perfect Mode Enabled In Preview.')
                print(f'resize_mode = {rm}')
                print(f'raw_H = {raw_H}')
                print(f'raw_W = {raw_W}')
                print(f'target_H = {target_H}')
                print(f'target_W = {target_W}')
                print(f'estimation = {estimation}')

            class JsonAcceptor:
                def __init__(self) -> None:
                    self.value = ''

                def accept(self, json_string: str) -> None:
                    self.value = json_string
            json_acceptor = JsonAcceptor()

            print(f'Preview Resolution = {pres}')

            def is_openpose(module: str):
                return 'openpose' in module
            
            # Only openpose preprocessor returns a JSON output, pass json_acceptor
            # only when a JSON output is expected. This will make preprocessor cache
            # work for all other preprocessors other than openpose ones. JSON acceptor
            # instance are different every call, which means cache will never take 
            # effect.
            # TODO: Maybe we should let `preprocessor` return a Dict to alleviate this issue?
            # This requires changing all callsites though.
            result, is_image = preprocessor(img, res=pres, thr_a=pthr_a, thr_b=pthr_b,
                                            json_pose_callback=json_acceptor.accept if is_openpose(module) else None)

            if 'clip' in module:
                result = processor.clip_vision_visualization(result)
                is_image = True

            if is_image:
                if result.ndim == 3 and result.shape[2] == 4:
                    inpaint_mask = result[:, :, 3]
                    result = result[:, :, 0:3]
                    result[inpaint_mask > 127] = 0
                return (
                    # Update to `generated_image`
                    gr.update(value=result, visible=True, interactive=False),
                    # Update to `download_pose_link`
                    update_json_download_link(json_acceptor.value, 'pose.json'),
                    # preprocessor_preview
                    gr.update(value=True)
                )

            return (
                # Update to `generated_image`
                gr.update(value=None, visible=True),
                # Update to `download_pose_link`
                update_json_download_link(json_acceptor.value, 'pose.json'),
                # preprocessor_preview
                gr.update(value=True)
            )

        def shift_preview(is_on):
            return (
                # generated_image
                gr.update() if is_on else gr.update(value=None),
                # generated_image_group
                gr.update(visible=is_on),
                # download_pose_link
                gr.update() if is_on else gr.update(value=None),
            )

        preprocessor_preview.change(fn=shift_preview, inputs=[preprocessor_preview], 
                                    outputs=[generated_image, generated_image_group, download_pose_link])

        if is_img2img:
            send_dimen_button.click(fn=send_dimensions, inputs=[input_image], outputs=[self.img2img_w_slider, self.img2img_h_slider])
        else:
            send_dimen_button.click(fn=send_dimensions, inputs=[input_image], outputs=[self.txt2img_w_slider, self.txt2img_h_slider])

        control_mode = gr.Radio(choices=[e.value for e in external_code.ControlMode], value=default_unit.control_mode.value, label="Control Mode", elem_id=f'{elem_id_tabname}_{tabname}_controlnet_control_mode_radio', elem_classes='controlnet_control_mode_radio')

        resize_mode = gr.Radio(choices=[e.value for e in external_code.ResizeMode], value=default_unit.resize_mode.value, label="Resize Mode", elem_id=f'{elem_id_tabname}_{tabname}_controlnet_resize_mode_radio', elem_classes='controlnet_resize_mode_radio')

        loopback = gr.Checkbox(label='[Loopback] Automatically send generated images to this ControlNet unit', value=default_unit.loopback, elem_id=f'{elem_id_tabname}_{tabname}_controlnet_automatically_send_generated_images_checkbox', elem_classes='controlnet_loopback_checkbox')

        trigger_preprocessor.click(fn=run_annotator, inputs=[
            input_image, module, processor_res, threshold_a, threshold_b,
            self.img2img_w_slider if is_img2img else self.txt2img_w_slider,
            self.img2img_h_slider if is_img2img else self.txt2img_h_slider,
            pixel_perfect, resize_mode
        ], outputs=[generated_image, download_pose_link, preprocessor_preview])

        def fn_canvas(h, w):
            return np.zeros(shape=(h, w, 3), dtype=np.uint8) + 255, gr.Accordion.update(visible=False)

        canvas_create_button.click(fn=fn_canvas, inputs=[canvas_height, canvas_width], outputs=[input_image, create_canvas])

        input_mode = gr.State(batch_hijack.InputMode.SIMPLE)
        batch_image_dir_state = gr.State('')
        output_dir_state = gr.State('')
        unit_args = (input_mode, batch_image_dir_state, output_dir_state, loopback, enabled, module, model, weight, input_image, resize_mode, lowvram, processor_res, threshold_a, threshold_b, guidance_start, guidance_end, pixel_perfect, control_mode)
        self.register_modules(tabname, unit_args)

        input_image.orgpreprocess=input_image.preprocess
        input_image.preprocess=svgPreprocess

        unit = gr.State(default_unit)
        for comp in unit_args:
            event_subscribers = []
            if hasattr(comp, 'edit'):
                event_subscribers.append(comp.edit)
            elif hasattr(comp, 'click'):
                event_subscribers.append(comp.click)
            elif isinstance(comp, gr.Slider) and hasattr(comp, 'release'):
                event_subscribers.append(comp.release)
            elif hasattr(comp, 'change'):
                event_subscribers.append(comp.change)

            if hasattr(comp, 'clear'):
                event_subscribers.append(comp.clear)

            for event_subscriber in event_subscribers:
                event_subscriber(fn=UiControlNetUnit, inputs=list(unit_args), outputs=unit)

        # keep input_mode in sync
        def ui_controlnet_unit_for_input_mode(input_mode, *args):
            args = list(args)
            args[0] = input_mode
            return input_mode, UiControlNetUnit(*args)

        for input_tab in (
            (upload_tab, batch_hijack.InputMode.SIMPLE),
            (batch_tab, batch_hijack.InputMode.BATCH)
        ):
            input_tab[0].select(fn=ui_controlnet_unit_for_input_mode, inputs=[gr.State(input_tab[1])] + list(unit_args), outputs=[input_mode, unit])

        def determine_batch_dir(batch_dir, fallback_dir, fallback_fallback_dir):
            if batch_dir:
                return batch_dir
            elif fallback_dir:
                return fallback_dir
            else:
                return fallback_fallback_dir

        # keep batch_dir in sync with global batch input textboxes
        global img2img_batch_input_dir, img2img_batch_input_dir_callbacks
        def subscribe_for_batch_dir():
            global global_batch_input_dir, img2img_batch_input_dir
            batch_dirs = [batch_image_dir, global_batch_input_dir, img2img_batch_input_dir]
            for batch_dir_comp in batch_dirs:
                subscriber = getattr(batch_dir_comp, 'blur', None)
                if subscriber is None: continue
                subscriber(
                    fn=determine_batch_dir,
                    inputs=batch_dirs,
                    outputs=[batch_image_dir_state],
                    queue=False,
                )

        if img2img_batch_input_dir is None:
            # we are too soon, subscribe later when available
            img2img_batch_input_dir_callbacks.append(subscribe_for_batch_dir)
        else:
            subscribe_for_batch_dir()

        # keep output_dir in sync with global batch output textbox
        global img2img_batch_output_dir, img2img_batch_output_dir_callbacks
        def subscribe_for_output_dir():
            global img2img_batch_output_dir
            img2img_batch_output_dir.blur(
                fn=lambda a: a,
                inputs=[img2img_batch_output_dir],
                outputs=[output_dir_state],
                queue=False,
            )

        if img2img_batch_input_dir is None:
            # we are too soon, subscribe later when available
            img2img_batch_output_dir_callbacks.append(subscribe_for_output_dir)
        else:
            subscribe_for_output_dir()

        if is_img2img:
            img2img_submit_button.click(fn=UiControlNetUnit, inputs=list(unit_args), outputs=unit, queue=False)
        else:
            txt2img_submit_button.click(fn=UiControlNetUnit, inputs=list(unit_args), outputs=unit, queue=False)

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
            with gr.Accordion(f"ControlNet {controlnet_version.version_flag}", open = False, elem_id="controlnet"):
                if max_models > 1:
                    with gr.Tabs(elem_id=f"{elem_id_tabname}_tabs"):
                        for i in range(max_models):
                            with gr.Tab(f"ControlNet Unit {i}"):
                                controls += (self.uigroup(f"ControlNet-{i}", is_img2img, elem_id_tabname),)
                else:
                    with gr.Column():
                        controls += (self.uigroup(f"ControlNet", is_img2img, elem_id_tabname),)

        if shared.opts.data.get("control_net_sync_field_args", False):
            for _, field_name in self.infotext_fields:
                self.paste_field_names.append(field_name)

        return controls

    def register_modules(self, tabname, params):
        enabled, module, model, weight = params[4:8]
        guidance_start, guidance_end, pixel_perfect, control_mode = params[-4:]

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

        model_path = os.path.abspath(model_path)
        model_stem = Path(model_path).stem
        model_dir_name = os.path.dirname(model_path)

        possible_config_filenames = [
            os.path.join(model_dir_name, model_stem + ".yaml"),
            os.path.join(global_state.script_dir, 'models', model_stem + ".yaml"),
            os.path.join(model_dir_name, model_stem.replace('_fp16', '') + ".yaml"),
            os.path.join(global_state.script_dir, 'models', model_stem.replace('_fp16', '') + ".yaml"),
            os.path.join(model_dir_name, model_stem.replace('_diff', '') + ".yaml"),
            os.path.join(global_state.script_dir, 'models', model_stem.replace('_diff', '') + ".yaml"),
            os.path.join(model_dir_name, model_stem.replace('-fp16', '') + ".yaml"),
            os.path.join(global_state.script_dir, 'models', model_stem.replace('-fp16', '') + ".yaml"),
            os.path.join(model_dir_name, model_stem.replace('-diff', '') + ".yaml"),
            os.path.join(global_state.script_dir, 'models', model_stem.replace('-diff', '') + ".yaml")
        ]

        override_config = possible_config_filenames[0]

        for possible_config_filename in possible_config_filenames:
            if os.path.exists(possible_config_filename):
                override_config = possible_config_filename
                break

        if 'v11' in model_stem.lower() or 'shuffle' in model_stem.lower():
            assert os.path.exists(override_config), f'Error: The model config {override_config} is missing. ControlNet 1.1 must have configs.'

        if os.path.exists(override_config):
            network_config = override_config
        else:
            print(f'ERROR: ControlNet cannot find model config [{override_config}] \n'
                  f'ERROR: ControlNet will use a WRONG config [{network_config}] to load your model. \n'
                  f'ERROR: The WRONG config may not match your model. The generated results can be bad. \n'
                  f'ERROR: You are using a ControlNet model [{model_stem}] without correct YAML config file. \n'
                  f'ERROR: The performance of this model may be worse than your expectation. \n'
                  f'ERROR: If this model cannot get good results, the reason is that you do not have a YAML file for the model. \n'
                  f'Solution: Please download YAML file, or ask your model provider to provide [{override_config}] for you to download.\n'
                  f'Hint: You can take a look at [{os.path.join(global_state.script_dir, "models")}] to find many existing YAML files.\n')

        print(f"Loading config: {network_config}")
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
        unit.resize_mode = selector(p, "control_net_resize_mode", unit.resize_mode, idx)
        unit.low_vram = selector(p, "control_net_lowvram", unit.low_vram, idx)
        unit.processor_res = selector(p, "control_net_pres", unit.processor_res, idx)
        unit.threshold_a = selector(p, "control_net_pthr_a", unit.threshold_a, idx)
        unit.threshold_b = selector(p, "control_net_pthr_b", unit.threshold_b, idx)
        unit.guidance_start = selector(p, "control_net_guidance_start", unit.guidance_start, idx)
        unit.guidance_end = selector(p, "control_net_guidance_end", unit.guidance_end, idx)
        unit.guidance_end = selector(p, "control_net_guidance_strength", unit.guidance_end, idx)
        unit.control_mode = selector(p, "control_net_control_mode", unit.control_mode, idx)
        unit.pixel_perfect = selector(p, "control_net_pixel_perfect", unit.pixel_perfect, idx)

        return unit

    def detectmap_proc(self, detected_map, module, resize_mode, h, w):

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

            new_size_is_smaller = (size[0] * size[1]) < (x.shape[0] * x.shape[1])
            new_size_is_bigger = (size[0] * size[1]) > (x.shape[0] * x.shape[1])
            unique_color_count = np.unique(x.reshape(-1, x.shape[2]), axis=0).shape[0]
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
            if high_quality_background.shape[2] == 4:
                # Inpaint hijack
                inpaint_pad = 64
                if pad_h == 0:
                    high_quality_background[pad_h:pad_h + new_h, pad_w + inpaint_pad:pad_w + new_w - inpaint_pad, 3] = detected_map[:, inpaint_pad:-inpaint_pad, 3]
                else:
                    high_quality_background[pad_h + inpaint_pad:pad_h + new_h - inpaint_pad, pad_w:pad_w + new_w, 3] = detected_map[inpaint_pad:-inpaint_pad, :, 3]
                high_quality_background[pad_h:pad_h + new_h, pad_w:pad_w + new_w, 0:3] = detected_map[:, :, 0:3]
            else:
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
                log_key = f"ControlNet {idx}"
            else:
                log_key = "ControlNet"

            log_value = {
                "preprocessor": unit.module,
                "model": unit.model,
                "weight": unit.weight,
                "starting/ending": str((unit.guidance_start, unit.guidance_end)),
                "resize mode": str(unit.resize_mode),
                "pixel perfect": str(unit.pixel_perfect),
                "control mode": str(unit.control_mode),
                "preprocessor params": str((unit.processor_res, unit.threshold_a, unit.threshold_b)),
            }
            log_value = str(log_value).replace('\'', '').replace('{', '').replace('}', '')

            p.extra_generation_params.update({log_key: log_value})

        return enabled_units

    def process(self, p, *args):
        """
        This function is called before processing begins for AlwaysVisible scripts.
        You can modify the processing object (p) here, inject hooks, etc.
        args contains all values returned by components from ui()
        """

        sd_ldm = p.sd_model
        unet = sd_ldm.model.diffusion_model

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
        post_processors = []
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
            control_mode = external_code.control_mode_from_value(unit.control_mode)

            if unit.low_vram:
                hook_lowvram = True

            if unit.module in model_free_preprocessors:
                model_net = None
            else:
                model_net = self.load_control_model(p, unet, unit.model, unit.low_vram)
                model_net.reset()

            if batch_hijack.instance.is_batch and getattr(p, "image_control", None) is not None:
                input_image = HWC3(np.asarray(p.image_control))
            elif p_input_image is not None:
                if isinstance(p_input_image, dict) and "mask" in p_input_image and "image" in p_input_image:
                    color = HWC3(np.asarray(p_input_image['image']))
                    alpha = np.asarray(p_input_image['mask'])[..., None]
                    input_image = np.concatenate([color, alpha], axis=2)
                else:
                    input_image = HWC3(np.asarray(p_input_image))
            elif image is not None:
                # Need to check the image for API compatibility
                if isinstance(image['image'], str):
                    from modules.api.api import decode_base64_to_image
                    input_image = HWC3(np.asarray(decode_base64_to_image(image['image'])))
                else:
                    input_image = HWC3(image['image'])

                have_mask = 'mask' in image and not ((image['mask'][:, :, 0] == 0).all() or (image['mask'][:, :, 0] == 255).all())

                if 'inpaint' in unit.module:
                    print("using inpaint as input")
                    color = HWC3(image['image'])
                    if have_mask:
                        alpha = image['mask'][:, :, 0:1]
                    else:
                        alpha = np.zeros_like(color)[:, :, 0:1]
                    input_image = np.concatenate([color, alpha], axis=2)
                else:
                    if have_mask:
                        print("using mask as input")
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
                a1111_i2i_resize_mode = getattr(p, "resize_mode", None)
                if a1111_i2i_resize_mode is not None:
                    if a1111_i2i_resize_mode == 0:
                        resize_mode = external_code.ResizeMode.RESIZE
                    elif a1111_i2i_resize_mode == 1:
                        resize_mode = external_code.ResizeMode.INNER_FIT
                    elif a1111_i2i_resize_mode == 2:
                        resize_mode = external_code.ResizeMode.OUTER_FIT

            has_mask = False
            if input_image.ndim == 3:
                if input_image.shape[2] == 4:
                    if np.max(input_image[:, :, 3]) > 127:
                        has_mask = True

            a1111_mask = getattr(p, "image_mask", None)
            if 'inpaint' in unit.module and not has_mask and a1111_mask is not None:
                a1111_mask = a1111_mask.convert('L')
                if getattr(p, "inpainting_mask_invert", False):
                    a1111_mask = ImageOps.invert(a1111_mask)
                if getattr(p, "mask_blur", 0) > 0:
                    a1111_mask = a1111_mask.filter(ImageFilter.GaussianBlur(p.mask_blur))
                a1111_mask = np.asarray(a1111_mask)
                if a1111_mask.ndim == 2:
                    if a1111_mask.shape[0] == input_image.shape[0]:
                        if a1111_mask.shape[1] == input_image.shape[1]:
                            input_image = np.concatenate([input_image[:, :, 0:3], a1111_mask[:, :, None]], axis=2)
                            input_image = np.ascontiguousarray(input_image.copy()).copy()
                            a1111_i2i_resize_mode = getattr(p, "resize_mode", None)
                            if a1111_i2i_resize_mode is not None:
                                if a1111_i2i_resize_mode == 0:
                                    resize_mode = external_code.ResizeMode.RESIZE
                                elif a1111_i2i_resize_mode == 1:
                                    resize_mode = external_code.ResizeMode.INNER_FIT
                                elif a1111_i2i_resize_mode == 2:
                                    resize_mode = external_code.ResizeMode.OUTER_FIT

            if 'reference' not in unit.module and issubclass(type(p), StableDiffusionProcessingImg2Img) \
                    and p.inpaint_full_res and p.image_mask is not None:

                input_image = [input_image[:, :, i] for i in range(input_image.shape[2])]
                input_image = [Image.fromarray(x) for x in input_image]

                mask = p.image_mask.convert('L')
                if p.inpainting_mask_invert:
                    mask = ImageOps.invert(mask)
                if p.mask_blur > 0:
                    mask = mask.filter(ImageFilter.GaussianBlur(p.mask_blur))

                crop_region = masking.get_crop_region(np.array(mask), p.inpaint_full_res_padding)
                crop_region = masking.expand_crop_region(crop_region, p.width, p.height, mask.width, mask.height)

                if resize_mode == external_code.ResizeMode.INNER_FIT:
                    input_image = [images.resize_image(1, i, mask.width, mask.height) for i in input_image]
                elif resize_mode == external_code.ResizeMode.OUTER_FIT:
                    input_image = [images.resize_image(2, i, mask.width, mask.height) for i in input_image]
                else:
                    input_image = [images.resize_image(0, i, mask.width, mask.height) for i in input_image]

                input_image = [x.crop(crop_region) for x in input_image]
                input_image = [images.resize_image(2, x, p.width, p.height) for x in input_image]

                input_image = [np.asarray(x)[:, :, 0] for x in input_image]
                input_image = np.stack(input_image, axis=2)

            if 'inpaint' in unit.module and issubclass(type(p), StableDiffusionProcessingImg2Img) \
                    and p.inpainting_fill and p.image_mask is not None:
                print('A1111 inpaint and ControlNet inpaint duplicated. ControlNet support enabled.')
                unit.module = 'inpaint'

            try:
                tmp_seed = int(p.all_seeds[0] if p.seed == -1 else max(int(p.seed), 0))
                tmp_subseed = int(p.all_seeds[0] if p.subseed == -1 else max(int(p.subseed), 0))
                np.random.seed((tmp_seed + tmp_subseed) & 0xFFFFFFFF)
            except Exception as e:
                print(e)
                print('Warning: Failed to use consistent random seed.')

            # safe numpy
            input_image = np.ascontiguousarray(input_image.copy()).copy()

            print(f"Loading preprocessor: {unit.module}")
            preprocessor = self.preprocessor[unit.module]
            h, w, bsz = p.height, p.width, p.batch_size

            h = (h // 8) * 8
            w = (w // 8) * 8

            preprocessor_resolution = unit.processor_res
            if unit.pixel_perfect:
                raw_H, raw_W, _ = input_image.shape
                target_H, target_W = h, w

                k0 = float(target_H) / float(raw_H)
                k1 = float(target_W) / float(raw_W)

                if resize_mode == external_code.ResizeMode.OUTER_FIT:
                    estimation = min(k0, k1) * float(min(raw_H, raw_W))
                else:
                    estimation = max(k0, k1) * float(min(raw_H, raw_W))

                preprocessor_resolution = int(np.round(estimation))

                print(f'Pixel Perfect Mode Enabled.')
                print(f'resize_mode = {str(resize_mode)}')
                print(f'raw_H = {raw_H}')
                print(f'raw_W = {raw_W}')
                print(f'target_H = {target_H}')
                print(f'target_W = {target_W}')
                print(f'estimation = {estimation}')

            print(f'preprocessor resolution = {preprocessor_resolution}')
            detected_map, is_image = preprocessor(input_image, res=preprocessor_resolution, thr_a=unit.threshold_a, thr_b=unit.threshold_b)

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

                hr_y = (hr_y // 8) * 8
                hr_x = (hr_x // 8) * 8

                if is_image:
                    hr_control, hr_detected_map = self.detectmap_proc(detected_map, unit.module, resize_mode, hr_y, hr_x)
                    detected_maps.append((hr_detected_map, unit.module))
                else:
                    hr_control = detected_map
            else:
                hr_control = None

            if is_image:
                control, detected_map = self.detectmap_proc(detected_map, unit.module, resize_mode, h, w)
                detected_maps.append((detected_map, unit.module))
            else:
                control = detected_map
                if unit.module == 'clip_vision':
                    detected_maps.append((processor.clip_vision_visualization(detected_map), unit.module))

            control_model_type = ControlModelType.ControlNet

            if isinstance(model_net, PlugableAdapter):
                control_model_type = ControlModelType.T2I_Adapter

            if getattr(model_net, "target", None) == "scripts.adapter.StyleAdapter":
                control_model_type = ControlModelType.T2I_StyleAdapter

            if 'reference' in unit.module:
                control_model_type = ControlModelType.AttentionInjection

            global_average_pooling = False

            if model_net is not None:
                if model_net.config.model.params.get("global_average_pooling", False):
                    global_average_pooling = True

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

            if unit.module == 'inpaint_only':

                final_inpaint_feed = hr_control if hr_control is not None else control
                final_inpaint_feed = final_inpaint_feed.detach().cpu().numpy()[0].transpose([1, 2, 0])
                final_inpaint_feed = np.ascontiguousarray(final_inpaint_feed).copy()
                final_inpaint_mask = final_inpaint_feed[:, :, 3].astype(np.float32)
                final_inpaint_raw = final_inpaint_feed[:, :, 0:3].astype(np.float32) * 255.0
                final_inpaint_mask = cv2.GaussianBlur(final_inpaint_mask, (0, 0), 4)[:, :, None]
                Hmask, Wmask, _ = final_inpaint_mask.shape

                def inpaint_only_post_processing(x):
                    img = np.asarray(x).astype(np.float32)
                    H, W, C = img.shape
                    if Hmask != H or Wmask != W:
                        return x
                    result = final_inpaint_mask * img + final_inpaint_raw * (1 - final_inpaint_mask)
                    result = result.clip(0, 255).astype(np.uint8)
                    result = np.ascontiguousarray(result).copy()
                    return Image.fromarray(result)

                post_processors.append(inpaint_only_post_processing)

            del model_net

        self.latest_network = UnetHook(lowvram=hook_lowvram)
        self.latest_network.hook(model=unet, sd_ldm=sd_ldm, control_params=forward_params, process=p)
        self.detected_map = detected_maps
        self.post_processors = post_processors

    def postprocess(self, p, processed, *args):
        processor_params_flag = (', '.join(getattr(processed, 'extra_generation_params', []))).lower()

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

        if 'sd upscale' not in processor_params_flag:
            for post_processor in self.post_processors:
                processed.images = list(map(post_processor, processed.images))

        if not batch_hijack.instance.is_batch:
            if not shared.opts.data.get("control_net_no_detectmap", False):
                if 'sd upscale' not in processor_params_flag:
                    if self.detected_map is not None:
                        for detect_map, module in self.detected_map:
                            if detect_map is None:
                                continue
                            detect_map = np.ascontiguousarray(detect_map.copy()).copy()
                            if detect_map.ndim == 3 and detect_map.shape[2] == 4:
                                inpaint_mask = detect_map[:, :, 3]
                                detect_map = detect_map[:, :, 0:3]
                                detect_map[inpaint_mask > 127] = 0
                            processed.images.extend([
                                Image.fromarray(
                                    detect_map.clip(0, 255).astype(np.uint8)
                                )
                            ])

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
                    print(f'Warning: No loopback image found for controlnet unit {unit_i}. Using control map from last batch iteration instead')

    def batch_tab_postprocess(self, p, *args, **kwargs):
        self.enabled_units.clear()
        self.input_image = None
        if self.latest_network is None: return

        self.latest_network.restore(shared.sd_model.model.diffusion_model)
        self.latest_network = None
        self.detected_map.clear()


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
    shared.opts.add_option("control_net_modules_path", shared.OptionInfo(
        "", "Path to directory containing annotator model directories (requires restart, overrides corresponding command line flag)", section=section))
    shared.opts.add_option("control_net_max_models_num", shared.OptionInfo(
        1, "Multi ControlNet: Max models amount (requires restart)", gr.Slider, {"minimum": 1, "maximum": 10, "step": 1}, section=section))
    shared.opts.add_option("control_net_model_cache_size", shared.OptionInfo(
        1, "Model cache size (requires restart)", gr.Slider, {"minimum": 1, "maximum": 5, "step": 1}, section=section))
    shared.opts.add_option("control_net_no_detectmap", shared.OptionInfo(
        False, "Do not append detectmap to output", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("control_net_detectmap_autosaving", shared.OptionInfo(
        False, "Allow detectmap auto saving", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("control_net_allow_script_control", shared.OptionInfo(
        False, "Allow other script to control this extension", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("control_net_sync_field_args", shared.OptionInfo(
        False, "Passing ControlNet parameters with \"Send to img2img\"", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("controlnet_show_batch_images_in_ui", shared.OptionInfo(
        False, "Show batch images in gradio gallery output", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("controlnet_increment_seed_during_batch", shared.OptionInfo(
        False, "Increment seed after each controlnet batch iteration", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("controlnet_disable_control_type", shared.OptionInfo(
        False, "Disable control type selection", gr.Checkbox, {"interactive": True}, section=section))


def on_after_component(component, **_kwargs):
    global txt2img_submit_button
    if getattr(component, 'elem_id', None) == 'txt2img_generate':
        txt2img_submit_button = component
        return

    global img2img_submit_button
    if getattr(component, 'elem_id', None) == 'img2img_generate':
        img2img_submit_button = component
        return

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
