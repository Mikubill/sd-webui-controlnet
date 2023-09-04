import json
import gradio as gr
import functools
from copy import copy
from typing import List, Optional, Union, Callable
import numpy as np

from scripts.utils import svg_preprocess
from scripts import (
    global_state,
    external_code,
    processor,
    batch_hijack,
)
from scripts.processor import (
    preprocessor_sliders_config,
    no_control_mode_preprocessors,
    flag_preprocessor_resolution,
    model_free_preprocessors,
    preprocessor_filters,
    HWC3,
)
from scripts.logging import logger
from scripts.controlnet_ui.openpose_editor import OpenposeEditor
from scripts.controlnet_ui.preset import ControlNetPresetUI
from scripts.controlnet_ui.tool_button import ToolButton
from modules import shared
from modules.ui_components import FormRow


class UiControlNetUnit(external_code.ControlNetUnit):
    """The data class that stores all states of a ControlNetUnit."""

    def __init__(
        self,
        input_mode: batch_hijack.InputMode = batch_hijack.InputMode.SIMPLE,
        batch_images: Optional[Union[str, List[external_code.InputImage]]] = None,
        output_dir: str = "",
        loopback: bool = False,
        use_preview_as_input: bool = False,
        generated_image: Optional[np.ndarray] = None,
        enabled: bool = True,
        module: Optional[str] = None,
        model: Optional[str] = None,
        weight: float = 1.0,
        image: Optional[np.ndarray] = None,
        *args,
        **kwargs,
    ):
        if use_preview_as_input and generated_image is not None:
            input_image = generated_image
            module = "none"
        else:
            input_image = image

        super().__init__(enabled, module, model, weight, input_image, *args, **kwargs)
        self.is_ui = True
        self.input_mode = input_mode
        self.batch_images = batch_images
        self.output_dir = output_dir
        self.loopback = loopback


class ControlNetUiGroup(object):
    # Note: Change symbol hints mapping in `javascript/hints.js` when you change the symbol values.
    refresh_symbol = "\U0001f504"  # 🔄
    switch_values_symbol = "\U000021C5"  # ⇅
    camera_symbol = "\U0001F4F7"  # 📷
    reverse_symbol = "\U000021C4"  # ⇄
    tossup_symbol = "\u2934"
    trigger_symbol = "\U0001F4A5"  # 💥
    open_symbol = "\U0001F4DD"  # 📝

    global_batch_input_dir = gr.Textbox(
        label="Controlnet input directory",
        placeholder="Leave empty to use input directory",
        **shared.hide_dirs,
        elem_id="controlnet_batch_input_dir",
    )
    img2img_batch_input_dir = None
    img2img_batch_input_dir_callbacks = []
    img2img_batch_output_dir = None
    img2img_batch_output_dir_callbacks = []
    txt2img_submit_button = None
    img2img_submit_button = None

    # Slider controls from A1111 WebUI.
    txt2img_w_slider = None
    txt2img_h_slider = None
    img2img_w_slider = None
    img2img_h_slider = None

    def __init__(
        self,
        gradio_compat: bool,
        default_unit: external_code.ControlNetUnit,
        preprocessors: List[Callable],
    ):
        self.gradio_compat = gradio_compat
        self.default_unit = default_unit
        self.preprocessors = preprocessors
        self.webcam_enabled = False
        self.webcam_mirrored = False

        # Note: All gradio elements declared in `render` will be defined as member variable.
        self.upload_tab = None
        self.image = None
        self.generated_image_group = None
        self.generated_image = None
        self.batch_tab = None
        self.batch_image_dir = None
        self.create_canvas = None
        self.canvas_width = None
        self.canvas_height = None
        self.canvas_create_button = None
        self.canvas_cancel_button = None
        self.open_new_canvas_button = None
        self.webcam_enable = None
        self.webcam_mirror = None
        self.send_dimen_button = None
        self.enabled = None
        self.low_vram = None
        self.pixel_perfect = None
        self.preprocessor_preview = None
        self.type_filter = None
        self.module = None
        self.trigger_preprocessor = None
        self.model = None
        self.refresh_models = None
        self.weight = None
        self.guidance_start = None
        self.guidance_end = None
        self.advanced = None
        self.processor_res = None
        self.threshold_a = None
        self.threshold_b = None
        self.control_mode = None
        self.resize_mode = None
        self.loopback = None
        self.use_preview_as_input = None
        self.openpose_editor = None
        self.preset_panel = None
        self.upload_independent_img_in_img2img = None
        self.image_upload_panel = None

        # Internal states for UI state pasting.
        self.prevent_next_n_module_update = 0
        self.prevent_next_n_slider_value_update = 0

    def render(self, tabname: str, elem_id_tabname: str, is_img2img: bool) -> None:
        """The pure HTML structure of a single ControlNetUnit. Calling this
        function will populate `self` with all gradio element declared
        in local scope.

        Args:
            tabname:
            elem_id_tabname:

        Returns:
            None
        """
        with gr.Group(visible=not is_img2img) as self.image_upload_panel:
            with gr.Tabs():
                with gr.Tab(label="Single Image") as self.upload_tab:
                    with gr.Row(elem_classes=["cnet-image-row"], equal_height=True):
                        with gr.Group(elem_classes=["cnet-input-image-group"]):
                            self.image = gr.Image(
                                source="upload",
                                brush_radius=20,
                                mirror_webcam=False,
                                type="numpy",
                                tool="sketch",
                                elem_id=f"{elem_id_tabname}_{tabname}_input_image",
                                elem_classes=["cnet-image"],
                                brush_color=shared.opts.img2img_inpaint_mask_brush_color
                                if hasattr(
                                    shared.opts, "img2img_inpaint_mask_brush_color"
                                )
                                else None,
                            )
                        with gr.Group(
                            visible=False, elem_classes=["cnet-generated-image-group"]
                        ) as self.generated_image_group:
                            self.generated_image = gr.Image(
                                value=None,
                                label="Preprocessor Preview",
                                elem_id=f"{elem_id_tabname}_{tabname}_generated_image",
                                elem_classes=["cnet-image"],
                                interactive=True,
                                height=242
                            )  # Gradio's magic number. Only 242 works.

                            with gr.Group(
                                elem_classes=["cnet-generated-image-control-group"]
                            ):
                                self.openpose_editor = OpenposeEditor()
                                preview_check_elem_id = f"{elem_id_tabname}_{tabname}_controlnet_preprocessor_preview_checkbox"
                                preview_close_button_js = f"document.querySelector('#{preview_check_elem_id} input[type=\\'checkbox\\']').click();"
                                gr.HTML(
                                    value=f"""<a title="Close Preview" onclick="{preview_close_button_js}">Close</a>""",
                                    visible=True,
                                    elem_classes=["cnet-close-preview"],
                                )

                with gr.Tab(label="Batch") as self.batch_tab:
                    self.batch_image_dir = gr.Textbox(
                        label="Input Directory",
                        placeholder="Leave empty to use img2img batch controlnet input directory",
                        elem_id=f"{elem_id_tabname}_{tabname}_batch_image_dir",
                    )

            with gr.Accordion(
                label="Open New Canvas", visible=False
            ) as self.create_canvas:
                self.canvas_width = gr.Slider(
                    label="New Canvas Width",
                    minimum=256,
                    maximum=1024,
                    value=512,
                    step=64,
                    elem_id=f"{elem_id_tabname}_{tabname}_controlnet_canvas_width",
                )
                self.canvas_height = gr.Slider(
                    label="New Canvas Height",
                    minimum=256,
                    maximum=1024,
                    value=512,
                    step=64,
                    elem_id=f"{elem_id_tabname}_{tabname}_controlnet_canvas_height",
                )
                with gr.Row():
                    self.canvas_create_button = gr.Button(
                        value="Create New Canvas",
                        elem_id=f"{elem_id_tabname}_{tabname}_controlnet_canvas_create_button",
                    )
                    self.canvas_cancel_button = gr.Button(
                        value="Cancel",
                        elem_id=f"{elem_id_tabname}_{tabname}_controlnet_canvas_cancel_button",
                    )

            with gr.Row(elem_classes="controlnet_image_controls"):
                gr.HTML(
                    value="<p>Set the preprocessor to [invert] If your image has white background and black lines.</p>",
                    elem_classes="controlnet_invert_warning",
                )
                self.open_new_canvas_button = ToolButton(
                    value=ControlNetUiGroup.open_symbol,
                    elem_id=f"{elem_id_tabname}_{tabname}_controlnet_open_new_canvas_button",
                )
                self.webcam_enable = ToolButton(
                    value=ControlNetUiGroup.camera_symbol,
                    elem_id=f"{elem_id_tabname}_{tabname}_controlnet_webcam_enable",
                )
                self.webcam_mirror = ToolButton(
                    value=ControlNetUiGroup.reverse_symbol,
                    elem_id=f"{elem_id_tabname}_{tabname}_controlnet_webcam_mirror",
                )
                self.send_dimen_button = ToolButton(
                    value=ControlNetUiGroup.tossup_symbol,
                    elem_id=f"{elem_id_tabname}_{tabname}_controlnet_send_dimen_button",
                )

        with FormRow(elem_classes=["controlnet_main_options"]):
            self.enabled = gr.Checkbox(
                label="Enable",
                value=self.default_unit.enabled,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_enable_checkbox",
                elem_classes=["cnet-unit-enabled"],
            )
            self.low_vram = gr.Checkbox(
                label="Low VRAM",
                value=self.default_unit.low_vram,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_low_vram_checkbox",
            )
            self.pixel_perfect = gr.Checkbox(
                label="Pixel Perfect",
                value=self.default_unit.pixel_perfect,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_pixel_perfect_checkbox",
            )
            self.preprocessor_preview = gr.Checkbox(
                label="Allow Preview",
                value=False,
                elem_id=preview_check_elem_id,
                visible=not is_img2img,
            )
            self.use_preview_as_input = gr.Checkbox(
                label="Preview as Input",
                value=False,
                elem_classes=["cnet-preview-as-input"],
                visible=False,
            )

        with gr.Row(elem_classes="controlnet_img2img_options"):
            if is_img2img:
                self.upload_independent_img_in_img2img = gr.Checkbox(
                    label="Upload independent control image",
                    value=False,
                    elem_id=f"{elem_id_tabname}_{tabname}_controlnet_same_img2img_checkbox",
                    elem_classes=["cnet-unit-same_img2img"],
                )
            else:
                self.upload_independent_img_in_img2img = None

        if not shared.opts.data.get("controlnet_disable_control_type", False):
            with gr.Row(elem_classes=["controlnet_control_type", "controlnet_row"]):
                self.type_filter = gr.Radio(
                    list(preprocessor_filters.keys()),
                    label=f"Control Type",
                    value="All",
                    elem_id=f"{elem_id_tabname}_{tabname}_controlnet_type_filter_radio",
                    elem_classes="controlnet_control_type_filter_group",
                )

        with gr.Row(elem_classes=["controlnet_preprocessor_model", "controlnet_row"]):
            self.module = gr.Dropdown(
                global_state.ui_preprocessor_keys,
                label=f"Preprocessor",
                value=self.default_unit.module,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_preprocessor_dropdown",
            )
            self.trigger_preprocessor = ToolButton(
                value=ControlNetUiGroup.trigger_symbol,
                visible=not is_img2img,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_trigger_preprocessor",
                elem_classes=["cnet-run-preprocessor"],
            )
            self.model = gr.Dropdown(
                list(global_state.cn_models.keys()),
                label=f"Model",
                value=self.default_unit.model,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_model_dropdown",
            )
            self.refresh_models = ToolButton(
                value=ControlNetUiGroup.refresh_symbol,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_refresh_models",
            )

        with gr.Row(elem_classes=["controlnet_weight_steps", "controlnet_row"]):
            self.weight = gr.Slider(
                label=f"Control Weight",
                value=self.default_unit.weight,
                minimum=0.0,
                maximum=2.0,
                step=0.05,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_control_weight_slider",
                elem_classes="controlnet_control_weight_slider",
            )
            self.guidance_start = gr.Slider(
                label="Starting Control Step",
                value=self.default_unit.guidance_start,
                minimum=0.0,
                maximum=1.0,
                interactive=True,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_start_control_step_slider",
                elem_classes="controlnet_start_control_step_slider",
            )
            self.guidance_end = gr.Slider(
                label="Ending Control Step",
                value=self.default_unit.guidance_end,
                minimum=0.0,
                maximum=1.0,
                interactive=True,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_ending_control_step_slider",
                elem_classes="controlnet_ending_control_step_slider",
            )

        # advanced options
        with gr.Column(visible=False) as self.advanced:
            self.processor_res = gr.Slider(
                label="Preprocessor resolution",
                value=self.default_unit.processor_res,
                minimum=64,
                maximum=2048,
                visible=False,
                interactive=True,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_preprocessor_resolution_slider",
            )
            self.threshold_a = gr.Slider(
                label="Threshold A",
                value=self.default_unit.threshold_a,
                minimum=64,
                maximum=1024,
                visible=False,
                interactive=True,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_threshold_A_slider",
            )
            self.threshold_b = gr.Slider(
                label="Threshold B",
                value=self.default_unit.threshold_b,
                minimum=64,
                maximum=1024,
                visible=False,
                interactive=True,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_threshold_B_slider",
            )

        self.control_mode = gr.Radio(
            choices=[e.value for e in external_code.ControlMode],
            value=self.default_unit.control_mode.value,
            label="Control Mode",
            elem_id=f"{elem_id_tabname}_{tabname}_controlnet_control_mode_radio",
            elem_classes="controlnet_control_mode_radio",
        )

        self.resize_mode = gr.Radio(
            choices=[e.value for e in external_code.ResizeMode],
            value=self.default_unit.resize_mode.value,
            label="Resize Mode",
            elem_id=f"{elem_id_tabname}_{tabname}_controlnet_resize_mode_radio",
            elem_classes="controlnet_resize_mode_radio",
            visible=not is_img2img,
        )

        self.loopback = gr.Checkbox(
            label="[Loopback] Automatically send generated images to this ControlNet unit",
            value=self.default_unit.loopback,
            elem_id=f"{elem_id_tabname}_{tabname}_controlnet_automatically_send_generated_images_checkbox",
            elem_classes="controlnet_loopback_checkbox",
            visible=not is_img2img,
        )

        self.preset_panel = ControlNetPresetUI(
            id_prefix=f"{elem_id_tabname}_{tabname}_"
        )

    def register_send_dimensions(self, is_img2img: bool):
        """Register event handler for send dimension button."""

        def send_dimensions(image):
            def closesteight(num):
                rem = num % 8
                if rem <= 4:
                    return round(num - rem)
                else:
                    return round(num + (8 - rem))

            if image:
                interm = np.asarray(image.get("image"))
                return closesteight(interm.shape[1]), closesteight(interm.shape[0])
            else:
                return gr.Slider.update(), gr.Slider.update()

        outputs = (
            [
                ControlNetUiGroup.img2img_w_slider,
                ControlNetUiGroup.img2img_h_slider,
            ]
            if is_img2img
            else [
                ControlNetUiGroup.txt2img_w_slider,
                ControlNetUiGroup.txt2img_h_slider,
            ]
        )
        self.send_dimen_button.click(
            fn=send_dimensions,
            inputs=[self.image],
            outputs=outputs,
        )

    def register_webcam_toggle(self):
        def webcam_toggle():
            self.webcam_enabled = not self.webcam_enabled
            return {
                "value": None,
                "source": "webcam" if self.webcam_enabled else "upload",
                "__type__": "update",
            }

        self.webcam_enable.click(webcam_toggle, inputs=None, outputs=self.image)

    def register_webcam_mirror_toggle(self):
        def webcam_mirror_toggle():
            self.webcam_mirrored = not self.webcam_mirrored
            return {"mirror_webcam": self.webcam_mirrored, "__type__": "update"}

        self.webcam_mirror.click(webcam_mirror_toggle, inputs=None, outputs=self.image)

    def register_refresh_all_models(self):
        def refresh_all_models(*inputs):
            global_state.update_cn_models()

            dd = inputs[0]
            selected = dd if dd in global_state.cn_models else "None"
            return gr.Dropdown.update(
                value=selected, choices=list(global_state.cn_models.keys())
            )

        self.refresh_models.click(refresh_all_models, self.model, self.model)

    def register_build_sliders(self):
        if not self.gradio_compat:
            return

        def build_sliders(module: str, pp: bool):
            logger.debug(
                f"Prevent update slider value: {self.prevent_next_n_slider_value_update}"
            )
            logger.debug(f"Build slider for module: {module} - {pp}")

            # Clear old slider values so that they do not cause confusion in
            # infotext.
            clear_slider_update = gr.update(
                visible=False,
                interactive=True,
                minimum=-1,
                maximum=-1,
                value=-1,
            )

            grs = []
            module = global_state.get_module_basename(module)
            if module not in preprocessor_sliders_config:
                default_res_slider_config = dict(
                    label=flag_preprocessor_resolution,
                    minimum=64,
                    maximum=2048,
                    step=1,
                )
                if self.prevent_next_n_slider_value_update == 0:
                    default_res_slider_config["value"] = 512

                grs += [
                    gr.update(
                        **default_res_slider_config,
                        visible=not pp,
                        interactive=True,
                    ),
                    copy(clear_slider_update),
                    copy(clear_slider_update),
                    gr.update(visible=True),
                ]
            else:
                for slider_config in preprocessor_sliders_config[module]:
                    if isinstance(slider_config, dict):
                        visible = True
                        if slider_config["name"] == flag_preprocessor_resolution:
                            visible = not pp
                        slider_update = gr.update(
                            label=slider_config["name"],
                            minimum=slider_config["min"],
                            maximum=slider_config["max"],
                            step=slider_config["step"]
                            if "step" in slider_config
                            else 1,
                            visible=visible,
                            interactive=True,
                        )
                        if self.prevent_next_n_slider_value_update == 0:
                            slider_update["value"] = slider_config["value"]

                        grs.append(slider_update)

                    else:
                        grs.append(copy(clear_slider_update))
                while len(grs) < 3:
                    grs.append(copy(clear_slider_update))
                grs.append(gr.update(visible=True))
            if module in model_free_preprocessors:
                grs += [
                    gr.update(visible=False, value="None"),
                    gr.update(visible=False),
                ]
            else:
                grs += [gr.update(visible=True), gr.update(visible=True)]

            self.prevent_next_n_slider_value_update = max(
                0, self.prevent_next_n_slider_value_update - 1
            )

            grs += [gr.update(visible=module not in no_control_mode_preprocessors)]

            return grs

        inputs = [
            self.module,
            self.pixel_perfect,
        ]
        outputs = [
            self.processor_res,
            self.threshold_a,
            self.threshold_b,
            self.advanced,
            self.model,
            self.refresh_models,
            self.control_mode
        ]
        self.module.change(build_sliders, inputs=inputs, outputs=outputs)
        self.pixel_perfect.change(build_sliders, inputs=inputs, outputs=outputs)

        if self.type_filter is not None:

            def filter_selected(k: str):
                logger.debug(f"Prevent update {self.prevent_next_n_module_update}")
                logger.debug(f"Switch to control type {k}")
                (
                    filtered_preprocessor_list,
                    filtered_model_list,
                    default_option,
                    default_model,
                ) = global_state.select_control_type(k)

                if self.prevent_next_n_module_update > 0:
                    self.prevent_next_n_module_update -= 1
                    return [
                        gr.Dropdown.update(choices=filtered_preprocessor_list),
                        gr.Dropdown.update(choices=filtered_model_list),
                    ]
                else:
                    return [
                        gr.Dropdown.update(
                            value=default_option, choices=filtered_preprocessor_list
                        ),
                        gr.Dropdown.update(
                            value=default_model, choices=filtered_model_list
                        ),
                    ]

            self.type_filter.change(
                filter_selected,
                inputs=[self.type_filter],
                outputs=[self.module, self.model],
            )

    def register_run_annotator(self, is_img2img: bool):
        def run_annotator(image, module, pres, pthr_a, pthr_b, t2i_w, t2i_h, pp, rm):
            if image is None:
                return (
                    gr.update(value=None, visible=True),
                    gr.update(),
                    *self.openpose_editor.update(""),
                )

            img = HWC3(image["image"])
            has_mask = not (
                (image["mask"][:, :, 0] <= 5).all()
                or (image["mask"][:, :, 0] >= 250).all()
            )
            if "inpaint" in module:
                color = HWC3(image["image"])
                alpha = image["mask"][:, :, 0:1]
                img = np.concatenate([color, alpha], axis=2)
            elif has_mask and not shared.opts.data.get(
                "controlnet_ignore_noninpaint_mask", False
            ):
                img = HWC3(image["mask"][:, :, 0])

            module = global_state.get_module_basename(module)
            preprocessor = self.preprocessors[module]

            if pp:
                pres = external_code.pixel_perfect_resolution(
                    img,
                    target_H=t2i_h,
                    target_W=t2i_w,
                    resize_mode=external_code.resize_mode_from_value(rm),
                )

            class JsonAcceptor:
                def __init__(self) -> None:
                    self.value = ""

                def accept(self, json_dict: dict) -> None:
                    self.value = json.dumps(json_dict)

            json_acceptor = JsonAcceptor()

            logger.info(f"Preview Resolution = {pres}")

            def is_openpose(module: str):
                return "openpose" in module

            # Only openpose preprocessor returns a JSON output, pass json_acceptor
            # only when a JSON output is expected. This will make preprocessor cache
            # work for all other preprocessors other than openpose ones. JSON acceptor
            # instance are different every call, which means cache will never take
            # effect.
            # TODO: Maybe we should let `preprocessor` return a Dict to alleviate this issue?
            # This requires changing all callsites though.
            result, is_image = preprocessor(
                img,
                res=pres,
                thr_a=pthr_a,
                thr_b=pthr_b,
                json_pose_callback=json_acceptor.accept
                if is_openpose(module)
                else None,
            )

            if not is_image:
                result = img
                is_image = True

            result = external_code.visualize_inpaint_mask(result)
            return (
                # Update to `generated_image`
                gr.update(value=result, visible=True, interactive=False),
                # preprocessor_preview
                gr.update(value=True),
                # openpose editor
                *self.openpose_editor.update(json_acceptor.value),
            )

        self.trigger_preprocessor.click(
            fn=run_annotator,
            inputs=[
                self.image,
                self.module,
                self.processor_res,
                self.threshold_a,
                self.threshold_b,
                ControlNetUiGroup.img2img_w_slider
                if is_img2img
                else ControlNetUiGroup.txt2img_w_slider,
                ControlNetUiGroup.img2img_h_slider
                if is_img2img
                else ControlNetUiGroup.txt2img_h_slider,
                self.pixel_perfect,
                self.resize_mode,
            ],
            outputs=[
                self.generated_image,
                self.preprocessor_preview,
                *self.openpose_editor.outputs(),
            ],
        )

    def register_shift_preview(self):
        def shift_preview(is_on):
            return (
                # generated_image
                gr.update() if is_on else gr.update(value=None),
                # generated_image_group
                gr.update(visible=is_on),
                # use_preview_as_input,
                gr.update(visible=False),  # Now this is automatically managed
                # download_pose_link
                gr.update() if is_on else gr.update(value=None),
                # modal edit button
                gr.update() if is_on else gr.update(visible=False),
            )

        self.preprocessor_preview.change(
            fn=shift_preview,
            inputs=[self.preprocessor_preview],
            outputs=[
                self.generated_image,
                self.generated_image_group,
                self.use_preview_as_input,
                self.openpose_editor.download_link,
                self.openpose_editor.modal,
            ],
        )

    def register_create_canvas(self):
        self.open_new_canvas_button.click(
            lambda: gr.Accordion.update(visible=True),
            inputs=None,
            outputs=self.create_canvas,
        )
        self.canvas_cancel_button.click(
            lambda: gr.Accordion.update(visible=False),
            inputs=None,
            outputs=self.create_canvas,
        )

        def fn_canvas(h, w):
            return np.zeros(shape=(h, w, 3), dtype=np.uint8) + 255, gr.Accordion.update(
                visible=False
            )

        self.canvas_create_button.click(
            fn=fn_canvas,
            inputs=[self.canvas_height, self.canvas_width],
            outputs=[self.image, self.create_canvas],
        )

    def register_img2img_same_input(self):
        def fn_same_checked(x):
            return [
                gr.update(value=None),
                gr.update(value=None),
                gr.update(value=False, visible=x),
            ] + [gr.update(visible=x)] * 4

        self.upload_independent_img_in_img2img.change(
            fn_same_checked,
            inputs=self.upload_independent_img_in_img2img,
            outputs=[
                self.image,
                self.batch_image_dir,
                self.preprocessor_preview,
                self.image_upload_panel,
                self.trigger_preprocessor,
                self.loopback,
                self.resize_mode,
            ],
        )
        return

    def register_callbacks(self, is_img2img: bool):
        """Register callbacks on the UI elements.

        Args:
            is_img2img: Whether ControlNet is under img2img. False when in txt2img mode.

        Returns:
            None
        """
        self.register_send_dimensions(is_img2img)
        self.register_webcam_toggle()
        self.register_webcam_mirror_toggle()
        self.register_refresh_all_models()
        self.register_build_sliders()
        self.register_run_annotator(is_img2img)
        self.register_shift_preview()
        self.register_create_canvas()
        self.openpose_editor.register_callbacks(
            self.generated_image, self.use_preview_as_input
        )
        self.preset_panel.register_callbacks(
            self,
            self.type_filter,
            *[
                getattr(self, key)
                for key in vars(external_code.ControlNetUnit()).keys()
            ],
        )
        if is_img2img:
            self.register_img2img_same_input()

    def render_and_register_unit(self, tabname: str, is_img2img: bool):
        """Render the invisible states elements for misc persistent
        purposes. Register callbacks on loading/unloading the controlnet
        unit and handle batch processes.

        Args:
            tabname:
            is_img2img:

        Returns:
            The data class "ControlNetUnit" representing this ControlNetUnit.
        """
        input_mode = gr.State(batch_hijack.InputMode.SIMPLE)
        batch_image_dir_state = gr.State("")
        output_dir_state = gr.State("")
        unit_args = (
            input_mode,
            batch_image_dir_state,
            output_dir_state,
            self.loopback,
            # Non-persistent fields.
            # Following inputs will not be persistent on `ControlNetUnit`.
            # They are only used during object construction.
            self.use_preview_as_input,
            self.generated_image,
            # End of Non-persistent fields.
            self.enabled,
            self.module,
            self.model,
            self.weight,
            self.image,
            self.resize_mode,
            self.low_vram,
            self.processor_res,
            self.threshold_a,
            self.threshold_b,
            self.guidance_start,
            self.guidance_end,
            self.pixel_perfect,
            self.control_mode,
        )

        self.image.preprocess = functools.partial(
            svg_preprocess, preprocess=self.image.preprocess
        )

        unit = gr.State(self.default_unit)
        for comp in unit_args:
            event_subscribers = []
            if hasattr(comp, "edit"):
                event_subscribers.append(comp.edit)
            elif hasattr(comp, "click"):
                event_subscribers.append(comp.click)
            elif isinstance(comp, gr.Slider) and hasattr(comp, "release"):
                event_subscribers.append(comp.release)
            elif hasattr(comp, "change"):
                event_subscribers.append(comp.change)

            if hasattr(comp, "clear"):
                event_subscribers.append(comp.clear)

            for event_subscriber in event_subscribers:
                event_subscriber(
                    fn=UiControlNetUnit, inputs=list(unit_args), outputs=unit
                )

        def clear_preview(x):
            if x:
                logger.info("Preview as input is cancelled.")
            return gr.update(value=False), gr.update(value=None)

        for comp in (
            self.pixel_perfect,
            self.module,
            self.image,
            self.processor_res,
            self.threshold_a,
            self.threshold_b,
            self.upload_independent_img_in_img2img,
        ):
            event_subscribers = []
            if hasattr(comp, "edit"):
                event_subscribers.append(comp.edit)
            elif hasattr(comp, "click"):
                event_subscribers.append(comp.click)
            elif isinstance(comp, gr.Slider) and hasattr(comp, "release"):
                event_subscribers.append(comp.release)
            elif hasattr(comp, "change"):
                event_subscribers.append(comp.change)
            if hasattr(comp, "clear"):
                event_subscribers.append(comp.clear)
            for event_subscriber in event_subscribers:
                event_subscriber(
                    fn=clear_preview,
                    inputs=self.use_preview_as_input,
                    outputs=[self.use_preview_as_input, self.generated_image],
                )

        # keep input_mode in sync
        def ui_controlnet_unit_for_input_mode(input_mode, *args):
            args = list(args)
            args[0] = input_mode
            return input_mode, UiControlNetUnit(*args)

        for input_tab in (
            (self.upload_tab, batch_hijack.InputMode.SIMPLE),
            (self.batch_tab, batch_hijack.InputMode.BATCH),
        ):
            input_tab[0].select(
                fn=ui_controlnet_unit_for_input_mode,
                inputs=[gr.State(input_tab[1])] + list(unit_args),
                outputs=[input_mode, unit],
            )

        def determine_batch_dir(batch_dir, fallback_dir, fallback_fallback_dir):
            if batch_dir:
                return batch_dir
            elif fallback_dir:
                return fallback_dir
            else:
                return fallback_fallback_dir

        # keep batch_dir in sync with global batch input textboxes
        def subscribe_for_batch_dir():
            batch_dirs = [
                self.batch_image_dir,
                ControlNetUiGroup.global_batch_input_dir,
                ControlNetUiGroup.img2img_batch_input_dir,
            ]
            for batch_dir_comp in batch_dirs:
                subscriber = getattr(batch_dir_comp, "blur", None)
                if subscriber is None:
                    continue
                subscriber(
                    fn=determine_batch_dir,
                    inputs=batch_dirs,
                    outputs=[batch_image_dir_state],
                    queue=False,
                )

        if ControlNetUiGroup.img2img_batch_input_dir is None:
            # we are too soon, subscribe later when available
            ControlNetUiGroup.img2img_batch_input_dir_callbacks.append(
                subscribe_for_batch_dir
            )
        else:
            subscribe_for_batch_dir()

        # keep output_dir in sync with global batch output textbox
        def subscribe_for_output_dir():
            ControlNetUiGroup.img2img_batch_output_dir.blur(
                fn=lambda a: a,
                inputs=[ControlNetUiGroup.img2img_batch_output_dir],
                outputs=[output_dir_state],
                queue=False,
            )

        if ControlNetUiGroup.img2img_batch_input_dir is None:
            # we are too soon, subscribe later when available
            ControlNetUiGroup.img2img_batch_output_dir_callbacks.append(
                subscribe_for_output_dir
            )
        else:
            subscribe_for_output_dir()

        (
            ControlNetUiGroup.img2img_submit_button
            if is_img2img
            else ControlNetUiGroup.txt2img_submit_button
        ).click(
            fn=UiControlNetUnit,
            inputs=list(unit_args),
            outputs=unit,
            queue=False,
        )

        return unit

    @staticmethod
    def on_after_component(component, **_kwargs):
        elem_id = getattr(component, "elem_id", None)

        if elem_id == "txt2img_generate":
            ControlNetUiGroup.txt2img_submit_button = component
            return

        if elem_id == "img2img_generate":
            ControlNetUiGroup.img2img_submit_button = component
            return

        if elem_id == "img2img_batch_input_dir":
            ControlNetUiGroup.img2img_batch_input_dir = component
            for callback in ControlNetUiGroup.img2img_batch_input_dir_callbacks:
                callback()
            return

        if elem_id == "img2img_batch_output_dir":
            ControlNetUiGroup.img2img_batch_output_dir = component
            for callback in ControlNetUiGroup.img2img_batch_output_dir_callbacks:
                callback()
            return

        if elem_id == "img2img_batch_inpaint_mask_dir":
            ControlNetUiGroup.global_batch_input_dir.render()
            return

        if elem_id == "txt2img_width":
            ControlNetUiGroup.txt2img_w_slider = component
            return

        if elem_id == "txt2img_height":
            ControlNetUiGroup.txt2img_h_slider = component
            return

        if elem_id == "img2img_width":
            ControlNetUiGroup.img2img_w_slider = component
            return

        if elem_id == "img2img_height":
            ControlNetUiGroup.img2img_h_slider = component
            return
