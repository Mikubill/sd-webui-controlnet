import os
import gradio as gr

from typing import Dict, List

from modules import scripts
from scripts.infotext import parse_unit, serialize_unit
from scripts.controlnet_ui.tool_button import ToolButton
from scripts.logging import logger
from scripts.processor import preprocessor_filters
from scripts import external_code

save_symbol = "\U0001f4be"  # 💾
delete_symbol = "\U0001f5d1\ufe0f"  # 🗑️
refresh_symbol = "\U0001f504"  # 🔄

NEW_PRESET = "New Preset"


def load_presets(preset_dir: str) -> Dict[str, str]:
    if not os.path.exists(preset_dir):
        os.makedirs(preset_dir)
        return {}

    presets = {}
    for filename in os.listdir(preset_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(preset_dir, filename), "r") as f:
                name = filename.replace(".txt", "")
                if name == NEW_PRESET:
                    continue
                presets[name] = f.read()
    return presets


def infer_control_type(module: str, model: str) -> str:
    control_types = preprocessor_filters.keys()
    control_type_candidates = [
        control_type
        for control_type in control_types
        if control_type.lower() in module or control_type.lower() in model
    ]
    if len(control_type_candidates) != 1:
        raise ValueError(
            f"Unable to infer control type from module {module} and model {model}"
        )
    return control_type_candidates[0]


class ControlNetPresetUI(object):
    preset_directory = os.path.join(scripts.basedir(), "presets")
    presets = load_presets(preset_directory)

    def __init__(self, id_prefix: str):
        self.dropdown = None
        self.save_button = None
        self.delete_button = None
        self.refresh_button = None
        self.preset_name = None
        self.confirm_preset_name = None
        self.name_dialog = None
        self.render(id_prefix)

    def render(self, id_prefix: str):
        with gr.Row():
            self.dropdown = gr.Dropdown(
                label="Presets",
                show_label=True,
                elem_classes=["cnet-preset-dropdown"],
                choices=ControlNetPresetUI.dropdown_choices(),
                value=NEW_PRESET,
            )
            self.save_button = ToolButton(
                value=save_symbol,
                elem_classes=["cnet-preset-save"],
                tooltip="Save preset",
            )
            self.delete_button = ToolButton(
                value=delete_symbol,
                elem_classes=["cnet-preset-delete"],
                tooltip="Delete preset",
            )
            self.refresh_button = ToolButton(
                value=refresh_symbol,
                elem_classes=["cnet-preset-refresh"],
                tooltip="Refresh preset",
            )

        with gr.Box(
            elem_classes=["popup-dialog", "cnet-preset-enter-name"],
            elem_id=f"{id_prefix}_cnet_preset_enter_name",
        ) as self.name_dialog:
            with gr.Row():
                self.preset_name = gr.Textbox(
                    label="Preset name",
                    show_label=True,
                    lines=1,
                    elem_classes=["cnet-preset-name"],
                )
                self.confirm_preset_name = ToolButton(
                    value=save_symbol,
                    elem_classes=["cnet-preset-confirm-name"],
                    tooltip="Save preset",
                )

    def register_callbacks(self, control_type: gr.Radio, *ui_states):
        # Do application 5 times
        # If first update triggers control type change, wrong module will be
        # selected
        # 2nd update will be executed at the same time with the module update
        # If 3rd update triggers module change, wrong slider values will
        # be used.
        # 4th update will be executed at the same time with slider updates
        # 5th update will be updating slider values
        # TODO(huchenlei): This is exetremely hacky, need to find a better way
        # to achieve the functionality.
        self.dropdown.change(
            fn=ControlNetPresetUI.apply_preset,
            inputs=[self.dropdown],
            outputs=[self.delete_button, control_type, *ui_states],
            show_progress=False,
        ).then(
            fn=ControlNetPresetUI.apply_preset,
            inputs=[self.dropdown],
            outputs=[self.delete_button, control_type, *ui_states],
            show_progress=False,
        ).then(
            fn=ControlNetPresetUI.apply_preset,
            inputs=[self.dropdown],
            outputs=[self.delete_button, control_type, *ui_states],
            show_progress=False,
        ).then(
            fn=ControlNetPresetUI.apply_preset,
            inputs=[self.dropdown],
            outputs=[self.delete_button, control_type, *ui_states],
            show_progress=False,
        ).then(
            fn=ControlNetPresetUI.apply_preset,
            inputs=[self.dropdown],
            outputs=[self.delete_button, control_type, *ui_states],
            show_progress=False,
        )

        def save_preset(name: str, *ui_states):
            if name == NEW_PRESET:
                return gr.update(visible=True), gr.update()

            ControlNetPresetUI.save_preset(
                name, external_code.ControlNetUnit(*ui_states)
            )
            return gr.update(), gr.update(
                choices=ControlNetPresetUI.dropdown_choices(), value=name
            )

        self.save_button.click(
            fn=save_preset,
            inputs=[self.dropdown, *ui_states],
            outputs=[self.name_dialog, self.dropdown],
            show_progress=False,
        ).then(
            fn=None,
            _js=f"""
            (name) => {{
                if (name === "{NEW_PRESET}")
                    popup(gradioApp().getElementById('{self.name_dialog.elem_id}'));
            }}""",
            inputs=[self.dropdown],
        )

        def delete_preset(name: str):
            ControlNetPresetUI.delete_preset(name)
            return gr.Dropdown.update(
                choices=ControlNetPresetUI.dropdown_choices(),
                value=NEW_PRESET,
            )

        self.delete_button.click(
            fn=delete_preset,
            inputs=[self.dropdown],
            outputs=[self.dropdown],
            show_progress=False,
        )

        self.name_dialog.visible = False

        def save_new_preset(new_name: str, *ui_states):
            if new_name == NEW_PRESET:
                logger.warn(f"Cannot save preset with reserved name '{NEW_PRESET}'")
                return gr.update(visible=False), gr.update()

            ControlNetPresetUI.save_preset(
                new_name, external_code.ControlNetUnit(*ui_states)
            )
            return gr.update(visible=False), gr.update(
                choices=ControlNetPresetUI.dropdown_choices(), value=new_name
            )

        self.confirm_preset_name.click(
            fn=save_new_preset,
            inputs=[self.preset_name, *ui_states],
            outputs=[self.name_dialog, self.dropdown],
            show_progress=False,
        ).then(fn=None, _js="closePopup")

        self.refresh_button.click(
            fn=ControlNetPresetUI.refresh_preset,
            inputs=None,
            outputs=[self.dropdown],
            show_progress=False,
        )

    @staticmethod
    def dropdown_choices() -> List[str]:
        return list(ControlNetPresetUI.presets.keys()) + [NEW_PRESET]

    @staticmethod
    def save_preset(name: str, unit: external_code.ControlNetUnit):
        infotext = serialize_unit(unit)
        with open(
            os.path.join(ControlNetPresetUI.preset_directory, f"{name}.txt"), "w"
        ) as f:
            f.write(infotext)

        ControlNetPresetUI.presets[name] = infotext

    @staticmethod
    def delete_preset(name: str):
        if name not in ControlNetPresetUI.presets:
            return

        del ControlNetPresetUI.presets[name]

        file = os.path.join(ControlNetPresetUI.preset_directory, f"{name}.txt")
        if os.path.exists(file):
            os.unlink(file)

    @staticmethod
    def apply_preset(name: str):
        if name == NEW_PRESET:
            return (
                gr.update(visible=False),
                *(
                    (gr.update(),)
                    * (len(vars(external_code.ControlNetUnit()).keys()) + 1)
                ),
            )

        assert name in ControlNetPresetUI.presets
        infotext = ControlNetPresetUI.presets[name]
        unit = parse_unit(infotext)

        try:
            control_type_update = gr.update(
                value=infer_control_type(unit.module, unit.model)
            )
        except ValueError as e:
            logger.error(e)
            control_type_update = gr.update()

        return (
            gr.update(visible=True),
            control_type_update,
            *[
                gr.update(value=value) if value is not None else gr.update()
                for value in vars(unit).values()
            ],
        )

    @staticmethod
    def refresh_preset():
        ControlNetPresetUI.presets = load_presets(ControlNetPresetUI.preset_directory)
        return gr.update(choices=ControlNetPresetUI.dropdown_choices())
