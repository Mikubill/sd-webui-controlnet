import os
import gradio as gr

from typing import Dict, List

from modules import ui_components, scripts
from scripts.infotext import parse_unit, serialize_unit
from scripts import external_code

save_symbol = "\U0001f4be"  # 💾
delete_symbol = "\U0001f5d1\ufe0f"  # 🗑️

NEW_PRESET = "New Preset"


def load_presets(preset_dir: str) -> Dict[str, str]:
    if not os.path.exists(preset_dir):
        os.makedirs(preset_dir)
        return {}

    presets = {}
    for filename in os.listdir(preset_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(preset_dir, filename), "r") as f:
                presets[filename.replace(".txt", "")] = f.read()
    return presets


class ControlNetPresetUI(object):
    preset_directory = os.path.join(scripts.basedir(), "presets")
    presets = load_presets(preset_directory)

    def __init__(self):
        self.dropdown = None
        self.save_button = None
        self.delete_button = None
        self.preset_name = None
        self.confirm_preset_name = None
        self.name_dialog = None
        self.render()

    def render(self):
        with gr.Row():
            self.dropdown = gr.Dropdown(
                label="Styles",
                show_label=False,
                elem_classes=["cnet-preset-dropdown"],
                choices=ControlNetPresetUI.dropdown_choices(),
                value=NEW_PRESET,
                tooltip="Presets",
            )
            self.save_button = ui_components.ToolButton(
                value=save_symbol,
                elem_classes=["cnet-preset-save"],
                tooltip="Save preset",
            )
            self.delete_button = ui_components.ToolButton(
                value=delete_symbol,
                elem_classes=["cnet-preset-delete"],
                tooltip="Delete preset",
            )

        with gr.Box(
            elem_classes=["popup-dialog", "cnet-preset-enter-name"],
        ) as self.name_dialog:
            self.preset_name = gr.Textbox(
                label="Preset name",
                show_label=True,
                lines=1,
                elem_classes=["cnet-preset-name"],
            )
            self.confirm_preset_name = gr.Button(
                value="Confirm", elem_classes=["cnet-preset-confirm-name"]
            )

    def register_callbacks(self, *ui_states):
        self.dropdown.change(
            fn=ControlNetPresetUI.apply_preset,
            inputs=[self.dropdown],
            outputs=[self.delete_button, *ui_states],
            show_progress=False,
        )

        def save_preset(name: str, *ui_states):
            if name == NEW_PRESET:
                return gr.update(visible=True)

            ControlNetPresetUI.save_preset(
                name, external_code.ControlNetUnit(*ui_states)
            )
            return gr.update()

        self.save_button.click(
            fn=save_preset,
            inputs=[self.dropdown, *ui_states],
            outputs=[self.name_dialog],
            show_progress=False,
        ).then(
            fn=None,
            _js=f"""
            (name) => {{
                if (name !== "{NEW_PRESET}")
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
            ControlNetPresetUI.save_preset(
                new_name, external_code.ControlNetUnit(*ui_states)
            )

        self.confirm_preset_name.click(
            fn=save_new_preset,
            inputs=[self.preset_name, *ui_states],
            outputs=None,
            show_progress=False,
        ).then(fn=None, _js="closePopup")

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
                (gr.update(),) * len(vars(external_code.ControlNetUnit()).keys()),
            )

        assert name in ControlNetPresetUI.presets
        infotext = ControlNetPresetUI.presets[name]
        unit = parse_unit(infotext)

        return gr.update(visible=True), *[
            gr.update(value=value) if value is not None else gr.update()
            for value in vars(unit).values()
        ]
