import gradio as gr
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import io
import json
from PIL import Image
from typing import List

from scripts.enums import StableDiffusionVersion
from scripts.global_state import get_sd_version
from scripts.ipadapter.weight import calc_weights


INPUT_BLOCK_COLOR = "#61bdee"
MIDDLE_BLOCK_COLOR = "#e2e2e2"
OUTPUT_BLOCK_COLOR = "#dc6e55"


def get_bar_colors(
    sd_version: StableDiffusionVersion, input_color, middle_color, output_color
):
    middle_block_idx = 4 if sd_version == StableDiffusionVersion.SDXL else 6

    def get_color(idx):
        if idx < middle_block_idx:
            return input_color
        elif idx == middle_block_idx:
            return middle_color
        else:
            return output_color

    return [get_color(i) for i in range(sd_version.transformer_block_num)]


def plot_weights(
    numbers: List[float],
    colors: List[str],
):
    # Create a bar chart
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(numbers)), numbers, color=colors)
    plt.xlabel("Transformer Index")
    plt.ylabel("Weight")
    plt.legend(
        handles=[
            Patch(color=color, label=label)
            for color, label in (
                (INPUT_BLOCK_COLOR, "Input Block"),
                (MIDDLE_BLOCK_COLOR, "Middle Block"),
                (OUTPUT_BLOCK_COLOR, "Output Block"),
            )
        ],
        loc="best",
    )

    # Save the plot to a BytesIO buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    plt.close()
    buffer.seek(0)

    # Convert the buffer to a PIL image and return it
    image = Image.open(buffer)
    return image


class AdvancedWeightControl:
    def __init__(self):
        self.group = None
        self.weight_type = None
        self.weight_plot = None
        self.weight_editor = None
        self.weight_composition = None

    def render(self):
        with gr.Group(visible=False) as self.group:
            with gr.Row():
                self.weight_type = gr.Dropdown(
                    choices=[
                        "normal",
                        "ease in",
                        "ease out",
                        "ease in-out",
                        "reverse in-out",
                        "weak input",
                        "weak output",
                        "weak middle",
                        "strong middle",
                        "style transfer",
                        "composition",
                        "strong style transfer",
                        "style and composition",
                        "strong style and composition",
                    ],
                    label="Weight Type",
                    value="normal",
                )
                self.weight_composition = gr.Slider(
                    label="Composition Weight",
                    minimum=0,
                    maximum=2.0,
                    value=1.0,
                    step=0.01,
                    visible=False,
                )
                self.weight_editor = gr.Textbox(label="Weights", visible=False)

            self.weight_plot = gr.Image(
                value=None,
                label="Weight Plot",
                interactive=False,
                visible=False,
            )

    def register_callbacks(
        self,
        weight_input: gr.Slider,
        advanced_weighting: gr.State,
        control_type: gr.Radio,
        update_unit_counter: gr.Number,
    ):
        def advanced_weighting_supported(control_type: str) -> bool:
            return control_type in ("IP-Adapter", "Instant-ID")

        self.weight_type.change(
            fn=lambda weight_type: gr.update(
                visible=weight_type
                in ("style and composition", "strong style and composition")
            ),
            inputs=[self.weight_type],
            outputs=[self.weight_composition],
        )

        def update_weight_textbox(
            control_type: str,
            weight_type: str,
            weight: float,
            weight_composition: float,
        ):
            if not advanced_weighting_supported(control_type):
                return gr.update()

            sd_version = get_sd_version()
            weights = calc_weights(weight_type, weight, sd_version, weight_composition)
            return gr.update(value=str([round(w, 2) for w in weights]), visible=True)

        trigger_inputs = [self.weight_type, weight_input, self.weight_composition]
        for trigger_input in trigger_inputs:
            trigger_input.change(
                fn=update_weight_textbox,
                inputs=[
                    control_type,
                    self.weight_type,
                    weight_input,
                    self.weight_composition,
                ],
                outputs=[self.weight_editor],
            )

        def update_plot(weights_string: str):
            try:
                weights = json.loads(weights_string)
                assert isinstance(weights, list)
            except Exception:
                return gr.update(visible=False)

            sd_version = get_sd_version()
            weight_plot = plot_weights(
                weights,
                get_bar_colors(
                    sd_version,
                    input_color=INPUT_BLOCK_COLOR,
                    middle_color=MIDDLE_BLOCK_COLOR,
                    output_color=OUTPUT_BLOCK_COLOR,
                ),
            )
            return gr.update(value=weight_plot, visible=True)

        def update_advanced_weighting(weights_string: str):
            try:
                weights = json.loads(weights_string)
                assert isinstance(weights, list)
            except Exception:
                return None
            return weights

        self.weight_editor.change(
            fn=update_plot,
            inputs=[self.weight_editor],
            outputs=[self.weight_plot],
        )

        self.weight_editor.change(
            fn=update_advanced_weighting,
            inputs=[self.weight_editor],
            outputs=[advanced_weighting],
        ).then(
            fn=lambda x: gr.update(value=x + 1),
            inputs=[update_unit_counter],
            outputs=[update_unit_counter],
        )  # Necessary to flush gr.State change to unit state.

        # TODO: Expose advanced weighting control for other control types.
        def control_type_change(control_type: str, old_weights):
            supported = advanced_weighting_supported(control_type)
            if supported:
                return (
                    gr.update(visible=supported),
                    old_weights,
                    gr.update(),
                    gr.update(),
                )
            else:
                return (
                    gr.update(visible=supported),
                    None,
                    gr.update(visible=False),
                    gr.update(visible=False),
                )

        control_type.change(
            fn=control_type_change,
            inputs=[control_type, advanced_weighting],
            outputs=[
                self.group,
                advanced_weighting,
                self.weight_editor,
                self.weight_plot,
            ],
        )
