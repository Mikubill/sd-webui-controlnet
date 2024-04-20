import gradio as gr
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import io
from PIL import Image
from typing import List

from scripts.enums import StableDiffusionVersion, UnetBlockType
from scripts.global_state import get_sd_version
from scripts.ipadapter.weight import calc_weights


INPUT_BLOCK_COLOR = "#61bdee"
MIDDLE_BLOCK_COLOR = "#e2e2e2"
OUTPUT_BLOCK_COLOR = "#dc6e55"


def get_bar_colors(
    sd_version: StableDiffusionVersion, input_color, middle_color, output_color
):
    return [
        (
            input_color
            if transformer_id.block_type == UnetBlockType.INPUT
            else (
                middle_color
                if transformer_id.block_type == UnetBlockType.MIDDLE
                else output_color
            )
        )
        for transformer_id in sd_version.transformer_ids.to_list()
    ]


def plot_weights(
    numbers: List[float],
    colors: List[str],
):
    # Create a bar chart
    plt.figure(figsize=(8, 4))
    bars = plt.bar(range(len(numbers)), numbers, color=colors)
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
                    ],
                    label="Weight Type",
                    value="normal",
                )
                self.weight_composition = gr.Slider(
                    label="Composition Weight",
                    minimum=0,
                    maximum=2.0,
                    value=0.0,
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
        self.weight_type.change(
            fn=lambda weight_type: gr.update(
                visible=weight_type
                in ("style and composition", "strong style and composition")
            ),
            inputs=[self.weight_type],
            outputs=[self.weight_composition],
        )

        def weight_update(weight_type: str, weight: float, weight_composition: float):
            sd_version = get_sd_version()
            weights = calc_weights(weight_type, weight, sd_version, weight_composition)
            weight_plot = plot_weights(
                weights,
                get_bar_colors(
                    sd_version,
                    input_color=INPUT_BLOCK_COLOR,
                    middle_color=MIDDLE_BLOCK_COLOR,
                    output_color=OUTPUT_BLOCK_COLOR,
                ),
            )
            return (
                gr.update(value=str(weights), visible=True),
                gr.update(value=weight_plot, visible=True),
                weights,
            )

        trigger_inputs = [self.weight_type, weight_input, self.weight_composition]
        for trigger_input in trigger_inputs:
            trigger_input.change(
                fn=weight_update,
                inputs=[self.weight_type, weight_input, self.weight_composition],
                outputs=[
                    self.weight_editor,
                    self.weight_plot,
                    advanced_weighting,
                ],
            ).then(
                fn=lambda x: gr.update(value=x + 1),
                inputs=[update_unit_counter],
                outputs=[update_unit_counter],
            )  # Necessary to flush gr.State change to unit state.

        # TODO: Expose advanced weighting control for other control types.
        control_type.change(
            fn=lambda t: gr.update(visible=t in ("IP-Adapter", "Instant-ID")),
            inputs=[control_type],
            outputs=[self.group],
        )
