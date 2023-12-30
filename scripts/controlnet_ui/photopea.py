import gradio as gr

from scripts.controlnet_ui.modal import ModalInterface


PHOTOPEA_URL = "https://www.photopea.com/"
PHOTOPEA_LOGO = "https://www.photopea.com/promo/icon256.png"


class Photopea(object):
    def __init__(self) -> None:
        self.modal = None
        self.triggers = []
        self.render_editor()

    def render_editor(self):
        """Render the editor modal."""
        with gr.Group(elem_classes=["cnet-photopea-edit"]):
            self.modal = ModalInterface(
                f"""
                <div class="photopea-button-group">
                    <button class="photopea-button photopea-fetch">Fetch from ControlNet</button>
                    <button class="photopea-button photopea-send">Send to ControlNet</button>
                </div>
                <iframe class="photopea-iframe" src="{PHOTOPEA_URL}"></iframe>
                """,
                open_button_text="Edit",
                open_button_classes=["cnet-photopea-main-trigger"],
                open_button_extra_attrs="hidden",
            ).create_modal(visible=True)

    def render_child_trigger(self):
        self.triggers.append(
            gr.HTML(
                f"""<div class="cnet-photopea-child-trigger">
                Edit
                <img src="{PHOTOPEA_LOGO}" style="width: 0.75rem; height 0.75rem; margin-left: 2px;"/>
            </div>"""
            )
        )

    def attach_photopea_output(self, generated_image: gr.Image):
        """Called in ControlNetUiGroup to attach preprocessor preview image Gradio element
        as the photopea output. If the front-end directly change the img HTML element's src
        to reflect the edited image result from photopea, the backend won't be notified.

        In this method we let the front-end upload the result image an invisible gr.Image
        instance and mirrors the value to preprocessor preview gr.Image. This is because
        the generated image gr.Image instance is inferred to be an output image by Gradio
        and has no ability to accept image upload directly.
        
        Arguments:
            generated_image: preprocessor result Gradio Image output element.
        
        Returns:
            None
        """
        output = gr.Image(
            visible=False,
            source="upload",
            type="numpy",
            elem_classes=[f"cnet-photopea-output"],
        )

        output.upload(
            fn=lambda img: img,
            inputs=[output],
            outputs=[generated_image],
        )
