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
        self.modal = ModalInterface(
            f'<iframe src="{PHOTOPEA_URL}"></iframe>',
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
