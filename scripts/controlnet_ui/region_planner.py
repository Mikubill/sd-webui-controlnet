import gradio as gr

from modules import shared
from modules.ui_components import InputAccordion


class RegionPlanner:
    """Let user specify effective region of unit."""

    def __init__(self, unit_num: int) -> None:
        self.unit_num = unit_num
        self.enabled = None
        self.editor = None

    @property
    def feature_enabled(self) -> bool:
        return not shared.opts.data.get("controlnet_disable_region_planner", False)

    def render(self):
        if not self.feature_enabled:
            return

        with InputAccordion(
            value=False,
            label="Region Planner",
        ) as self.enabled:
            gr.HTML(
                """
                <div class="cnet-region-planner"></div>
                <div class="cnet-region-planner-snapshot-canvas" hidden></div>
                """
            )

    def register_callbacks(self):
        if not self.feature_enabled:
            return
