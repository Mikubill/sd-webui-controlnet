from typing import List, Tuple
from enum import Enum
import gradio as gr

from modules.processing import StableDiffusionProcessing

from internal_controlnet.external_code import ControlNetUnit
from scripts.logging import logger


class Infotext(object):
    def __init__(self) -> None:
        self.infotext_fields: List[Tuple[gr.components.IOComponent, str]] = []
        self.paste_field_names: List[str] = []

    @staticmethod
    def unit_prefix(unit_index: int) -> str:
        return f"ControlNet {unit_index}"

    def register_unit(self, unit_index: int, uigroup) -> None:
        """Register the unit's UI group. By regsitering the unit, A1111 will be
        able to paste values from infotext to IOComponents.

        Args:
            unit_index: The index of the ControlNet unit
            uigroup: The ControlNetUiGroup instance that contains all gradio
                     iocomponents.
        """
        unit_prefix = Infotext.unit_prefix(unit_index)
        for field in ControlNetUnit.infotext_fields():
            # Every field in ControlNetUnit should have a cooresponding
            # IOComponent in ControlNetUiGroup.
            io_component = getattr(uigroup, field)
            component_locator = f"{unit_prefix} {field}"
            self.infotext_fields.append((io_component, component_locator))
            self.paste_field_names.append(component_locator)

    @staticmethod
    def write_infotext(units: List[ControlNetUnit], p: StableDiffusionProcessing):
        """Write infotext to `p`."""
        p.extra_generation_params.update(
            {
                Infotext.unit_prefix(i): unit.serialize()
                for i, unit in enumerate(units)
                if unit.enabled
            }
        )

    @staticmethod
    def on_infotext_pasted(infotext: str, results: dict) -> None:
        """Parse ControlNet infotext string and write result to `results` dict."""
        updates = {}
        for k, v in results.items():
            if not k.startswith("ControlNet"):
                continue

            assert isinstance(v, str), f"Expect string but got {v}."
            try:
                for field, value in vars(ControlNetUnit.parse(v)).items():
                    if field not in ControlNetUnit.infotext_fields():
                        continue
                    if value is None:
                        logger.debug(
                            f"InfoText: Skipping {field} because value is None."
                        )
                        continue

                    component_locator = f"{k} {field}"
                    if isinstance(value, Enum):
                        value = value.value

                    updates[component_locator] = value
                    logger.debug(f"InfoText: Setting {component_locator} = {value}")
            except Exception as e:
                logger.warn(
                    f"Failed to parse infotext, legacy format infotext is no longer supported:\n{v}\n{e}"
                )

        results.update(updates)
