from typing import List, Tuple, Union

import gradio as gr

from modules.processing import StableDiffusionProcessing

from scripts import external_code
from scripts.logging import logger


def field_to_displaytext(fieldname: str) -> str:
    return " ".join([word.capitalize() for word in fieldname.split("_")])


def displaytext_to_field(text: str) -> str:
    return "_".join([word.lower() for word in text.split(" ")])


def parse_value(value: str) -> Union[str, float, int, bool]:
    if value in ("True", "False"):
        return value == "True"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value  # Plain string.


def serialize_unit(unit: external_code.ControlNetUnit) -> str:
    log_value = {
        field_to_displaytext(field): getattr(unit, field)
        for field in vars(external_code.ControlNetUnit()).keys()
        if field not in ("image", "enabled") and getattr(unit, field) != -1
        # Note: exclude hidden slider values.
    }
    if not all("," not in str(v) and ":" not in str(v) for v in log_value.values()):
        logger.error(f"Unexpected tokens encountered:\n{log_value}")
        return ""
    
    return ", ".join(f"{field}: {value}" for field, value in log_value.items())


def parse_unit(text: str) -> external_code.ControlNetUnit:
    return external_code.ControlNetUnit(
        enabled=True,
        **{
            displaytext_to_field(key): parse_value(value)
            for item in text.split(",")
            for (key, value) in (item.strip().split(": "),)
        },
    )


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
        for field in vars(external_code.ControlNetUnit()).keys():
            # Exclude image for infotext.
            if field == "image":
                continue

            # Every field in ControlNetUnit should have a cooresponding
            # IOComponent in ControlNetUiGroup.
            io_component = getattr(uigroup, field)
            component_locator = f"{unit_prefix} {field}"
            self.infotext_fields.append((io_component, component_locator))
            self.paste_field_names.append(component_locator)

    @staticmethod
    def write_infotext(
        units: List[external_code.ControlNetUnit], p: StableDiffusionProcessing
    ):
        """Write infotext to `p`."""
        p.extra_generation_params.update(
            {
                Infotext.unit_prefix(i): serialize_unit(unit)
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
                for field, value in vars(parse_unit(v)).items():
                    if field == "image":
                        continue

                    assert value is not None, f"{field} == None"
                    component_locator = f"{k} {field}"
                    updates[component_locator] = value
                    logger.debug(f"InfoText: Setting {component_locator} = {value}")
            except Exception:
                logger.warn(
                    f"Failed to parse infotext, legacy format infotext is no longer supported:\n{v}"
                )

        results.update(updates)
