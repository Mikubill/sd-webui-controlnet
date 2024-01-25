from typing import Tuple, List, NamedTuple

from internal_controlnet.external_code import ControlNetUnit

class FilteredUnits(NamedTuple):
    instant_id_units: List[Tuple[ControlNetUnit, ControlNetUnit]]
    remaining_units: List[ControlNetUnit]

def filter_instant_id_units(units: List[ControlNetUnit]) -> FilteredUnits:
    instant_id_candidates = [unit for unit in units if "instant_id" in unit.module]
    remaining_units = [unit for unit in units if "instant_id" not in unit.module]
    instant_id_models = {unit.model for unit in instant_id_candidates}
    assert len(instant_id_models) == 2, "Both InstantID models (ControlNet, IPAdapter) are required."
    model1, model2 = instant_id_models
    model_mapping = {
        unit.threshold_a: unit
        for unit in instant_id_candidates
        if unit.model == model2
    }
    instant_id_units = []
    for unit in instant_id_candidates:
        if unit.model == model2:
            continue

        if unit.threshold_a not in model_mapping:
            raise Exception(f"No pairing unit with {model2} found.")

        instant_id_units.append((unit, model_mapping[unit.threshold_a]))

    return FilteredUnits(instant_id_units, remaining_units)

