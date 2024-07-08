# From https://github.com/cubiq/ComfyUI_IPAdapter_plus

from ..enums import StableDiffusionVersion, TransformerIDResult


def calc_weights(
    weight_type: str,
    weight: float,
    sd_version: StableDiffusionVersion,
    weight_composition: float = 0.0,
):
    layers = sd_version.transformer_block_num
    return [
        _calc_weight(
            weight_type,
            weight,
            sd_version,
            t_idx,
            weight_composition,
        )
        for t_idx in range(layers)
    ]


def _calc_weight(
    weight_type: str,
    weight: float,
    sd_version: StableDiffusionVersion,
    t_idx: int,
    weight_composition: float = 0.0,
):
    is_sdxl = sd_version == StableDiffusionVersion.SDXL
    layers = sd_version.transformer_block_num
    ids: TransformerIDResult = sd_version.transformer_ids
    block_type = [i for i in ids.to_list() if i.transformer_index == t_idx][
        0
    ].block_type.value
    if weight_type == "normal":
        return weight
    elif weight_type == "ease in":
        weight = weight * (0.05 + 0.95 * (1 - t_idx / layers))
    elif weight_type == "ease out":
        weight = weight * (0.05 + 0.95 * (t_idx / layers))
    elif weight_type == "ease in-out":
        weight = weight * (0.05 + 0.95 * (1 - abs(t_idx - (layers / 2)) / (layers / 2)))
    elif weight_type == "reverse in-out":
        weight = weight * (0.05 + 0.95 * (abs(t_idx - (layers / 2)) / (layers / 2)))
    elif weight_type == "weak input":
        weight = weight * 0.2 if block_type == "input" else weight
    elif weight_type == "weak middle":
        weight = weight * 0.2 if block_type == "middle" else weight
    elif weight_type == "weak output":
        weight = weight * 0.2 if block_type == "output" else weight
    elif weight_type == "strong middle":
        weight = (
            weight * 0.2
            if (block_type == "input" or block_type == "output")
            else weight
        )
    elif weight_type.startswith("style transfer"):
        weight = (
            {6: weight}
            if is_sdxl
            else {
                0: weight,
                1: weight,
                2: weight,
                3: weight,
                9: weight,
                10: weight,
                11: weight,
                12: weight,
                13: weight,
                14: weight,
                15: weight,
            }
        )
    elif weight_type.startswith("composition"):
        weight = {3: weight} if is_sdxl else {4: weight * 0.25, 5: weight}
    elif weight_type == "strong style transfer":
        if is_sdxl:
            weight = {
                0: weight,
                1: weight,
                2: weight,
                4: weight,
                5: weight,
                6: weight,
                7: weight,
                8: weight,
                9: weight,
                10: weight,
            }
        else:
            weight = {
                0: weight,
                1: weight,
                2: weight,
                3: weight,
                6: weight,
                7: weight,
                8: weight,
                9: weight,
                10: weight,
                11: weight,
                12: weight,
                13: weight,
                14: weight,
                15: weight,
            }
    elif weight_type == "style and composition":
        if is_sdxl:
            weight = {3: weight_composition, 6: weight}
        else:
            weight = {
                0: weight,
                1: weight,
                2: weight,
                3: weight,
                4: weight_composition * 0.25,
                5: weight_composition,
                9: weight,
                10: weight,
                11: weight,
                12: weight,
                13: weight,
                14: weight,
                15: weight,
            }
    elif weight_type == "strong style and composition":
        if is_sdxl:
            weight = {
                0: weight,
                1: weight,
                2: weight,
                3: weight_composition,
                4: weight,
                5: weight,
                6: weight,
                7: weight,
                8: weight,
                9: weight,
                10: weight,
            }
        else:
            weight = {
                0: weight,
                1: weight,
                2: weight,
                3: weight,
                4: weight_composition,
                5: weight_composition,
                6: weight,
                7: weight,
                8: weight,
                9: weight,
                10: weight,
                11: weight,
                12: weight,
                13: weight,
                14: weight,
                15: weight,
            }
    else:
        raise Exception("Unrecognized weight type!")

    if isinstance(weight, dict):
        return weight.get(t_idx, 0.0)
    else:
        return weight
