def calc_weight(
    weight_type: str,
    weight: float,
    weight_composition: float,
    is_sdxl: bool,
    t_idx,
    block_type: str,
):
    layers = 11 if is_sdxl else 16

    if weight_type == "ease in":
        weight = weight * (0.05 + 0.95 * (1 - t_idx / layers))
    elif weight_type == "ease out":
        weight = weight * (0.05 + 0.95 * (t_idx / layers))
    elif weight_type == "ease in-out":
        weight = weight * (0.05 + 0.95 * (1 - abs(t_idx - (layers / 2)) / (layers / 2)))
    elif weight_type == "reverse in-out":
        weight = weight * (0.05 + 0.95 * (abs(t_idx - (layers / 2)) / (layers / 2)))
    elif weight_type == "weak input" and block_type == "input":
        weight = weight * 0.2
    elif weight_type == "weak middle" and block_type == "middle":
        weight = weight * 0.2
    elif weight_type == "weak output" and block_type == "output":
        weight = weight * 0.2
    elif weight_type == "strong middle" and (
        block_type == "input" or block_type == "output"
    ):
        weight = weight * 0.2
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

    if isinstance(weight, dict):
        return weight.get(t_idx, 0.0)
    else:
        return weight


import json

if __name__ == "__main__":
    result = {}
    for is_sdxl in (True, False):
        layers = 11 if is_sdxl else 16
        for t_idx in range(layers):
            
