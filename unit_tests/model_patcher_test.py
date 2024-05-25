import pytest
import torch

from internal_controlnet.model_patcher import (
    LoRAWeight,
    ModelPatcher,
    WeightPatch,
    PatchType,
)


def test_patch_diff():
    weight = torch.tensor([1, 2, 3, 4])
    patch = WeightPatch(
        weight=torch.tensor([0.5, 0.5, 0.5, -0.5]), patch_type=PatchType.DIFF, alpha=0.5
    )
    patched_weight = patch.apply(weight)
    expected_weight = torch.tensor([1.25, 2.25, 3.25, 3.75])
    assert torch.all(torch.eq(patched_weight, expected_weight))


target_weight = weight = torch.tensor(
    [
        [1, 1],
        [1, 1],
    ]
)
down_mat = torch.tensor(
    [
        [2, 2],
        [2, 2],
    ]
)

up_mat = torch.tensor(
    [
        [4, 4],
        [4, 4],
    ]
)


@pytest.mark.parametrize(
    "test_case",
    [
        (
            LoRAWeight(
                down=down_mat,
                up=up_mat,
            ),
            target_weight + (down_mat @ up_mat),
        ),
    ],
)
def test_patch_lora(test_case):
    lora_weight, expected_weight = test_case
    patch = WeightPatch(
        weight=lora_weight,
        patch_type=PatchType.LORA,
        alpha=1.0,
    )
    patched_weight = patch.apply(weight)
    assert torch.all(
        torch.eq(patched_weight, expected_weight)
    ), f"{patched_weight} != {expected_weight}"
