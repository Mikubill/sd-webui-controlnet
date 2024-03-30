import pytest
import requests
from typing import List

from .template import (
    APITestTemplate,
    realistic_girl_face_img,
    save_base64,
    get_dest_dir,
    disable_in_cq,
)


def get_modules() -> List[str]:
    return requests.get(APITestTemplate.BASE_URL + "controlnet/module_list").json()[
        "module_list"
    ]


def detect_template(payload, output_name: str, status: int = 200):
    url = APITestTemplate.BASE_URL + "controlnet/detect"
    resp = requests.post(url, json=payload)
    assert resp.status_code == status
    if status != 200:
        return

    resp_json = resp.json()
    assert "images" in resp_json
    assert len(resp_json["images"]) == len(payload["controlnet_input_images"])
    if not APITestTemplate.is_cq_run:
        for i, img in enumerate(resp_json["images"]):
            if img == "Detect result is not image":
                continue
            dest = get_dest_dir() / f"{output_name}_{i}.png"
            save_base64(img, dest)
    return resp_json


@disable_in_cq
@pytest.mark.parametrize("module", get_modules())
def test_detect_all_modules(module: str):
    payload = dict(
        controlnet_input_images=[realistic_girl_face_img],
        controlnet_module=module,
    )
    detect_template(payload, f"detect_{module}")


def test_detect_simple():
    detect_template(
        dict(
            controlnet_input_images=[realistic_girl_face_img],
            controlnet_module="canny",  # Canny does not require model download.
        ),
        "simple_detect",
    )


def test_detect_multiple_inputs():
    detect_template(
        dict(
            controlnet_input_images=[realistic_girl_face_img, realistic_girl_face_img],
            controlnet_module="canny",  # Canny does not require model download.
        ),
        "multiple_inputs_detect",
    )


def test_detect_with_invalid_module():
    detect_template({"controlnet_module": "INVALID"}, "invalid module", 422)


def test_detect_with_no_input_images():
    detect_template({"controlnet_input_images": []}, "invalid module", 422)
