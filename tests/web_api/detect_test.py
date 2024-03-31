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


# Need to allow detect of CLIP preprocessor result.
# https://github.com/Mikubill/sd-webui-controlnet/pull/2590
# FAILED extensions/sd-webui-controlnet/tests/web_api/detect_test.py::test_detect_all_modules[clip_vision] - PIL.UnidentifiedImageError: cannot identify image file <_io.BytesIO object at 0x000001589ADD1210>  
# FAILED extensions/sd-webui-controlnet/tests/web_api/detect_test.py::test_detect_all_modules[revision_clipvision] - PIL.UnidentifiedImageError: cannot identify image file <_io.BytesIO object at 0x000001589AFB00E0>
# FAILED extensions/sd-webui-controlnet/tests/web_api/detect_test.py::test_detect_all_modules[revision_ignore_prompt] - PIL.UnidentifiedImageError: cannot identify image file <_io.BytesIO object at 0x000001589AF3C9A0>
# FAILED extensions/sd-webui-controlnet/tests/web_api/detect_test.py::test_detect_all_modules[ip-adapter_clip_sd15] - PIL.UnidentifiedImageError: cannot identify image file <_io.BytesIO object at 0x000001589AF5B740>
# FAILED extensions/sd-webui-controlnet/tests/web_api/detect_test.py::test_detect_all_modules[ip-adapter_clip_sdxl_plus_vith] - PIL.UnidentifiedImageError: cannot identify image file <_io.BytesIO object at 0x000001589AF3D0D0>
# FAILED extensions/sd-webui-controlnet/tests/web_api/detect_test.py::test_detect_all_modules[ip-adapter_clip_sdxl] - PIL.UnidentifiedImageError: cannot identify image file <_io.BytesIO object at 0x00000158FF7753F0>
# FAILED extensions/sd-webui-controlnet/tests/web_api/detect_test.py::test_detect_all_modules[ip-adapter_face_id] - PIL.UnidentifiedImageError: cannot identify image file <_io.BytesIO object at 0x000001589B0414E0>
# FAILED extensions/sd-webui-controlnet/tests/web_api/detect_test.py::test_detect_all_modules[ip-adapter_face_id_plus] - PIL.UnidentifiedImageError: cannot identify image file <_io.BytesIO object at 0x000001589AEE3100>
# FAILED extensions/sd-webui-controlnet/tests/web_api/detect_test.py::test_detect_all_modules[instant_id_face_embedding] - PIL.UnidentifiedImageError: cannot identify image file <_io.BytesIO object at 0x000001589AFF6CF0>

# TODO: file issue on these failures.
# FAILED extensions/sd-webui-controlnet/tests/web_api/detect_test.py::test_detect_all_modules[depth_zoe] - assert 500 == 200
# FAILED extensions/sd-webui-controlnet/tests/web_api/detect_test.py::test_detect_all_modules[inpaint_only+lama] - assert 500 == 200
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
