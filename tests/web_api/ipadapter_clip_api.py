import requests

from .template import (
    APITestTemplate,
    realistic_girl_face_img,
    girl_img,
    mask_img,
    disable_in_cq,
    get_model,
)


def detect_template(payload, status: int = 200):
    url = APITestTemplate.BASE_URL + "controlnet/detect"
    resp = requests.post(url, json=payload)
    assert resp.status_code == status
    if status != 200:
        return

    resp_json = resp.json()
    assert "tensor" in resp_json
    assert len(resp_json["tensor"]) == len(payload["controlnet_input_images"])
    return resp_json


@disable_in_cq
def test_ipadapter_clip_api():
    """Use previously saved CLIP output in ipadapter run."""
    resp = detect_template(
        dict(
            controlnet_input_images=[realistic_girl_face_img],
            controlnet_module="ip-adapter_clip_h",
        )
    )
    ipadapter_input = resp["tensor"]
    APITestTemplate(
        "test_ipadapter_clip_api",
        "txt2img",
        payload_overrides={},
        unit_overrides={
            "ipadapter_input": ipadapter_input,
            "model": get_model("ip-adapter_sd15"),
        },
    ).exec()


@disable_in_cq
def test_ipadapter_clip_api_mask():
    resp = detect_template(
        dict(
            controlnet_input_images=[girl_img],
            controlnet_masks=[mask_img],
            controlnet_module="ip-adapter_clip_h",
        )
    )
    ipadapter_input = resp["tensor"]
    APITestTemplate(
        "test_ipadapter_clip_api_mask",
        "txt2img",
        payload_overrides={},
        unit_overrides={
            "ipadapter_input": ipadapter_input,
            "model": get_model("ip-adapter_sd15"),
        },
    ).exec()
