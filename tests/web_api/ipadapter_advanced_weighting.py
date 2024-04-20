from .template import (
    APITestTemplate,
    realistic_girl_face_img,
    disable_in_cq,
    get_model,
)


@disable_in_cq
def test_ipadapter_advanced_weighting():
    weights = [0.0] * 16  # 16 weights for SD15 / 11 weights for SDXL
    # SD15 composition
    weights[4] = 0.25
    weights[5] = 1.0

    APITestTemplate(
        "test_ipadapter_advanced_weighting",
        "txt2img",
        payload_overrides={
            "width": 512,
            "height": 512,
        },
        unit_overrides={
            "image": realistic_girl_face_img,
            "module": "ip-adapter-auto",
            "model": get_model("ip-adapter_sd15"),
            "advanced_weighting": weights,
        },
    ).exec()

    APITestTemplate(
        "test_ipadapter_advanced_weighting_ref",
        "txt2img",
        payload_overrides={
            "width": 512,
            "height": 512,
        },
        unit_overrides={
            "image": realistic_girl_face_img,
            "module": "ip-adapter-auto",
            "model": get_model("ip-adapter_sd15"),
        },
    ).exec()
