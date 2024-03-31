from .template import (
    APITestTemplate,
    girl_img,
    mask_img,
    disable_in_cq,
    get_model,
)


@disable_in_cq
def test_clip_mask_txt2img_control():
    """No mask control group."""
    assert APITestTemplate(
        "test_clip_mask_txt2img_control",
        "txt2img",
        payload_overrides={},
        unit_overrides={
            "module": "ip-adapter-auto",
            "model": get_model("ip-adapter_sd15"),
            "image": girl_img,
        },
    ).exec()


@disable_in_cq
def test_clip_mask_txt2img_experiment():
    """With mask experiment group."""
    assert APITestTemplate(
        "test_clip_mask_txt2img_experiment",
        "txt2img",
        payload_overrides={},
        unit_overrides={
            "module": "ip-adapter-auto",
            "model": get_model("ip-adapter_sd15"),
            "image": girl_img,
            "mask_image": mask_img,
        },
    ).exec()


@disable_in_cq
def test_clip_mask_img2img():
    """CLIP mask should not work in img2img inpaint."""
    assert APITestTemplate(
        "test_clip_mask_img2img",
        "img2img",
        payload_overrides={
            "init_images": [girl_img],
            "mask": mask_img,
        },
        unit_overrides={
            "module": "ip-adapter-auto",
            "model": get_model("ip-adapter_sd15"),
        },
    ).exec()
