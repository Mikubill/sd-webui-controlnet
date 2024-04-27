from .template import (
    APITestTemplate,
    girl_img,
    realistic_girl_face_img,
    living_room_img,
    mask_left,
    mask_right,
    disable_in_cq,
    get_model,
)


@disable_in_cq
def test_effective_region_ipadapter():
    assert APITestTemplate(
        "test_effective_region_ipadapter",
        "txt2img",
        payload_overrides={
            "prompt": "(masterpiece: 1.3), (highres: 1.3), best quality, (2girls: 1.3)",
            "width": 768,
            "height": 512,
            "steps": 20,
        },
        unit_overrides=[
            {
                "module": "ip-adapter-auto",
                "model": get_model("ip-adapter_sd15"),
                "image": girl_img,
                "effective_region_mask": mask_left,
            },
            {
                "module": "ip-adapter-auto",
                "model": get_model("ip-adapter_sd15"),
                "image": realistic_girl_face_img,
                "effective_region_mask": mask_right,
            },
        ],
    ).exec()


@disable_in_cq
def test_effective_region_depth():
    assert APITestTemplate(
        "test_effective_region_depth",
        "txt2img",
        payload_overrides={
            "prompt": "(masterpiece: 1.3), (highres: 1.3), best quality, A cozy living room",
            "width": 768,
            "height": 512,
        },
        unit_overrides={
            "module": "depth_midas",
            "model": get_model("control_v11f1p_sd15_depth"),
            "image": living_room_img,
            "effective_region_mask": mask_left,
        },
    ).exec()
