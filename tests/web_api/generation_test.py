import pytest

from .template import (
    APITestTemplate,
    girl_img,
    mask_img,
    disable_in_cq,
    get_model,
    console_log_context,
)


@pytest.mark.parametrize("gen_type", ["img2img", "txt2img"])
def test_no_unit(gen_type):
    assert APITestTemplate(
        f"test_no_unit{gen_type}",
        gen_type,
        payload_overrides={},
        unit_overrides=[],
        input_image=girl_img,
    ).exec()


@pytest.mark.parametrize("gen_type", ["img2img", "txt2img"])
def test_unrecognized_param(gen_type):
    assert APITestTemplate(
        f"test_unrecognized_param_{gen_type}",
        gen_type,
        payload_overrides={},
        unit_overrides={
            "foo": True,
            "is_ui": False,
        },
        input_image=girl_img,
    ).exec()


@pytest.mark.parametrize("gen_type", ["img2img", "txt2img"])
def test_multiple_iter(gen_type):
    assert APITestTemplate(
        f"test_multiple_iter{gen_type}",
        gen_type,
        payload_overrides={"n_iter": 2},
        unit_overrides={},
        input_image=girl_img,
    ).exec()


@pytest.mark.parametrize("gen_type", ["img2img", "txt2img"])
def test_batch_size(gen_type):
    assert APITestTemplate(
        f"test_batch_size{gen_type}",
        gen_type,
        payload_overrides={"batch_size": 2},
        unit_overrides={},
        input_image=girl_img,
    ).exec()


@pytest.mark.parametrize("gen_type", ["img2img", "txt2img"])
def test_2_units(gen_type):
    assert APITestTemplate(
        f"test_2_units{gen_type}",
        gen_type,
        payload_overrides={},
        unit_overrides=[{}, {}],
        input_image=girl_img,
    ).exec()


@pytest.mark.parametrize("gen_type", ["img2img", "txt2img"])
def test_preprocessor(gen_type):
    assert APITestTemplate(
        f"test_preprocessor{gen_type}",
        gen_type,
        payload_overrides={},
        unit_overrides={"module": "canny"},
        input_image=girl_img,
    ).exec()


@pytest.mark.parametrize("param_name", ("processor_res", "threshold_a", "threshold_b"))
@pytest.mark.parametrize("gen_type", ["img2img", "txt2img"])
def test_invalid_param(gen_type, param_name):
    with console_log_context() as log_context:
        assert APITestTemplate(
            f"test_invalid_param{(gen_type, param_name)}",
            gen_type,
            payload_overrides={},
            unit_overrides={param_name: -100},
            input_image=girl_img,
        ).exec()
        number = "-100" if param_name == "processor_res" else "-100.0"
        assert log_context.is_in_console_logs(
            [
                f"[canny.{param_name}] Invalid value({number}), using default value",
            ]
        )


@pytest.mark.parametrize("save_map", [True, False])
@pytest.mark.parametrize("gen_type", ["img2img", "txt2img"])
def test_save_map(gen_type, save_map):
    assert APITestTemplate(
        f"test_save_map{(gen_type, save_map)}",
        gen_type,
        payload_overrides={},
        unit_overrides={"save_detected_map": save_map},
        input_image=girl_img,
    ).exec(expected_output_num=2 if save_map else 1)


def test_ip_adapter_face():
    assert APITestTemplate(
        "test_ip_adapter_face",
        "txt2img",
        payload_overrides={},
        unit_overrides={
            "module": "ip-adapter_clip_sd15",
            "model": get_model("ip-adapter-plus-face_sd15"),
        },
        input_image=girl_img,
    ).exec()


def test_ip_adapter_fullface():
    assert APITestTemplate(
        "test_ip_adapter_fullface",
        "txt2img",
        payload_overrides={},
        unit_overrides={
            "module": "ip-adapter_clip_sd15",
            "model": get_model("ip-adapter-full-face_sd15"),
        },
        input_image=girl_img,
    ).exec()


def test_control_lora():
    assert APITestTemplate(
        "test_control_lora",
        "txt2img",
        payload_overrides={},
        unit_overrides={
            "module": "canny",
            "model": get_model("control_lora_rank128_v11p_sd15_canny"),
        },
        input_image=girl_img,
    ).exec()


def test_t2i_adapter():
    assert APITestTemplate(
        "test_t2i_adapter",
        "txt2img",
        payload_overrides={},
        unit_overrides={
            "module": "canny",
            "model": get_model("t2iadapter_canny_sd15v2"),
        },
        input_image=girl_img,
    ).exec()


def test_reference():
    assert APITestTemplate(
        "test_reference",
        "txt2img",
        payload_overrides={},
        unit_overrides={
            "module": "reference_only",
            "model": "None",
        },
        input_image=girl_img,
    ).exec(result_only=False)


def test_advanced_weighting():
    assert APITestTemplate(
        "test_advanced_weighting",
        "txt2img",
        payload_overrides={},
        unit_overrides={"advanced_weighting": [0.75] * 13},  # SD1.5
        input_image=girl_img,
    ).exec()


def test_hr_option():
    assert APITestTemplate(
        "test_hr_option",
        "txt2img",
        payload_overrides={
            "enable_hr": True,
            "denoising_strength": 0.75,
        },
        unit_overrides={"hr_option": "Both"},
        input_image=girl_img,
    ).exec(expected_output_num=3)


def test_hr_option_default():
    """In non-hr run, hr_option should be ignored."""
    assert APITestTemplate(
        "test_hr_option_default",
        "txt2img",
        payload_overrides={"enable_hr": False},
        unit_overrides={"hr_option": "Both"},
        input_image=girl_img,
    ).exec(expected_output_num=2)


@disable_in_cq
def test_masked_controlnet_txt2img():
    assert APITestTemplate(
        f"test_masked_controlnet_txt2img",
        "txt2img",
        payload_overrides={},
        unit_overrides={
            "image": girl_img,
            "mask_image": mask_img,
        },
    ).exec()


@disable_in_cq
def test_masked_controlnet_img2img():
    assert APITestTemplate(
        f"test_masked_controlnet_img2img",
        "img2img",
        payload_overrides={
            "init_images": [girl_img],
        },
        # Note: Currently you must give ControlNet unit input image to specify
        # mask.
        # TODO: Fix this for img2img.
        unit_overrides={
            "image": girl_img,
            "mask_image": mask_img,
        },
    ).exec()


@disable_in_cq
def test_txt2img_inpaint():
    assert APITestTemplate(
        "txt2img_inpaint",
        "txt2img",
        payload_overrides={},
        unit_overrides={
            "image": girl_img,
            "mask_image": mask_img,
            "model": get_model("v11p_sd15_inpaint"),
            "module": "inpaint_only",
        },
    ).exec()


@disable_in_cq
def test_img2img_inpaint():
    assert APITestTemplate(
        "img2img_inpaint",
        "img2img",
        payload_overrides={
            "init_images": [girl_img],
            "mask": mask_img,
        },
        unit_overrides={
            "model": get_model("v11p_sd15_inpaint"),
            "module": "inpaint_only",
        },
    ).exec()


@disable_in_cq
def test_lama_outpaint():
    assert APITestTemplate(
        "txt2img_lama_outpaint",
        "txt2img",
        payload_overrides={
            "width": 768,
            "height": 768,
        },
        # Outpaint should not need a mask.
        unit_overrides={
            "image": girl_img,
            "model": get_model("v11p_sd15_inpaint"),
            "module": "inpaint_only+lama",
            "resize_mode": "Resize and Fill",  # OUTER_FIT
        },
    ).exec(result_only=False)


@disable_in_cq
def test_ip_adapter_auto():
    with console_log_context() as log_context:
        assert APITestTemplate(
            "txt2img_ip_adapter_auto",
            "txt2img",
            payload_overrides={},
            unit_overrides={
                "image": girl_img,
                "model": get_model("ip-adapter_sd15"),
                "module": "ip-adapter-auto",
            },
        ).exec()

        assert log_context.is_in_console_logs(["ip-adapter-auto => ip-adapter_clip_h"])
