import io
import os
import cv2
import base64
from typing import Dict, Any, List, Union, Literal
from pathlib import Path
import datetime
from enum import Enum
import numpy as np

import requests
from PIL import Image


PayloadOverrideType = Dict[str, Any]

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
test_result_dir = Path(__file__).parent / "results" / f"test_result_{timestamp}"
test_expectation_dir = Path(__file__).parent / "expectations"
os.makedirs(test_expectation_dir, exist_ok=True)
resource_dir = Path(__file__).parents[2] / "images"


def read_image(img_path: Path) -> str:
    img = cv2.imread(str(img_path))
    _, bytes = cv2.imencode(".png", img)
    encoded_image = base64.b64encode(bytes).decode("utf-8")
    return encoded_image


girl_img = read_image(resource_dir / "1girl.png")
mask_img = read_image(resource_dir / "mask.png")
mask_small_img = read_image(resource_dir / "mask_small.png")


class StableDiffusionVersion(Enum):
    """The version family of stable diffusion model."""

    UNKNOWN = 0
    SD1x = 1
    SD2x = 2
    SDXL = 3


sd_version = StableDiffusionVersion(
    int(os.environ.get("CONTROLNET_TEST_SD_VERSION", StableDiffusionVersion.SD1x.value))
)

is_full_coverage = os.environ.get("CONTROLNET_TEST_FULL_COVERAGE", None) is not None


class APITestTemplate:
    is_set_expectation_run = False

    def __init__(
        self,
        name: str,
        gen_type: Union[Literal["img2img"], Literal["txt2img"]],
        payload_overrides: PayloadOverrideType,
        unit_overrides: Union[PayloadOverrideType, List[PayloadOverrideType]],
    ):
        self.name = name
        self.url = "http://localhost:7860/sdapi/v1/" + gen_type
        self.payload = {
            **(txt2img_payload if gen_type == "txt2img" else img2img_payload),
            **payload_overrides,
        }
        unit_overrides = (
            unit_overrides
            if isinstance(unit_overrides, (list, tuple))
            else [unit_overrides]
        )
        self.payload["alwayson_scripts"]["ControlNet"]["args"] = [
            {
                **default_unit,
                **unit_override,
            }
            for unit_override in unit_overrides
        ]

    def exec(self) -> bool:
        if not APITestTemplate.is_set_expectation_run:
            os.makedirs(test_result_dir, exist_ok=True)

        failed = False

        response = requests.post(url=self.url, json=self.payload).json()
        if "images" not in response:
            print(response)
            return False

        dest_dir = (
            test_expectation_dir
            if APITestTemplate.is_set_expectation_run
            else test_result_dir
        )
        for i, base64image in enumerate(response["images"]):
            img_file_name = f"{self.name}_{i}.png"
            Image.open(io.BytesIO(base64.b64decode(base64image.split(",", 1)[0]))).save(
                dest_dir / img_file_name
            )

            if not APITestTemplate.is_set_expectation_run:
                try:
                    img1 = cv2.imread(os.path.join(test_expectation_dir, img_file_name))
                    img2 = cv2.imread(os.path.join(test_result_dir, img_file_name))
                except Exception as e:
                    print(f"Get exception reading imgs: {e}")
                    failed = True
                    continue

                if img1 is None:
                    print(f"Warn: No expectation file found {img_file_name}.")
                    continue

                if not expect_same_image(
                    img1,
                    img2,
                    diff_img_path=str(test_result_dir
                    / img_file_name.replace(".png", "_diff.png")),
                ):
                    failed = True
        return not failed


def expect_same_image(img1, img2, diff_img_path: str) -> bool:
    # Calculate the difference between the two images
    diff = cv2.absdiff(img1, img2)

    # Set a threshold to highlight the different pixels
    threshold = 30
    diff_highlighted = np.where(diff > threshold, 255, 0).astype(np.uint8)

    # Assert that the two images are similar within a tolerance
    similar = np.allclose(img1, img2, rtol=0.5, atol=1)
    if not similar:
        # Save the diff_highlighted image to inspect the differences
        cv2.imwrite(diff_img_path, diff_highlighted)

    return similar


default_unit = {
    "control_mode": 0,
    "enabled": True,
    "guidance_end": 1,
    "guidance_start": 0,
    "low_vram": False,
    "pixel_perfect": True,
    "processor_res": 512,
    "resize_mode": 1,
    "threshold_a": 64,
    "threshold_b": 64,
    "weight": 1,
}

img2img_payload = {
    "batch_size": 1,
    "cfg_scale": 7,
    "height": 768,
    "width": 512,
    "n_iter": 1,
    "steps": 10,
    "sampler_name": "Euler a",
    "prompt": "(masterpiece: 1.3), (highres: 1.3), best quality,",
    "negative_prompt": "",
    "seed": 42,
    "seed_enable_extras": False,
    "seed_resize_from_h": 0,
    "seed_resize_from_w": 0,
    "subseed": -1,
    "subseed_strength": 0,
    "override_settings": {},
    "override_settings_restore_afterwards": False,
    "do_not_save_grid": False,
    "do_not_save_samples": False,
    "s_churn": 0,
    "s_min_uncond": 0,
    "s_noise": 1,
    "s_tmax": None,
    "s_tmin": 0,
    "script_args": [],
    "script_name": None,
    "styles": [],
    "alwayson_scripts": {"ControlNet": {"args": [default_unit]}},
    "denoising_strength": 0.75,
    "initial_noise_multiplier": 1,
    "inpaint_full_res": 0,
    "inpaint_full_res_padding": 32,
    "inpainting_fill": 1,
    "inpainting_mask_invert": 0,
    "mask_blur_x": 4,
    "mask_blur_y": 4,
    "mask_blur": 4,
    "resize_mode": 0,
}

txt2img_payload = {
    "alwayson_scripts": {"ControlNet": {"args": [default_unit]}},
    "batch_size": 1,
    "cfg_scale": 7,
    "comments": {},
    "disable_extra_networks": False,
    "do_not_save_grid": False,
    "do_not_save_samples": False,
    "enable_hr": False,
    "height": 768,
    "hr_negative_prompt": "",
    "hr_prompt": "",
    "hr_resize_x": 0,
    "hr_resize_y": 0,
    "hr_scale": 2,
    "hr_second_pass_steps": 0,
    "hr_upscaler": "Latent",
    "n_iter": 1,
    "negative_prompt": "",
    "override_settings": {},
    "override_settings_restore_afterwards": True,
    "prompt": "(masterpiece: 1.3), (highres: 1.3), best quality,",
    "restore_faces": False,
    "s_churn": 0.0,
    "s_min_uncond": 0,
    "s_noise": 1.0,
    "s_tmax": None,
    "s_tmin": 0.0,
    "sampler_name": "Euler a",
    "script_args": [],
    "script_name": None,
    "seed": 42,
    "seed_enable_extras": True,
    "seed_resize_from_h": -1,
    "seed_resize_from_w": -1,
    "steps": 10,
    "styles": [],
    "subseed": -1,
    "subseed_strength": 0,
    "tiling": False,
    "width": 512,
}
