import io
import os
import cv2
import base64
import functools
from typing import Dict, Any, List, Union, Literal, Optional
from pathlib import Path
import datetime
from enum import Enum
import numpy as np
import pytest
from contextlib import contextmanager

import requests
from PIL import Image


def disable_in_cq(func):
    """Skips the decorated test func in CQ run."""

    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        if APITestTemplate.is_cq_run:
            pytest.skip()
        return func(*args, **kwargs)

    return wrapped_func


PayloadOverrideType = Dict[str, Any]

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
test_result_dir = Path(__file__).parent / "results" / f"test_result_{timestamp}"
test_expectation_dir = Path(__file__).parent / "expectations"
os.makedirs(test_expectation_dir, exist_ok=True)
resource_dir = Path(__file__).parents[1] / "images"


def get_dest_dir():
    if APITestTemplate.is_set_expectation_run:
        return test_expectation_dir
    else:
        return test_result_dir


def save_base64(base64img: str, dest: Path):
    Image.open(io.BytesIO(base64.b64decode(base64img.split(",", 1)[0]))).save(dest)


def read_image(img_path: Path) -> str:
    img = cv2.imread(str(img_path))
    _, bytes = cv2.imencode(".png", img)
    encoded_image = base64.b64encode(bytes).decode("utf-8")
    return encoded_image


def read_image_dir(
    img_dir: Path, suffixes=(".png", ".jpg", ".jpeg", ".webp")
) -> List[str]:
    """Try read all images in given img_dir."""
    img_dir = str(img_dir)
    images = []
    for filename in os.listdir(img_dir):
        if filename.endswith(suffixes):
            img_path = os.path.join(img_dir, filename)
            try:
                images.append(read_image(img_path))
            except IOError:
                print(f"Error opening {img_path}")
    return images


girl_img = read_image(resource_dir / "1girl.png")
mask_img = read_image(resource_dir / "mask.png")
mask_small_img = read_image(resource_dir / "mask_small.png")
mask_left = read_image(resource_dir / "mask_left.png")
mask_right = read_image(resource_dir / "mask_right.png")
portrait_imgs = read_image_dir(resource_dir / "portrait")
realistic_girl_face_img = portrait_imgs[0]
living_room_img = read_image(resource_dir / "living_room.webp")


general_negative_prompt = """
(worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality,
((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot,
backlight,(ugly:1.331), (duplicate:1.331), (morbid:1.21), (mutilated:1.21),
(tranny:1.331), mutated hands, (poorly drawn hands:1.331), blurry, (bad anatomy:1.21),
(bad proportions:1.331), extra limbs, (missing arms:1.331), (extra legs:1.331),
(fused fingers:1.61051), (too many fingers:1.61051), (unclear eyes:1.331), bad hands,
missing fingers, extra digit, bad body, easynegative, nsfw"""


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
    is_set_expectation_run = os.environ.get("CONTROLNET_SET_EXP", "True") == "True"
    is_cq_run = os.environ.get("FORGE_CQ_TEST", "False") == "True"
    BASE_URL = "http://localhost:7860/"

    def __init__(
        self,
        name: str,
        gen_type: Union[Literal["img2img"], Literal["txt2img"]],
        payload_overrides: PayloadOverrideType,
        unit_overrides: Union[PayloadOverrideType, List[PayloadOverrideType]],
        input_image: Optional[str] = None,
    ):
        self.name = name
        self.url = APITestTemplate.BASE_URL + "sdapi/v1/" + gen_type
        self.payload = {
            **(txt2img_payload if gen_type == "txt2img" else img2img_payload),
            **payload_overrides,
        }
        if gen_type == "img2img" and input_image is not None:
            self.payload["init_images"] = [input_image]

        # CQ runs on CPU. Reduce steps/width/height to increase test speed.
        if APITestTemplate.is_cq_run:
            if "steps" not in payload_overrides:
                self.payload["steps"] = 3
            if "width" not in payload_overrides:
                self.payload["width"] = 64
            if "height" not in payload_overrides:
                self.payload["height"] = 64

        unit_overrides = (
            unit_overrides
            if isinstance(unit_overrides, (list, tuple))
            else [unit_overrides]
        )
        self.payload["alwayson_scripts"]["ControlNet"]["args"] = [
            {
                **default_unit,
                **unit_override,
                **(
                    {"image": input_image}
                    if gen_type == "txt2img" and input_image is not None
                    else {}
                ),
            }
            for unit_override in unit_overrides
        ]
        self.active_unit_count = len(unit_overrides)

    def exec(self, *args, **kwargs) -> bool:
        if APITestTemplate.is_cq_run:
            return self.exec_cq(*args, **kwargs)
        else:
            return self.exec_local(*args, **kwargs)

    def exec_cq(
        self, expected_output_num: Optional[int] = None, *args, **kwargs
    ) -> bool:
        """Execute test in CQ environment."""
        res = requests.post(url=self.url, json=self.payload)
        if res.status_code != 200:
            print(f"Unexpected status code {res.status_code}")
            return False

        response = res.json()
        if "images" not in response:
            print(response.keys())
            return False

        if expected_output_num is None:
            expected_output_num = (
                self.payload["n_iter"] * self.payload["batch_size"]
                + self.active_unit_count
            )

        if len(response["images"]) != expected_output_num:
            print(f"{len(response['images'])} != {expected_output_num}")
            return False

        return True

    def exec_local(self, result_only: bool = True, *args, **kwargs) -> bool:
        """Execute test in local environment."""
        if not APITestTemplate.is_set_expectation_run:
            os.makedirs(test_result_dir, exist_ok=True)

        failed = False

        response = requests.post(url=self.url, json=self.payload).json()
        if "images" not in response:
            print(response.keys())
            return False

        dest_dir = get_dest_dir()
        results = response["images"][:1] if result_only else response["images"]
        for i, base64image in enumerate(results):
            img_file_name = f"{self.name}_{i}.png"
            save_base64(base64image, dest_dir / img_file_name)

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
                    diff_img_path=str(
                        test_result_dir / img_file_name.replace(".png", "_diff.png")
                    ),
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

    matching_pixels = np.isclose(img1, img2, rtol=0.5, atol=1)
    similar_in_general = (matching_pixels.sum() / matching_pixels.size) >= 0.95
    return similar_in_general


@contextmanager
def console_log_context(output_file="output.txt"):
    log_encoding = "utf-8" if APITestTemplate.is_cq_run else "utf-16"
    class Context:
        def __init__(self, output_file) -> None:
            self.output_file = output_file
            self.init_line_count = 0
            with open(self.output_file, "r", encoding=log_encoding) as file:
                for _ in file:
                    self.init_line_count += 1

        def is_in_console_logs(self, expected_lines: List[str]) -> bool:
            with open(self.output_file, "r", encoding=log_encoding) as file:
                for i, line in enumerate(file):
                    if not expected_lines:
                        break
                    if i < self.init_line_count:
                        continue
                    if expected_lines[0] in line:
                        expected_lines.pop(0)
            return len(expected_lines) == 0

    yield Context(output_file)


def get_model(model_name: str) -> str:
    """Find an available model with specified model name."""
    if model_name.lower() == "none":
        return "None"

    r = requests.get(APITestTemplate.BASE_URL + "controlnet/model_list")
    result = r.json()
    if "model_list" not in result:
        raise ValueError(f"No model available\n{result}")

    candidates = [
        model for model in result["model_list"] if model_name.lower() in model.lower()
    ]

    if not candidates:
        raise ValueError("No suitable model available")

    return candidates[0]


default_unit = {
    "control_mode": "Balanced",
    "enabled": True,
    "guidance_end": 1,
    "guidance_start": 0,
    "pixel_perfect": True,
    "processor_res": 512,
    "resize_mode": "Crop and Resize",
    "threshold_a": -1,
    "threshold_b": -1,
    "weight": 1,
    "module": "canny",
    "model": get_model("sd15_canny"),
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
