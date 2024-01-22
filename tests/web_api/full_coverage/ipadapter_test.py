import unittest
import pytest
from typing import NamedTuple, Optional

from .template import (
    sd_version,
    StableDiffusionVersion,
    is_full_coverage,
    APITestTemplate,
    portrait_imgs,
    realistic_girl_face_img,
    general_negative_prompt,
)


class AdapterSetting(NamedTuple):
    module: str
    model: str
    lora: Optional[str] = None


# Used to fix pose for better comparison between different settings.
openpose_unit = {
    "module": "openpose",
    "model": (
        "control_v11p_sd15_openpose [cab727d4]"
        if sd_version != StableDiffusionVersion.SDXL
        else "kohya_controllllite_xl_openpose_anime [7e5349e5]"
    ),
    "image": realistic_girl_face_img,
    "weight": 0.8,
}
base_prompt = "1girl, simple background, (white_background: 1.2), portrait"

sd15_face_id = AdapterSetting(
    "ip-adapter_face_id",
    "ip-adapter-faceid_sd15 [0a1757e9]",
    "ip-adapter-faceid_sd15_lora",
)
sd15_face_id_plus = AdapterSetting(
    "ip-adapter_face_id_plus",
    "ip-adapter-faceid-plus_sd15 [d86a490f]",
    "ip-adapter-faceid-plus_sd15_lora",
)
sd15_face_id_plus_v2 = AdapterSetting(
    "ip-adapter_face_id_plus",
    "ip-adapter-faceid-plusv2_sd15 [6e14fc1a]",
    "ip-adapter-faceid-plusv2_sd15_lora",
)
sd15_face_id_portrait = AdapterSetting(
    "ip-adapter_face_id",
    "ip-adapter-faceid-portrait_sd15 [b2609049]",
)
sdxl_face_id = AdapterSetting(
    "ip-adapter_face_id",
    "ip-adapter-faceid_sdxl [59ee31a3]",
    "ip-adapter-faceid_sdxl_lora",
)


class TestIPAdapterFullCoverage(unittest.TestCase):
    def setUp(self):
        if not is_full_coverage:
            pytest.skip()

        if sd_version == StableDiffusionVersion.SDXL:
            self.settings = [sdxl_face_id]
        else:
            self.settings = [
                sd15_face_id,
                sd15_face_id_plus,
                sd15_face_id_plus_v2,
                sd15_face_id_portrait,
            ]

    def test_face_id(self):
        for module, model, lora in self.settings:
            name = str((module, model, lora))
            with self.subTest(name=name):
                self.assertTrue(
                    APITestTemplate(
                        name,
                        "txt2img",
                        payload_overrides={
                            "prompt": f"{base_prompt} <lora:{lora}:0.6>",
                            "negative_prompt": general_negative_prompt,
                            "steps": 20,
                            "width": 512,
                            "height": 512,
                        },
                        unit_overrides=[
                            {
                                "module": module,
                                "model": model,
                                "image": realistic_girl_face_img,
                            },
                            openpose_unit,
                        ],
                    ).exec()
                )

    def test_face_id_multi_inputs(self):
        for module, model, lora in self.settings:
            name = "multi_inputs" + str((module, model, lora))
            with self.subTest(name=name):
                self.assertTrue(
                    APITestTemplate(
                        name=name,
                        gen_type="txt2img",
                        payload_overrides={
                            "prompt": base_prompt,
                            "negative_prompt": general_negative_prompt,
                            "steps": 20,
                            "width": 512,
                            "height": 512,
                        },
                        unit_overrides=[openpose_unit]
                        + [
                            {
                                "image": img,
                                "module": module,
                                "model": model,
                                "weight": 1 / len(portrait_imgs),
                            }
                            for img in portrait_imgs
                        ],
                    ).exec()
                )

    def test_face_id_real_multi_inputs(self):
        for module, model, lora in (sd15_face_id, sd15_face_id_portrait):
            name = "real_multi_inputs" + str((module, model, lora))
            with self.subTest(name=name):
                self.assertTrue(
                    APITestTemplate(
                        name=name,
                        gen_type="txt2img",
                        payload_overrides={
                            "prompt": base_prompt,
                            "negative_prompt": general_negative_prompt,
                            "steps": 20,
                            "width": 512,
                            "height": 512,
                        },
                        unit_overrides=[
                            openpose_unit,
                            {
                                "image": [{"image": img} for img in portrait_imgs],
                                "module": module,
                                "model": model,
                            },
                        ],
                    ).exec()
                )


if __name__ == "__main__":
    unittest.main()
