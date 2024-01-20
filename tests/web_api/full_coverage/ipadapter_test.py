import unittest
import pytest
from typing import NamedTuple, Optional

from .template import (
    sd_version,
    StableDiffusionVersion,
    is_full_coverage,
    APITestTemplate,
    realistic_girl_face_img,
)


class AdapterSetting(NamedTuple):
    module: str
    model: str
    lora: Optional[str] = None


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
sdxl_face_id = AdapterSetting(
    "ip-adapter_face_id",
    "ip-adapter-faceid_sdxl [59ee31a3]",
    "ip-adapter-faceid_sdxl_lora",
)


class TestInpaintFullCoverage(unittest.TestCase):
    def setUp(self):
        if not is_full_coverage:
            pytest.skip()

        if sd_version == StableDiffusionVersion.SDXL:
            self.settings = [sdxl_face_id]
        else:
            self.settings = [sd15_face_id, sd15_face_id_plus, sd15_face_id_plus_v2]

    def test_face_id(self):
        for module, model, lora in self.settings:
            name = str((module, model, lora))
            with self.subTest(name=name):
                self.assertTrue(
                    APITestTemplate(
                        name,
                        "txt2img",
                        payload_overrides={
                            "prompt": f"1girl, <lora:{lora}:0.6>",
                        },
                        unit_overrides={
                            "module": module,
                            "model": model,
                            "image": realistic_girl_face_img,
                        },
                    ).exec()
                )


if __name__ == "__main__":
    unittest.main()
