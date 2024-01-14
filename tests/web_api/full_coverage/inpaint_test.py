import unittest
import pytest
from .template import is_full_coverage, APITestTemplate, girl_img, mask_img


class TestInpaintFullCoverage(unittest.TestCase):
    def setUp(self):
        if not is_full_coverage:
            pytest.skip()

    def test_inpaint(self):
        for gen_type in ("img2img", "txt2img"):
            if gen_type == "img2img":
                payload = {
                    "init_images": [girl_img],
                    "mask": mask_img,
                }
                unit = {}
            else:
                payload = {}
                unit = {
                    "image": {
                        "image": girl_img,
                        "mask": mask_img,
                    }
                }

            unit["model"] = "control_v11p_sd15_inpaint [ebff9138]"

            for i_resize, resize_mode in enumerate(("Just Resize", "Crop and Resize", "Resize and Fill")):
                # Gen 512x768(input image size) for resize.
                if resize_mode == "Crop and Resize":
                    payload["height"] = 768
                    payload["width"] = 512

                # Gen 512x512 for inner fit.
                if resize_mode == "Crop and Resize":
                    payload["height"] = 512
                    payload["width"] = 512

                # Gen 768x768 for outer fit.
                if resize_mode == "Resize and Fill":
                    payload["height"] = 768
                    payload["width"] = 768

                if gen_type == "img2img":
                    payload["resize_mode"] = i_resize
                else:
                    unit["resize_mode"] = resize_mode

                for module in ("inpaint_only", "inpaint", "inpaint_only+lama"):
                    unit["module"] = module

                    with self.subTest(
                        gen_type=gen_type,
                        resize_mode=resize_mode,
                        module=module,
                    ):
                        self.assertTrue(
                            APITestTemplate(
                                f"{gen_type}_{resize_mode}_{module}",
                                gen_type,
                                payload_overrides=payload,
                                unit_overrides=unit,
                            ).exec()
                        )

if __name__ == "__main__":
    unittest.main()
