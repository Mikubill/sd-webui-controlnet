import unittest
from PIL import Image
import numpy as np

import importlib
utils = importlib.import_module('extensions.sd-webui-controlnet.tests.utils', 'utils')
utils.setup_test_env()

from scripts.controlnet import prepare_mask
from modules import processing


class TestPrepareMask(unittest.TestCase):
    def test_prepare_mask(self):
        p = processing.StableDiffusionProcessing()
        p.inpainting_mask_invert = True
        p.mask_blur = 5

        mask = Image.new('RGB', (10, 10), color='white')

        processed_mask = prepare_mask(mask, p)

        # Check that mask is correctly converted to grayscale
        self.assertTrue(processed_mask.mode, "L")

        # Check that mask colors are correctly inverted
        self.assertEqual(processed_mask.getpixel((0, 0)), 0)  # inverted white should be black

        p.inpainting_mask_invert = False
        processed_mask = prepare_mask(mask, p)

        # Check that mask colors are not inverted when 'inpainting_mask_invert' is False
        self.assertEqual(processed_mask.getpixel((0, 0)), 255)  # white should remain white

        p.mask_blur = 0
        mask = Image.new('RGB', (10, 10), color='black')
        processed_mask = prepare_mask(mask, p)

        # Check that mask is not blurred when 'mask_blur' is 0
        self.assertEqual(processed_mask.getpixel((0, 0)), 0)  # black should remain black


if __name__ == "__main__":
    unittest.main()
