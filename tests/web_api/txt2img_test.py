import unittest
import importlib
utils = importlib.import_module('extensions.sd-webui-controlnet.tests.utils', 'utils')
utils.setup_test_env()
import requests



class TestTxt2ImgWorkingBase(unittest.TestCase):
    def setup_route(self, setup_args):
        self.url_txt2img = "http://localhost:7860/sdapi/v1/txt2img"
        self.simple_txt2img = {
            "enable_hr": False,
            "denoising_strength": 0,
            "firstphase_width": 0,
            "firstphase_height": 0,
            "prompt": "example prompt",
            "styles": [],
            "seed": -1,
            "subseed": -1,
            "subseed_strength": 0,
            "seed_resize_from_h": -1,
            "seed_resize_from_w": -1,
            "batch_size": 1,
            "n_iter": 1,
            "steps": 3,
            "cfg_scale": 7,
            "width": 64,
            "height": 64,
            "restore_faces": False,
            "tiling": False,
            "negative_prompt": "",
            "eta": 0,
            "s_churn": 0,
            "s_tmax": 0,
            "s_tmin": 0,
            "s_noise": 1,
            "sampler_index": "Euler a",
            "alwayson_scripts": {}
        }
        self.setup_controlnet_params(setup_args)

    def setup_controlnet_params(self, setup_args):
        self.simple_txt2img["alwayson_scripts"]["ControlNet"] = {
            "args": setup_args
        }

    def assert_status_ok(self, msg=None):
        self.assertEqual(requests.post(self.url_txt2img, json=self.simple_txt2img).status_code, 200, msg)


class TestDeprecatedTxt2ImgWorking(TestTxt2ImgWorkingBase, unittest.TestCase):
    def setUp(self):
        controlnet_unit = [
            True, "none", utils.get_model(), 1.0,
            utils.readImage("test/test_files/img2img_basic.png"),
            False, "Crop and Resize", False,
            512, 64, 64, 0.0, 1.0, False, False,
            "Balanced",
        ]
        setup_args = controlnet_unit * getattr(self, 'units_count', 1)
        self.setup_route(setup_args)

    def test_txt2img_simple_performed(self):
        self.assert_status_ok()

    def test_txt2img_multiple_batches_performed(self):
        self.simple_txt2img["n_iter"] = 2
        self.assert_status_ok()

    def test_txt2img_batch_performed(self):
        self.simple_txt2img["batch_size"] = 2
        self.assert_status_ok()

    def test_txt2img_2_units(self):
        self.units_count = 2
        self.setUp()
        self.assert_status_ok()


class TestAlwaysonTxt2ImgWorking(TestTxt2ImgWorkingBase, unittest.TestCase):
    def setUp(self):
        controlnet_unit = {
            "enabled": True,
            "module": "none",
            "model": utils.get_model(),
            "weight": 1.0,
            "image": utils.readImage("test/test_files/img2img_basic.png"),
            "mask": utils.readImage("test/test_files/img2img_basic.png"),
            "resize_mode": 1,
            "lowvram": False,
            "processor_res": 64,
            "threshold_a": 64,
            "threshold_b": 64,
            "guidance_start": 0.0,
            "guidance_end": 1.0,
            "guessmode": False,
            "pixel_perfect": False
        }
        setup_args = [controlnet_unit] * getattr(self, 'units_count', 1)
        self.setup_route(setup_args)

    def test_txt2img_simple_performed(self):
        self.assert_status_ok()

    def test_txt2img_multiple_batches_performed(self):
        self.simple_txt2img["n_iter"] = 2
        self.assert_status_ok()

    def test_txt2img_batch_performed(self):
        self.simple_txt2img["batch_size"] = 2
        self.assert_status_ok()

    def test_txt2img_2_units(self):
        self.units_count = 2
        self.setUp()
        self.assert_status_ok()

    def test_txt2img_8_units(self):
        self.units_count = 8
        self.setUp()
        self.assert_status_ok()

    def test_txt2img_default_params(self):
        self.simple_txt2img["alwayson_scripts"]["ControlNet"]["args"] = [
            {
                "input_image": utils.readImage("test/test_files/img2img_basic.png"),
                "model": utils.get_model(),
            }
        ]

        self.assert_status_ok()

    def test_call_with_preprocessors(self):
        avaliable_modules = utils.get_modules()
        for module in ['depth', 'openpose_full']:
            assert module in avaliable_modules, f'Failed to find {module}.'
            with self.subTest(module=module):
                self.simple_txt2img["alwayson_scripts"]["ControlNet"]["args"] = [
                    {
                        "input_image": utils.readImage("test/test_files/img2img_basic.png"),
                        "model": utils.get_model(),
                        "module": module
                    }
                ]
                self.assert_status_ok(f'Running preprocessor module: {module}')


if __name__ == "__main__":
    unittest.main()
