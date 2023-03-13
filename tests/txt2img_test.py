import unittest
import requests
import cv2
from base64 import b64encode

def readImage(path):
    img = cv2.imread(path)
    retval, buffer = cv2.imencode('.jpg', img)
    b64img = b64encode(buffer).decode("utf-8")
    return b64img

def get_model():
    r = requests.get("http://localhost:7860/controlnet/model_list")
    result = r.json()
    if "model_list" in result:
        result = result["model_list"]
        for item in result:
            print("Using model: ", item)
            return item
    return "None"


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


class TestOldTxt2ImgWorking(TestTxt2ImgWorkingBase, unittest.TestCase):
    def setUp(self):
        setup_args = [
            False, True, "none", get_model(), 1.0,
            readImage("test/test_files/img2img_basic.png"),
            False, "Scale to Fit (Inner Fit)", False, False,
            64, 64, 64, 0.0, 1.0, False
        ]
        self.setup_route(setup_args)

    def test_txt2img_simple_performed(self):
        self.assertEqual(requests.post(self.url_txt2img, json=self.simple_txt2img).status_code, 200)

    def test_txt2img_multiple_batches_performed(self):
        self.simple_txt2img["n_iter"] = 2
        self.assertEqual(requests.post(self.url_txt2img, json=self.simple_txt2img).status_code, 200)

    def test_txt2img_batch_performed(self):
        self.simple_txt2img["batch_size"] = 2
        self.assertEqual(requests.post(self.url_txt2img, json=self.simple_txt2img).status_code, 200)


class TestNewTxt2ImgWorking(TestTxt2ImgWorkingBase, unittest.TestCase):
    def setUp(self):
        setup_args = [
            False, {
                "enabled": True,
                "module": "none",
                "model": get_model(),
                "weight": 1.0,
                "input_image": readImage("test/test_files/img2img_basic.png"),
                "invert_image": False,
                "resize_mode": 1,
                "rgbbgr_mode": False,
                "lowvram": False,
                "processor_res": 64,
                "threshold_a": 64,
                "threshold_b": 64,
                "guidance_start": 0.0,
                "guidance_end": 1.0,
                "guessmode": False,
            }
        ]
        self.setup_route(setup_args)

    def test_txt2img_simple_performed(self):
        self.assertEqual(requests.post(self.url_txt2img, json=self.simple_txt2img).status_code, 200)

    def test_txt2img_multiple_batches_performed(self):
        self.simple_txt2img["n_iter"] = 2
        self.assertEqual(requests.post(self.url_txt2img, json=self.simple_txt2img).status_code, 200)

    def test_txt2img_batch_performed(self):
        self.simple_txt2img["batch_size"] = 2
        self.assertEqual(requests.post(self.url_txt2img, json=self.simple_txt2img).status_code, 200)


if __name__ == "__main__":
    unittest.main()