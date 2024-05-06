import unittest
import importlib

import numpy as np

utils = importlib.import_module('extensions.sd-webui-controlnet.tests.utils', 'utils')


from copy import copy
from scripts import external_code
from scripts import controlnet
from scripts.enums import ResizeMode
from internal_controlnet.external_code import ControlNetUnit
from modules import scripts, ui, shared


class TestExternalCodeWorking(unittest.TestCase):
    max_models = 6
    args_offset = 10

    def setUp(self):
        self.scripts = copy(scripts.scripts_txt2img)
        self.scripts.initialize_scripts(False)
        ui.create_ui()
        self.cn_script = controlnet.Script()
        self.cn_script.args_from = self.args_offset
        self.cn_script.args_to = self.args_offset + self.max_models
        self.scripts.alwayson_scripts = [self.cn_script]
        self.script_args = [None] * self.cn_script.args_from

        self.initial_max_models = shared.opts.data.get("control_net_unit_count", 3)
        shared.opts.data.update(control_net_unit_count=self.max_models)

        self.extra_models = 0

    def tearDown(self):
        shared.opts.data.update(control_net_unit_count=self.initial_max_models)

    def get_expected_args_to(self):
        args_len = max(self.max_models, len(self.cn_units))
        return self.args_offset + args_len

    def assert_update_in_place_ok(self):
        external_code.update_cn_script_in_place(self.scripts, self.script_args, self.cn_units)
        self.assertEqual(self.cn_script.args_to, self.get_expected_args_to())

    def test_empty_resizes_min_args(self):
        self.cn_units = []
        self.assert_update_in_place_ok()

    def test_empty_resizes_extra_args(self):
        extra_models = 1
        self.cn_units = [ControlNetUnit()] * (self.max_models + extra_models)
        self.assert_update_in_place_ok()


class TestPixelPerfectResolution(unittest.TestCase):
    def test_outer_fit(self):
        image = np.zeros((100, 100, 3))
        target_H, target_W = 50, 100
        resize_mode = ResizeMode.OUTER_FIT
        result = external_code.pixel_perfect_resolution(image, target_H, target_W, resize_mode)
        expected = 50  # manually computed expected result
        self.assertEqual(result, expected)

    def test_inner_fit(self):
        image = np.zeros((100, 100, 3))
        target_H, target_W = 50, 100
        resize_mode = ResizeMode.INNER_FIT
        result = external_code.pixel_perfect_resolution(image, target_H, target_W, resize_mode)
        expected = 100  # manually computed expected result
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()