import unittest
import importlib
utils = importlib.import_module('extensions.sd-webui-controlnet.tests.utils', 'utils')
utils.setup_test_env()

from copy import copy
from scripts import external_code
from scripts import controlnet
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

        self.initial_max_models = shared.opts.data.get("control_net_max_models_num", 1)
        shared.opts.data.update(control_net_max_models_num=self.max_models)

        self.extra_models = 0

    def tearDown(self):
        shared.opts.data.update(control_net_max_models_num=self.initial_max_models)

    def get_expected_args_to(self):
        pos_args = 1  # is_ui
        args_len = pos_args + self.max_models + self.extra_models
        return self.args_offset + args_len

    def assert_update_in_place_ok(self):
        external_code.update_cn_script_in_place(self.scripts, self.script_args, self.cn_units)
        self.assertEqual(self.cn_script.args_to, self.get_expected_args_to())

    def test_empty_resizes_min_args(self):
        self.cn_units = []
        self.assert_update_in_place_ok()

    def test_empty_resizes_extra_args(self):
        self.extra_models = 1
        self.cn_units = [external_code.ControlNetUnit()] * (self.max_models + self.extra_models)
        self.assert_update_in_place_ok()


class TestControlNetUnitConversion(unittest.TestCase):
    def setUp(self):
        self.dummy_image = 'base64...'
        self.input = {}
        self.expected = external_code.ControlNetUnit()

    def assert_converts_to_expected(self):
        self.assertEqual(vars(external_code.to_processing_unit(self.input)), vars(self.expected))

    def test_empty_dict_works(self):
        self.assert_converts_to_expected()

    def test_image_works(self):
        self.input = {
            'image': self.dummy_image
        }
        self.expected = external_code.ControlNetUnit(image=self.dummy_image)
        self.assert_converts_to_expected()

    def test_image_alias_works(self):
        self.input = {
            'input_image': self.dummy_image
        }
        self.expected = external_code.ControlNetUnit(image=self.dummy_image)
        self.assert_converts_to_expected()


if __name__ == '__main__':
    unittest.main()