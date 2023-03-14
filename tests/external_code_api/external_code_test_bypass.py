import unittest
import importlib
from copy import copy

external_code = importlib.import_module('extensions.sd-webui-controlnet.scripts.external_code', 'external_code')
controlnet = importlib.import_module('extensions.sd-webui-controlnet.scripts.controlnet', 'controlnet')
from modules import scripts, ui, shared


class TestExternalCodeWorking(unittest.TestCase):
    max_models = 4
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

    def tearDown(self):
        shared.opts.data.update(control_net_max_models_num=self.initial_max_models)

    def test_empty_resizes_min_args(self):
        expected_args_len = 1 + self.max_models

        external_code.update_cn_script_in_place(self.scripts, self.script_args, cn_units=[])

        self.assertEqual(self.cn_script.args_to, self.args_offset + expected_args_len)

    def test_empty_resizes_extra_args(self):
        extra_models = 1
        expected_args_len = 1 + self.max_models + extra_models
        cn_units = [external_code.ControlNetUnit()] * (self.max_models + extra_models)

        external_code.update_cn_script_in_place(self.scripts, self.script_args, cn_units)

        self.assertEqual(self.cn_script.args_to, self.args_offset + expected_args_len)


if __name__ == '__main__':
    unittest.main()