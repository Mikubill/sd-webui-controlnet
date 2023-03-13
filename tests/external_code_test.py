import unittest
import importlib
from copy import copy

external_code = importlib.import_module('extensions.sd-webui-controlnet.scripts.external_code', 'external_code')
from modules import scripts


class TestExternalCodeWorking(unittest.TestCase):
    def setUp(self):
        self.scripts = copy(scripts.scripts_txt2img)
        self.cn_script = external_code.find_cn_script(self.scripts)
        self.script_args = [None] * self.cn_script.args_from

    def test_empty_resizes_min_args(self):
        external_code.update_cn_script_in_place(self.scripts, self.script_args, cn_units=[])
        self.assertEqual(len(self.script_args), self.cn_script.args_to)


if __name__ == '__main__':
    unittest.main()