import unittest
import importlib
utils = importlib.import_module('extensions.sd-webui-controlnet.tests.utils', 'utils')
utils.setup_test_env()

from scripts import external_code


class TestGetAllUnitsFrom(unittest.TestCase):
    def setUp(self):
        self.flat_control_unit = [
            True, "none", utils.get_model(), 1.0,
            utils.readImage("test/test_files/img2img_basic.png"),
            "Crop and Resize", False,
            64, 64, 64, 0.0, 1.0, False, False,
            external_code.ControlMode.BALANCED.value,
        ]
        self.object_unit = external_code.ControlNetUnit(*self.flat_control_unit)

    def test_empty_converts(self):
        script_args = []
        units = external_code.get_all_units_from(script_args)
        self.assertListEqual(units, [])

    def test_flattened_converts(self):
        script_args = self.flat_control_unit
        units = external_code.get_all_units_from(script_args)
        self.assertListEqual(units, [self.object_unit])

    def test_object_forwards(self):
        script_args = [self.object_unit]
        units = external_code.get_all_units_from(script_args)
        self.assertListEqual(units, [self.object_unit])

    def test_mixed_converts(self):
        script_args = [self.object_unit] + self.flat_control_unit + [self.object_unit] + self.flat_control_unit
        units = external_code.get_all_units_from(script_args)
        self.assertListEqual(units, [self.object_unit] * 4)


if __name__ == '__main__':
    unittest.main()