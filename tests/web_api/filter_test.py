import unittest
import importlib
utils = importlib.import_module(
    'extensions.sd-webui-controlnet.tests.utils', 'utils')
utils.setup_test_env()


class TestFilterEndpointWorking(unittest.TestCase):
    def setUp(self):
        self.base_args = {
            "keyword": "All",
        }

    def assert_response(self,data):
        data.pop('keywords', None)
        expected_value = {'module_list': [], 'model_list': [], 'default_option': 'none', 'default_model': 'None'}
        self.assertEqual(data, expected_value)


    def test_filter_with_invalid_keyword_performed(self):
        filter_args = self.base_args.copy()
        filter_args.update({
            "keyword": "INVALID",
        })
        response = utils.filter(filter_args)
        self.assertEqual(response.status_code, 200)
        self.assert_response(response.json().copy())
    

if __name__ == "__main__":
    unittest.main()
