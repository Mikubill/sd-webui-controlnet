from ..ipadapter.presets import IPAdapterPreset
from ..supported_preprocessor import Preprocessor
from ..logging import logger


class PreprocessorIPAdapterAuto(Preprocessor):
    def __init__(self):
        super().__init__(name="ip-adapter-auto")
        self.tags = ["IP-Adapter"]
        self.sorting_priority = 1000
        self.returns_image = False
        self.show_control_mode = False

    def __call__(self, *args, **kwargs):
        assert "model" in kwargs
        model: str = kwargs["model"]
        module: str = IPAdapterPreset.match_model(model).module
        logger.info(f"ip-adapter-auto => {module}")

        p = Preprocessor.get_preprocessor(module)
        assert p is not None
        return p(*args, **kwargs)


Preprocessor.add_supported_preprocessor(PreprocessorIPAdapterAuto())
