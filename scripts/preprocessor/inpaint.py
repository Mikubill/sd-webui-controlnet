from scripts.utils import visualize_inpaint_mask
from ..supported_preprocessor import Preprocessor, PreprocessorParameter


class PreprocessorInpaint(Preprocessor):
    def __init__(self):
        super().__init__(name="inpaint")
        self._label = "inpaint_global_harmonious"
        self.tags = ["Inpaint"]
        self.slider_resolution = PreprocessorParameter(visible=False)
        self.sorting_priority = 0
        self.accepts_mask = True
        self.requires_mask = True

    def __call__(
        self,
        input_image,
        resolution,
        slider_1=None,
        slider_2=None,
        slider_3=None,
        **kwargs
    ):
        return Preprocessor.Result(
            value=input_image,
            display_images=visualize_inpaint_mask(input_image)[None, :, :, :],
        )


class PreprocessorInpaintOnly(Preprocessor):
    def __init__(self):
        super().__init__(name="inpaint_only")
        self.tags = ["Inpaint"]
        self.slider_resolution = PreprocessorParameter(visible=False)
        self.sorting_priority = 100
        self.accepts_mask = True
        self.requires_mask = True

    def __call__(
        self,
        input_image,
        resolution,
        slider_1=None,
        slider_2=None,
        slider_3=None,
        **kwargs
    ):
        return Preprocessor.Result(
            value=input_image,
            display_images=visualize_inpaint_mask(input_image)[None, :, :, :],
        )


Preprocessor.add_supported_preprocessor(PreprocessorInpaint())
Preprocessor.add_supported_preprocessor(PreprocessorInpaintOnly())
