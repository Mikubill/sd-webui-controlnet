import numpy as np

from ..supported_preprocessor import Preprocessor, PreprocessorParameter


def visualize_inpaint_mask(img):
    if img.ndim == 3 and img.shape[2] == 4:
        result = img.copy()
        mask = result[:, :, 3]
        mask = 255 - mask // 2
        result[:, :, 3] = mask
        return np.ascontiguousarray(result.copy())
    return img


class PreprocessorInpaint(Preprocessor):
    def __init__(self):
        super().__init__(name="inpaint")
        self._label = "inpaint_global_harmonious"
        self.tags = ["Inpaint"]
        self.slider_resolution = PreprocessorParameter(visible=False)
        self.sorting_priority = 0
        self.accepts_mask = True
        self.requires_mask = True

    def get_display_image(self, input_image: np.ndarray, result):
        return visualize_inpaint_mask(result)

    def __call__(
        self,
        input_image,
        resolution,
        slider_1=None,
        slider_2=None,
        slider_3=None,
        **kwargs
    ):
        return input_image


class PreprocessorInpaintOnly(Preprocessor):
    def __init__(self):
        super().__init__(name="inpaint_only")
        self.tags = ["Inpaint"]
        self.slider_resolution = PreprocessorParameter(visible=False)
        self.sorting_priority = 100
        self.accepts_mask = True
        self.requires_mask = True

    def get_display_image(self, input_image: np.ndarray, result):
        return visualize_inpaint_mask(result)

    def __call__(
        self,
        input_image,
        resolution,
        slider_1=None,
        slider_2=None,
        slider_3=None,
        **kwargs
    ):
        return input_image


Preprocessor.add_supported_preprocessor(PreprocessorInpaint())
Preprocessor.add_supported_preprocessor(PreprocessorInpaintOnly())
