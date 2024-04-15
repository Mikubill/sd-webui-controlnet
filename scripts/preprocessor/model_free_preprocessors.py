"""Preprocessors that do not need to run a torch model."""

import cv2

from ..supported_preprocessor import Preprocessor, PreprocessorParameter
from ..utils import resize_image_with_pad
from annotator.util import HWC3


class PreprocessorNone(Preprocessor):
    def __init__(self):
        super().__init__(name="None")
        self._label = "none"
        self.sorting_priority = 10


class PreprocessorCanny(Preprocessor):
    def __init__(self):
        super().__init__(name="canny")
        self.tags = ["Canny"]
        self.model_filename_filters = ["canny"]
        self.slider_1 = PreprocessorParameter(
            minimum=0,
            maximum=256,
            step=1,
            value=100,
            label="Low Threshold",
        )
        self.slider_2 = PreprocessorParameter(
            minimum=0,
            maximum=256,
            step=1,
            value=200,
            label="High Threshold",
        )
        self.sorting_priority = 100
        self.use_soft_projection_in_hr_fix = True

    def __call__(
        self,
        input_image,
        resolution,
        slider_1=None,
        slider_2=None,
        slider_3=None,
        **kwargs
    ):
        input_image, remove_pad = resize_image_with_pad(input_image, resolution)
        canny_image = cv2.cvtColor(
            cv2.Canny(input_image, int(slider_1), int(slider_2)), cv2.COLOR_GRAY2RGB
        )
        return remove_pad(canny_image)


class PreprocessorInvert(Preprocessor):
    def __init__(self):
        super().__init__(name="invert")
        self._label = "invert (from white bg & black line)"
        self.tags = [
            "Canny",
            "Lineart",
            "Scribble",
            "Sketch",
            "MLSD",
        ]
        self.slider_resolution = PreprocessorParameter(visible=False)
        self.model_filename_filters = ["canny"]
        self.sorting_priority = 20

    def __call__(
        self,
        input_image,
        resolution,
        slider_1=None,
        slider_2=None,
        slider_3=None,
        **kwargs
    ):
        return 255 - HWC3(input_image)


class PreprocessorBlurGaussian(Preprocessor):
    def __init__(self):
        super().__init__(name="blur_gaussian")
        self.slider_1 = PreprocessorParameter(
            label="Sigma", minimum=64, maximum=2048, value=512
        )
        self.tags = ["Tile", "Blur"]

    def __call__(
        self,
        input_image,
        resolution,
        slider_1=None,
        slider_2=None,
        slider_3=None,
        input_mask=None,
        **kwargs
    ):
        img, remove_pad = resize_image_with_pad(input_image, resolution)
        img = remove_pad(img)
        result = cv2.GaussianBlur(img, (0, 0), float(slider_1))
        return result


Preprocessor.add_supported_preprocessor(PreprocessorNone())
Preprocessor.add_supported_preprocessor(PreprocessorCanny())
Preprocessor.add_supported_preprocessor(PreprocessorInvert())
Preprocessor.add_supported_preprocessor(PreprocessorBlurGaussian())
