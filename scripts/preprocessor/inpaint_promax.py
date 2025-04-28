import cv2
import numpy as np

from ..supported_preprocessor import Preprocessor, PreprocessorParameter

class PreprocessorProMaxInpaint(Preprocessor):
    def __init__(self):
        super().__init__(name="inpaint_promax")
        self.tags = ["Inpaint"]
        self.returns_image = True
        self.model = None
        self.slider_resolution = PreprocessorParameter(visible=False)
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
        img = input_image.copy()
        H, W, C = img.shape
        assert C == 4, "No mask is provided!"
    
        img[img[:,:,3] > 0.,:3] = 0
        img[:,:,3]=0

        return Preprocessor.Result(
            value=img,
            display_images=[
                img
            ],
        )

Preprocessor.add_supported_preprocessor(PreprocessorProMaxInpaint())