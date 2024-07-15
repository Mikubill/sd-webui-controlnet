from annotator.mobile_sam import SamDetector_Aux
from scripts.supported_preprocessor import Preprocessor
from scripts.utils import resize_image_with_pad

class PreprocessorMobileSam(Preprocessor):
    def __init__(self):
        super().__init__(name="mobile_sam")
        self.tags = ["Segmentation"]
        self.model = None

    def __call__(
        self,
        input_image,
        resolution,
        slider_1=None,
        slider_2=None,
        slider_3=None,
        **kwargs
    ):
        img, remove_pad = resize_image_with_pad(input_image, resolution)
        if self.model is None:
            self.model = SamDetector_Aux.from_pretrained()

        result = self.model(img, detect_resolution=resolution, image_resolution=resolution)
        return remove_pad(result)
    
Preprocessor.add_supported_preprocessor(PreprocessorMobileSam())