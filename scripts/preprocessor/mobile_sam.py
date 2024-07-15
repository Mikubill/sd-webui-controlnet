from annotator.mobile_sam import SamDetector_Aux
from scripts.supported_preprocessor import Preprocessor

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
        if self.model is None:
            self.model = SamDetector_Aux.from_pretrained()

        result = self.model(input_image, detect_resolution=resolution, image_resolution=resolution, output_type="cv2")
        return result
    
Preprocessor.add_supported_preprocessor(PreprocessorMobileSam())