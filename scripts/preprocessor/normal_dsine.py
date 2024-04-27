from ..supported_preprocessor import Preprocessor, PreprocessorParameter


class PreprocessorNormalDsine(Preprocessor):
    def __init__(self):
        super().__init__(name="normal_dsine")
        self.tags = ["NormalMap"]
        self.slider_1 = PreprocessorParameter(
            minimum=0,
            maximum=360,
            step=0.1,
            value=60,
            label="Fov",
        )
        self.slider_2 = PreprocessorParameter(
            minimum=1,
            maximum=20,
            step=1,
            value=5,
            label="Iterations",
        )
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
            from annotator.normaldsine import NormalDsineDetector

            self.model = NormalDsineDetector()

        result = self.model(
            input_image,
            new_fov=float(slider_1),
            iterations=int(slider_2),
            resulotion=resolution,
        )
        return result


Preprocessor.add_supported_preprocessor(PreprocessorNormalDsine())
