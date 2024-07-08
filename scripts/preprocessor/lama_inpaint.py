import cv2
import numpy as np

from ..supported_preprocessor import Preprocessor, PreprocessorParameter
from ..utils import resize_image_with_pad, visualize_inpaint_mask


class PreprocessorLamaInpaint(Preprocessor):
    def __init__(self):
        super().__init__(name="inpaint_only+lama")
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
        img = input_image
        H, W, C = img.shape
        assert C == 4, "No mask is provided!"
        raw_color = img[:, :, 0:3].copy()
        raw_mask = img[:, :, 3:4].copy()

        res = 256  # Always use 256 since lama is trained on 256

        img_res, remove_pad = resize_image_with_pad(img, res)

        if self.model is None:
            from annotator.lama import LamaInpainting

            self.model = LamaInpainting()
        # applied auto inversion
        prd_color = self.model(img_res)
        prd_color = remove_pad(prd_color)
        prd_color = cv2.resize(prd_color, (W, H))

        alpha = raw_mask.astype(np.float32) / 255.0
        fin_color = prd_color.astype(np.float32) * alpha + raw_color.astype(
            np.float32
        ) * (1 - alpha)
        fin_color = fin_color.clip(0, 255).astype(np.uint8)

        result = np.concatenate([fin_color, raw_mask], axis=2)
        return Preprocessor.Result(
            value=result,
            display_images=[
                result[:, :, :3],
                visualize_inpaint_mask(result),
            ],
        )


Preprocessor.add_supported_preprocessor(PreprocessorLamaInpaint())
