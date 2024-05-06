# https://github.com/ToTheBeginning/PuLID

import torch
import cv2
import numpy as np
from typing import Optional, List
from dataclasses import dataclass
from facexlib.parsing import init_parsing_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from torchvision.transforms.functional import normalize

from ..supported_preprocessor import Preprocessor, PreprocessorParameter
from scripts.utils import npimg2tensor, tensor2npimg, resize_image_with_pad


def to_gray(img):
    x = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    x = x.repeat(1, 3, 1, 1)
    return x


class PreprocessorFaceXLib(Preprocessor):
    def __init__(self):
        super().__init__(name="facexlib")
        self.tags = []
        self.slider_resolution = PreprocessorParameter(visible=False)
        self.model: Optional[FaceRestoreHelper] = None

    def load_model(self):
        if self.model is None:
            self.model = FaceRestoreHelper(
                upscale_factor=1,
                face_size=512,
                crop_ratio=(1, 1),
                det_model="retinaface_resnet50",
                save_ext="png",
                device=self.device,
            )
            self.model.face_parse = init_parsing_model(
                model_name="bisenet", device=self.device
            )
        self.model.face_parse.to(device=self.device)
        self.model.face_det.to(device=self.device)
        return self.model

    def unload(self) -> bool:
        """@Override"""
        if self.model is not None:
            self.model.face_parse.to(device="cpu")
            self.model.face_det.to(device="cpu")
            return True
        return False

    def __call__(
        self,
        input_image,
        resolution=512,
        slider_1=None,
        slider_2=None,
        slider_3=None,
        input_mask=None,
        return_tensor=False,
        **kwargs
    ):
        """
        @Override
        Returns black and white face features image with background removed.
        """
        self.load_model()
        self.model.clean_all()
        input_image, _ = resize_image_with_pad(input_image, resolution)
        # using facexlib to detect and align face
        image_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
        self.model.read_image(image_bgr)
        self.model.get_face_landmarks_5(only_center_face=True)
        self.model.align_warp_face()
        if len(self.model.cropped_faces) == 0:
            raise RuntimeError("facexlib align face fail")
        align_face = self.model.cropped_faces[0]
        align_face_rgb = cv2.cvtColor(align_face, cv2.COLOR_BGR2RGB)
        input = npimg2tensor(align_face_rgb)
        input = input.to(self.device)
        parsing_out = self.model.face_parse(
            normalize(input, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        )[0]
        parsing_out = parsing_out.argmax(dim=1, keepdim=True)
        bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
        bg = sum(parsing_out == i for i in bg_label).bool()
        white_image = torch.ones_like(input)
        # only keep the face features
        face_features_image = torch.where(bg, white_image, to_gray(input))
        if return_tensor:
            return face_features_image
        else:
            return tensor2npimg(face_features_image)


@dataclass
class PuLIDProjInput:
    id_ante_embedding: torch.Tensor
    id_cond_vit: torch.Tensor
    id_vit_hidden: List[torch.Tensor]


class PreprocessorPuLID(Preprocessor):
    """PuLID preprocessor."""

    def __init__(self):
        super().__init__(name="ip-adapter_pulid")
        self.tags = ["IP-Adapter"]
        self.slider_resolution = PreprocessorParameter(visible=False)
        self.returns_image = False
        self.preprocessors_deps = [
            "facexlib",
            "instant_id_face_embedding",
            "EVA02-CLIP-L-14-336",
        ]

    def facexlib_detect(self, input_image: np.ndarray) -> torch.Tensor:
        facexlib_preprocessor = Preprocessor.get_preprocessor("facexlib")
        return facexlib_preprocessor(input_image, return_tensor=True)

    def insightface_antelopev2_detect(self, input_image: np.ndarray) -> torch.Tensor:
        antelopev2_preprocessor = Preprocessor.get_preprocessor(
            "instant_id_face_embedding"
        )
        return antelopev2_preprocessor(input_image)

    def unload(self) -> bool:
        unloaded = False
        for p_name in self.preprocessors_deps:
            p = Preprocessor.get_preprocessor(p_name)
            if p is not None:
                unloaded = unloaded or p.unload()
        return unloaded

    def __call__(
        self,
        input_image,
        resolution,
        slider_1=None,
        slider_2=None,
        slider_3=None,
        input_mask=None,
        **kwargs
    ) -> Preprocessor.Result:
        id_ante_embedding = self.insightface_antelopev2_detect(input_image)
        if id_ante_embedding.ndim == 1:
            id_ante_embedding = id_ante_embedding.unsqueeze(0)

        face_features_image = self.facexlib_detect(input_image)
        evaclip_preprocessor = Preprocessor.get_preprocessor("EVA02-CLIP-L-14-336")
        assert (
            evaclip_preprocessor is not None
        ), "EVA02-CLIP-L-14-336 preprocessor not found! Please install sd-webui-controlnet-evaclip"
        r = evaclip_preprocessor(face_features_image)

        return Preprocessor.Result(
            value=PuLIDProjInput(
                id_ante_embedding=id_ante_embedding,
                id_cond_vit=r.id_cond_vit,
                id_vit_hidden=r.id_vit_hidden,
            ),
            display_images=[tensor2npimg(face_features_image)],
        )


Preprocessor.add_supported_preprocessor(PreprocessorFaceXLib())
Preprocessor.add_supported_preprocessor(PreprocessorPuLID())
