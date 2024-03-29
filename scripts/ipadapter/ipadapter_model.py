from typing import List, NamedTuple, Tuple, Union

import torch
import torch.nn as nn
import numpy as np
from transformers.models.clip.modeling_clip import CLIPVisionModelOutput

from .image_proj_models import (
    Resampler,
    ImageProjModel,
    MLPProjModel,
    MLPProjModelFaceId,
    ProjModelFaceIdPlus,
)


class ImageEmbed(NamedTuple):
    """Image embed for a single image."""

    cond_emb: torch.Tensor
    uncond_emb: torch.Tensor
    bypass_average: bool = False

    def eval(self, cond_mark: torch.Tensor) -> torch.Tensor:
        assert cond_mark.ndim == 4
        assert self.cond_emb.ndim == self.uncond_emb.ndim == 3
        assert (
            self.uncond_emb.shape[0] == 1
            or self.cond_emb.shape[0] == self.uncond_emb.shape[0]
        )
        assert (
            self.cond_emb.shape[0] == 1 or self.cond_emb.shape[0] == cond_mark.shape[0]
        )
        cond_mark = cond_mark[:, :, :, 0].to(self.cond_emb)
        device = cond_mark.device
        dtype = cond_mark.dtype
        return self.cond_emb.to(
            device=device, dtype=dtype
        ) * cond_mark + self.uncond_emb.to(device=device, dtype=dtype) * (1 - cond_mark)

    def average_of(*args: List[Tuple[torch.Tensor, torch.Tensor]]) -> "ImageEmbed":
        conds, unconds, _ = zip(*args)

        def average_tensors(tensors: List[torch.Tensor]) -> torch.Tensor:
            return torch.sum(torch.stack(tensors), dim=0) / len(tensors)

        return ImageEmbed(average_tensors(conds), average_tensors(unconds))


class To_KV(torch.nn.Module):
    def __init__(self, state_dict):
        super().__init__()

        self.to_kvs = nn.ModuleDict()
        for key, value in state_dict.items():
            k = key.replace(".weight", "").replace(".", "_")
            self.to_kvs[k] = nn.Linear(value.shape[1], value.shape[0], bias=False)
            self.to_kvs[k].weight.data = value


class IPAdapterModel(torch.nn.Module):
    def __init__(
        self,
        state_dict,
        clip_embeddings_dim,
        cross_attention_dim,
        is_plus,
        sdxl_plus,
        is_full,
        is_faceid: bool,
        is_portrait: bool,
        is_instantid: bool,
    ):
        super().__init__()
        self.device = "cpu"

        self.clip_embeddings_dim = clip_embeddings_dim
        self.cross_attention_dim = cross_attention_dim
        self.is_plus = is_plus
        self.sdxl_plus = sdxl_plus
        self.is_full = is_full
        self.clip_extra_context_tokens = 16 if (self.is_plus or is_portrait) else 4

        if is_instantid:
            self.image_proj_model = self.init_proj_instantid()
        elif is_faceid:
            self.image_proj_model = self.init_proj_faceid()
        elif self.is_plus:
            if self.is_full:
                self.image_proj_model = MLPProjModel(
                    cross_attention_dim=cross_attention_dim,
                    clip_embeddings_dim=clip_embeddings_dim,
                )
            else:
                self.image_proj_model = Resampler(
                    dim=1280 if sdxl_plus else cross_attention_dim,
                    depth=4,
                    dim_head=64,
                    heads=20 if sdxl_plus else 12,
                    num_queries=self.clip_extra_context_tokens,
                    embedding_dim=clip_embeddings_dim,
                    output_dim=self.cross_attention_dim,
                    ff_mult=4,
                )
        else:
            self.clip_extra_context_tokens = (
                state_dict["image_proj"]["proj.weight"].shape[0]
                // self.cross_attention_dim
            )

            self.image_proj_model = ImageProjModel(
                cross_attention_dim=self.cross_attention_dim,
                clip_embeddings_dim=clip_embeddings_dim,
                clip_extra_context_tokens=self.clip_extra_context_tokens,
            )

        self.load_ip_adapter(state_dict)

    def init_proj_faceid(self):
        if self.is_plus:
            image_proj_model = ProjModelFaceIdPlus(
                cross_attention_dim=self.cross_attention_dim,
                id_embeddings_dim=512,
                clip_embeddings_dim=self.clip_embeddings_dim,
                num_tokens=4,
            )
        else:
            image_proj_model = MLPProjModelFaceId(
                cross_attention_dim=self.cross_attention_dim,
                id_embeddings_dim=512,
                num_tokens=self.clip_extra_context_tokens,
            )
        return image_proj_model

    def init_proj_instantid(self, image_emb_dim=512, num_tokens=16):
        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=num_tokens,
            embedding_dim=image_emb_dim,
            output_dim=self.cross_attention_dim,
            ff_mult=4,
        )
        return image_proj_model

    def load_ip_adapter(self, state_dict):
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        self.ip_layers = To_KV(state_dict["ip_adapter"])

    @torch.inference_mode()
    def get_image_embeds(self, clip_vision_output: CLIPVisionModelOutput) -> ImageEmbed:
        self.image_proj_model.cpu()

        if self.is_plus:
            from annotator.clipvision import clip_vision_h_uc, clip_vision_vith_uc

            cond = self.image_proj_model(
                clip_vision_output["hidden_states"][-2].to(
                    device="cpu", dtype=torch.float32
                )
            )
            uncond = (
                clip_vision_vith_uc.to(cond)
                if self.sdxl_plus
                else self.image_proj_model(clip_vision_h_uc.to(cond))
            )
            return ImageEmbed(cond, uncond)

        clip_image_embeds = clip_vision_output["image_embeds"].to(
            device="cpu", dtype=torch.float32
        )
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        # input zero vector for unconditional.
        uncond_image_prompt_embeds = self.image_proj_model(
            torch.zeros_like(clip_image_embeds)
        )
        return ImageEmbed(image_prompt_embeds, uncond_image_prompt_embeds)

    @torch.inference_mode()
    def get_image_embeds_faceid_plus(
        self,
        face_embed: torch.Tensor,
        clip_vision_output: CLIPVisionModelOutput,
        is_v2: bool,
    ) -> ImageEmbed:
        face_embed = face_embed.to(self.device, dtype=torch.float32)
        from annotator.clipvision import clip_vision_h_uc

        clip_embed = clip_vision_output["hidden_states"][-2].to(
            device=self.device, dtype=torch.float32
        )
        return ImageEmbed(
            self.image_proj_model(face_embed, clip_embed, shortcut=is_v2),
            self.image_proj_model(
                torch.zeros_like(face_embed),
                clip_vision_h_uc.to(clip_embed),
                shortcut=is_v2,
            ),
        )

    @torch.inference_mode()
    def get_image_embeds_faceid(self, insightface_output: torch.Tensor) -> ImageEmbed:
        """Get image embeds for non-plus faceid. Multiple inputs are supported."""
        self.image_proj_model.to(self.device)
        faceid_embed = insightface_output.to(self.device, dtype=torch.float32)
        return ImageEmbed(
            self.image_proj_model(faceid_embed),
            self.image_proj_model(torch.zeros_like(faceid_embed)),
        )

    @torch.inference_mode()
    def get_image_embeds_instantid(
        self, prompt_image_emb: Union[torch.Tensor, np.ndarray]
    ) -> ImageEmbed:
        """Get image embeds for instantid."""
        image_proj_model_in_features = 512
        if isinstance(prompt_image_emb, torch.Tensor):
            prompt_image_emb = prompt_image_emb.clone().detach()
        else:
            prompt_image_emb = torch.tensor(prompt_image_emb)

        prompt_image_emb = prompt_image_emb.to(device=self.device, dtype=torch.float32)
        prompt_image_emb = prompt_image_emb.reshape(
            [1, -1, image_proj_model_in_features]
        )
        return ImageEmbed(
            self.image_proj_model(prompt_image_emb),
            self.image_proj_model(torch.zeros_like(prompt_image_emb)),
        )
