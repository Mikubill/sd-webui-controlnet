from __future__ import annotations

from ..enums import StableDiffusionVersion
from typing import NamedTuple, Optional, List


class IPAdapterPreset(NamedTuple):
    """Preset for IPAdapter."""

    name: str
    module: str  # Preprocessor
    model: str  # Name of model file
    sd_version: StableDiffusionVersion  # Supported SD version.
    lora: Optional[str] = None

    @staticmethod
    def match_model(model_name: str) -> IPAdapterPreset:
        model_name = model_name.split("[")[0].strip()
        assert (
            model_name in _preset_by_model
        ), f"{model_name} not found in ipadapter presets. Please try manually pick preprocessor."
        return _preset_by_model[model_name]


clip_h = "ip-adapter_clip_h"
clip_g = "ip-adapter_clip_g"
insightface = "ip-adapter_face_id"
insightface_clip_h = "ip-adapter_face_id_plus"


ipadapter_presets: List[IPAdapterPreset] = [
    IPAdapterPreset(
        name="light",
        module=clip_h,
        model="ip-adapter_sd15_light",
        sd_version=StableDiffusionVersion.SD1x,
    ),
    IPAdapterPreset(
        name="light_v11",
        module=clip_h,
        model="ip-adapter_sd15_light_v11",
        sd_version=StableDiffusionVersion.SD1x,
    ),
    IPAdapterPreset(
        name="vit-g",
        module=clip_g,
        model="ip-adapter_sd15_vit-G",
        sd_version=StableDiffusionVersion.SD1x,
    ),
    IPAdapterPreset(
        name="standard",
        module=clip_h,
        model="ip-adapter_sd15",
        sd_version=StableDiffusionVersion.SD1x,
    ),
    IPAdapterPreset(
        name="plus",
        module=clip_h,
        model="ip-adapter-plus_sd15",
        sd_version=StableDiffusionVersion.SD1x,
    ),
    IPAdapterPreset(
        name="plus",
        module=clip_h,
        model="ip-adapter_sd15_plus",
        sd_version=StableDiffusionVersion.SD1x,
    ),
    IPAdapterPreset(
        name="plus-composition",
        module=clip_h,
        model="ip-adapter_plus_composition_sd15",
        sd_version=StableDiffusionVersion.SD1x,
    ),
    IPAdapterPreset(
        name="plus_face",
        module=clip_h,
        model="ip-adapter-plus-face_sd15",
        sd_version=StableDiffusionVersion.SD1x,
    ),
    IPAdapterPreset(
        name="full_face",
        module=clip_h,
        model="ip-adapter-full-face_sd15",
        sd_version=StableDiffusionVersion.SD1x,
    ),
    IPAdapterPreset(
        name="face_id",
        module=insightface,
        model="ip-adapter-faceid_sd15",
        lora="ip-adapter-faceid_sd15_lora",
        sd_version=StableDiffusionVersion.SD1x,
    ),
    IPAdapterPreset(
        name="face_id_plus",
        module=insightface_clip_h,
        model="ip-adapter-faceid-plus_sd15",
        lora="ip-adapter-faceid-plus_sd15_lora",
        sd_version=StableDiffusionVersion.SD1x,
    ),
    IPAdapterPreset(
        name="face_id_plus_v2",
        module=insightface_clip_h,
        model="ip-adapter-faceid-plusv2_sd15",
        lora="ip-adapter-faceid-plusv2_sd15_lora",
        sd_version=StableDiffusionVersion.SD1x,
    ),
    IPAdapterPreset(
        name="face_id_portrait",
        module=insightface,
        model="ip-adapter-faceid-portrait_sd15",
        sd_version=StableDiffusionVersion.SD1x,
    ),
    IPAdapterPreset(
        name="face_id_portrait_v11",
        module=insightface,
        model="ip-adapter-faceid-portrait-v11_sd15",
        sd_version=StableDiffusionVersion.SD1x,
    ),
    IPAdapterPreset(
        name="standard-g",
        module=clip_g,
        model="ip-adapter_sdxl",
        sd_version=StableDiffusionVersion.SDXL,
    ),
    IPAdapterPreset(
        name="standard-h",
        module=clip_h,
        model="ip-adapter_sdxl_vit-h",
        sd_version=StableDiffusionVersion.SDXL,
    ),
    IPAdapterPreset(
        name="plus-h",
        module=clip_h,
        model="ip-adapter-plus_sdxl_vit-h",
        sd_version=StableDiffusionVersion.SDXL,
    ),
    IPAdapterPreset(
        name="plus-composition",
        module=clip_h,
        model="ip-adapter_plus_composition_sdxl",
        sd_version=StableDiffusionVersion.SDXL,
    ),
    IPAdapterPreset(
        name="plus_face-h",
        module=clip_h,
        model="ip-adapter-plus-face_sdxl_vit-h",
        sd_version=StableDiffusionVersion.SDXL,
    ),
    IPAdapterPreset(
        name="face_id",
        module=insightface,
        model="ip-adapter-faceid_sdxl",
        lora="ip-adapter-faceid_sdxl_lora",
        sd_version=StableDiffusionVersion.SDXL,
    ),
    IPAdapterPreset(
        name="face_id_plusv2",
        module=insightface_clip_h,
        model="ip-adapter-faceid-plusv2_sdxl",
        lora="ip-adapter-faceid-plusv2_sdxl_lora",
        sd_version=StableDiffusionVersion.SDXL,
    ),
    IPAdapterPreset(
        name="face_id_portrait",
        module=insightface,
        model="ip-adapter-faceid-portrait_sdxl",
        sd_version=StableDiffusionVersion.SDXL,
    ),
    IPAdapterPreset(
        name="pulid",
        module="ip-adapter_pulid",
        model="ip-adapter_pulid_sdxl_fp16",
        sd_version=StableDiffusionVersion.SDXL,
    ),
]

_preset_by_model = {p.model: p for p in ipadapter_presets}
