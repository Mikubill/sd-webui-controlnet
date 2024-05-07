from __future__ import annotations
import os
import torch
import numpy as np
from typing import Optional, List, Annotated, ClassVar, Callable, Any, Tuple, Union
from pydantic import BaseModel, validator, root_validator, Field
from PIL import Image
from logging import Logger
from copy import copy
from enum import Enum

from scripts.enums import (
    InputMode,
    ResizeMode,
    ControlMode,
    HiResFixOption,
    PuLIDMode,
)


def _unimplemented_func(*args, **kwargs):
    raise NotImplementedError("Not implemented.")


def field_to_displaytext(fieldname: str) -> str:
    return " ".join([word.capitalize() for word in fieldname.split("_")])


def displaytext_to_field(text: str) -> str:
    return "_".join([word.lower() for word in text.split(" ")])


def serialize_value(value) -> str:
    if isinstance(value, Enum):
        return value.value
    return str(value)


def parse_value(value: str) -> Union[str, float, int, bool]:
    if value in ("True", "False"):
        return value == "True"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value  # Plain string.


class ControlNetUnit(BaseModel):
    """
    Represents an entire ControlNet processing unit.
    """

    class Config:
        arbitrary_types_allowed = True
        extra = "ignore"

    cls_match_module: ClassVar[Callable[[str], bool]] = _unimplemented_func
    cls_match_model: ClassVar[Callable[[str], bool]] = _unimplemented_func
    cls_decode_base64: ClassVar[Callable[[str], np.ndarray]] = _unimplemented_func
    cls_torch_load_base64: ClassVar[Callable[[Any], torch.Tensor]] = _unimplemented_func
    cls_get_preprocessor: ClassVar[Callable[[str], Any]] = _unimplemented_func
    cls_logger: ClassVar[Logger] = Logger("ControlNetUnit")

    # UI only fields.
    is_ui: bool = False
    input_mode: InputMode = InputMode.SIMPLE
    batch_images: Optional[Any] = None
    output_dir: str = ""
    loopback: bool = False

    # General fields.
    enabled: bool = False
    module: str = "none"

    @validator("module", always=True, pre=True)
    def check_module(cls, value: str) -> str:
        if not ControlNetUnit.cls_match_module(value):
            raise ValueError(f"module({value}) not found in supported modules.")
        return value

    model: str = "None"

    @validator("model", always=True, pre=True)
    def check_model(cls, value: str) -> str:
        if not ControlNetUnit.cls_match_model(value):
            raise ValueError(f"model({value}) not found in supported models.")
        return value

    weight: Annotated[float, Field(ge=0.0, le=2.0)] = 1.0

    # The image to be used for this ControlNetUnit.
    image: Optional[Any] = None

    resize_mode: ResizeMode = ResizeMode.INNER_FIT

    @validator("resize_mode", always=True, pre=True)
    def check_resize_mode(cls, value) -> ResizeMode:
        resize_mode_aliases = {
            "Inner Fit (Scale to Fit)": "Crop and Resize",
            "Outer Fit (Shrink to Fit)": "Resize and Fill",
            "Scale to Fit (Inner Fit)": "Crop and Resize",
            "Envelope (Outer Fit)": "Resize and Fill",
        }
        if isinstance(value, str):
            return ResizeMode(resize_mode_aliases.get(value, value))
        assert isinstance(value, ResizeMode)
        return value

    low_vram: bool = False
    processor_res: int = -1
    threshold_a: float = -1
    threshold_b: float = -1

    @root_validator
    def bound_check_params(cls, values: dict) -> dict:
        """
        Checks and corrects negative parameters in ControlNetUnit 'unit' in place.
        Parameters 'processor_res', 'threshold_a', 'threshold_b' are reset to
        their default values if negative.
        """
        enabled = values.get("enabled")
        if not enabled:
            return values

        module = values.get("module")
        if not module:
            return values

        preprocessor = cls.cls_get_preprocessor(module)
        assert preprocessor is not None
        for unit_param, param in zip(
            ("processor_res", "threshold_a", "threshold_b"),
            ("slider_resolution", "slider_1", "slider_2"),
        ):
            value = values.get(unit_param)
            cfg = getattr(preprocessor, param)
            if value < cfg.minimum or value > cfg.maximum:
                values[unit_param] = cfg.value
                # Only report warning when non-default value is used.
                if value != -1:
                    cls.cls_logger.info(
                        f"[{module}.{unit_param}] Invalid value({value}), using default value {cfg.value}."
                    )
        return values

    guidance_start: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    guidance_end: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0

    @root_validator
    def guidance_check(cls, values: dict) -> dict:
        start = values.get("guidance_start")
        end = values.get("guidance_end")
        if start > end:
            raise ValueError(f"guidance_start({start}) > guidance_end({end})")
        return values

    pixel_perfect: bool = False
    control_mode: ControlMode = ControlMode.BALANCED
    # Whether to crop input image based on A1111 img2img mask. This flag is only used when `inpaint area`
    # in A1111 is set to `Only masked`. In API, this correspond to `inpaint_full_res = True`.
    inpaint_crop_input_image: bool = True
    # If hires fix is enabled in A1111, how should this ControlNet unit be applied.
    # The value is ignored if the generation is not using hires fix.
    hr_option: HiResFixOption = HiResFixOption.BOTH

    # Whether save the detected map of this unit. Setting this option to False prevents saving the
    # detected map or sending detected map along with generated images via API.
    # Currently the option is only accessible in API calls.
    save_detected_map: bool = True

    # Weight for each layer of ControlNet params.
    # For ControlNet:
    # - SD1.5: 13 weights (4 encoder block * 3 + 1 middle block)
    # - SDXL: 10 weights (3 encoder block * 3 + 1 middle block)
    # For T2IAdapter
    # - SD1.5: 5 weights (4 encoder block + 1 middle block)
    # - SDXL: 4 weights (3 encoder block + 1 middle block)
    # For IPAdapter
    # - SD15: 16 (6 input blocks + 9 output blocks + 1 middle block)
    # - SDXL: 11 weights (4 input blocks + 6 output blocks + 1 middle block)
    # Note1: Setting advanced weighting will disable `soft_injection`, i.e.
    # It is recommended to set ControlMode = BALANCED when using `advanced_weighting`.
    # Note2: The field `weight` is still used in some places, e.g. reference_only,
    # even advanced_weighting is set.
    advanced_weighting: Optional[List[float]] = None

    # The effective region mask that unit's effect should be restricted to.
    effective_region_mask: Optional[np.ndarray] = None

    @validator("effective_region_mask", pre=True)
    def parse_effective_region_mask(cls, value) -> np.ndarray:
        if isinstance(value, str):
            return cls.cls_decode_base64(value)
        assert isinstance(value, np.ndarray) or value is None
        return value

    # The weight mode for PuLID.
    # https://github.com/ToTheBeginning/PuLID
    pulid_mode: PuLIDMode = PuLIDMode.FIDELITY

    # ------- API only fields -------
    # The tensor input for ipadapter. When this field is set in the API,
    # the base64string will be interpret by torch.load to reconstruct ipadapter
    # preprocessor output.
    # Currently the option is only accessible in API calls.
    ipadapter_input: Optional[List[torch.Tensor]] = None

    @validator("ipadapter_input", pre=True)
    def parse_ipadapter_input(cls, value) -> Optional[List[torch.Tensor]]:
        if value is None:
            return None
        if isinstance(value, str):
            value = [value]
        result = [cls.cls_torch_load_base64(b) for b in value]
        assert result, "input cannot be empty"
        return result

    # The mask to be used on top of the image.
    mask: Optional[Any] = None

    # Backward compatible with animatediff impl.
    batch_mask_dir: Optional[str] = None

    @property
    def accepts_multiple_inputs(self) -> bool:
        """This unit can accept multiple input images."""
        return self.module in (
            "ip-adapter-auto",
            "ip-adapter_clip_sdxl",
            "ip-adapter_clip_sdxl_plus_vith",
            "ip-adapter_clip_sd15",
            "ip-adapter_face_id",
            "ip-adapter_face_id_plus",
            "ip-adapter_pulid",
            "instant_id_face_embedding",
        )

    @property
    def is_animate_diff_batch(self) -> bool:
        return getattr(self, "animatediff_batch", False)

    @property
    def uses_clip(self) -> bool:
        """Whether this unit uses clip preprocessor."""
        return any(
            (
                ("ip-adapter" in self.module and "face_id" not in self.module),
                self.module
                in ("clip_vision", "revision_clipvision", "revision_ignore_prompt"),
            )
        )

    @property
    def is_inpaint(self) -> bool:
        return "inpaint" in self.module

    def get_actual_preprocessor(self):
        if self.module == "ip-adapter-auto":
            return ControlNetUnit.cls_get_preprocessor(
                self.module
            ).get_preprocessor_by_model(self.model)
        return ControlNetUnit.cls_get_preprocessor(self.module)

    @classmethod
    def parse_image(cls, image) -> np.ndarray:
        if isinstance(image, np.ndarray):
            np_image = image
        elif isinstance(image, str):
            # Necessary for batch.
            if os.path.exists(image):
                np_image = np.array(Image.open(image)).astype("uint8")
            else:
                np_image = cls.cls_decode_base64(image)
        else:
            raise ValueError(f"Unrecognized image format {image}.")

        # [H, W] => [H, W, 3]
        if np_image.ndim == 2:
            np_image = np.stack([np_image, np_image, np_image], axis=-1)
        assert np_image.ndim == 3
        assert np_image.shape[2] == 3
        return np_image

    @classmethod
    def combine_image_and_mask(
        cls, np_image: np.ndarray, np_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """RGB + Alpha(Optional) => RGBA"""
        # TODO: Change protocol to use 255 as A channel value.
        # Note: mask is by default zeros, as both inpaint and
        # clip mask does extra work on masked area.
        np_mask = (np.zeros_like(np_image) if np_mask is None else np_mask)[:, :, 0:1]
        if np_image.shape[:2] != np_mask.shape[:2]:
            raise ValueError(
                f"image shape ({np_image.shape[:2]}) not aligned with mask shape ({np_mask.shape[:2]})"
            )
        return np.concatenate([np_image, np_mask], axis=2)  # [H, W, 4]

    @classmethod
    def legacy_field_alias(cls, values: dict) -> dict:
        ext_compat_keys = {
            "guidance": "guidance_end",
            "lowvram": "low_vram",
            "input_image": "image",
        }
        for alias, key in ext_compat_keys.items():
            if alias in values:
                assert key not in values, f"Conflict of field '{alias}' and '{key}'"
                values[key] = values[alias]
                cls.cls_logger.warn(
                    f"Deprecated alias '{alias}' detected. This field will be removed on 2024-06-01"
                    f"Please use '{key}' instead."
                )

        return values

    @classmethod
    def mask_alias(cls, values: dict) -> dict:
        """
        Field "mask_image" is the alias of field "mask".
        This is for compatibility with SD Forge API.
        """
        mask_image = values.get("mask_image")
        mask = values.get("mask")
        if mask_image is not None:
            if mask is not None:
                raise ValueError("Cannot specify both 'mask' and 'mask_image'!")
            values["mask"] = mask_image
        return values

    def get_input_images_rgba(self) -> Optional[List[np.ndarray]]:
        """
        RGBA images with potentially different size.
        Why we cannot have [B, H, W, C=4] here is that calculation of final
        resolution requires generation target's dimensions.

        Parse image with following formats.
        API
        - image = {"image": base64image, "mask": base64image,}
        - image = [image, mask]
        - image = (image, mask)
        - image = [{"image": ..., "mask": ...}, {"image": ..., "mask": ...}, ...]
        - image = base64image, mask = base64image

        UI:
        - image = {"image": np_image, "mask": np_image,}
        - image = np_image, mask = np_image
        """
        init_image = self.image
        init_mask = self.mask

        if init_image is None:
            assert init_mask is None
            return None

        if isinstance(init_image, (list, tuple)):
            if not init_image:
                raise ValueError(f"{init_image} is not a valid 'image' field value")
            if isinstance(init_image[0], dict):
                # [{"image": ..., "mask": ...}, {"image": ..., "mask": ...}, ...]
                images = init_image
            else:
                assert len(init_image) == 2
                # [image, mask]
                # (image, mask)
                images = [
                    {
                        "image": init_image[0],
                        "mask": init_image[1],
                    }
                ]
        elif isinstance(init_image, dict):
            # {"image": ..., "mask": ...}
            images = [init_image]
        elif isinstance(init_image, (str, np.ndarray)):
            # image = base64image, mask = base64image
            images = [
                {
                    "image": init_image,
                    "mask": init_mask,
                }
            ]
        else:
            raise ValueError(f"Unrecognized image field {init_image}")

        np_images = []
        for image_dict in images:
            assert isinstance(image_dict, dict)
            image = image_dict.get("image")
            mask = image_dict.get("mask")
            assert image is not None

            np_image = self.parse_image(image)
            np_mask = self.parse_image(mask) if mask is not None else None
            np_images.append(
                self.combine_image_and_mask(np_image, np_mask)
            )  # [H, W, 4]

        return np_images

    @classmethod
    def from_dict(cls, values: dict) -> ControlNetUnit:
        values = copy(values)
        values = cls.legacy_field_alias(values)
        values = cls.mask_alias(values)
        return ControlNetUnit(**values)

    @classmethod
    def from_infotext_args(cls, *args) -> ControlNetUnit:
        assert len(args) == len(ControlNetUnit.infotext_fields())
        return cls.from_dict(
            {k: v for k, v in zip(ControlNetUnit.infotext_fields(), args)}
        )

    @staticmethod
    def infotext_fields() -> Tuple[str]:
        """Fields that should be included in infotext.
        You should define a Gradio element with exact same name in ControlNetUiGroup
        as well, so that infotext can wire the value to correct field when pasting
        infotext.
        """
        return (
            "module",
            "model",
            "weight",
            "resize_mode",
            "processor_res",
            "threshold_a",
            "threshold_b",
            "guidance_start",
            "guidance_end",
            "pixel_perfect",
            "control_mode",
        )

    def serialize(self) -> str:
        """Serialize the unit for infotext."""
        infotext_dict = {
            field_to_displaytext(field): serialize_value(getattr(self, field))
            for field in ControlNetUnit.infotext_fields()
        }
        if not all(
            "," not in str(v) and ":" not in str(v) for v in infotext_dict.values()
        ):
            self.cls_logger.error(f"Unexpected tokens encountered:\n{infotext_dict}")
            return ""

        return ", ".join(f"{field}: {value}" for field, value in infotext_dict.items())

    @classmethod
    def parse(cls, text: str) -> ControlNetUnit:
        return ControlNetUnit(
            enabled=True,
            **{
                displaytext_to_field(key): parse_value(value)
                for item in text.split(",")
                for (key, value) in (item.strip().split(": "),)
            },
        )
