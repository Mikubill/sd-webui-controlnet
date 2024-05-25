# Original version from:
# https://github.com/comfyanonymous/ComfyUI/blob/ffc4b7c30e35eb2773ace52a0b00e0ca5c1f4362/comfy/model_patcher.py

from __future__ import annotations
from collections import defaultdict
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    ClassVar,
    Union,
    NamedTuple,
)

import torch
import logging
from pydantic import BaseModel, Field, validator


def set_attr(obj, attr, value):
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    prev = getattr(obj, attrs[-1])
    setattr(obj, attrs[-1], value)
    return prev


def get_attr(obj, attr):
    attrs = attr.split(".")
    for name in attrs:
        obj = getattr(obj, name)
    return obj


def module_size(module: torch.nn.Module) -> int:
    """Get the memory size of a module."""
    module_mem = 0
    sd = module.state_dict()
    for k in sd:
        t = sd[k]
        module_mem += t.nelement() * t.element_size()
    return module_mem


def apply_weight_decompose(dora_scale, weight):
    weight_norm = (
        weight.transpose(0, 1)
        .reshape(weight.shape[1], -1)
        .norm(dim=1, keepdim=True)
        .reshape(weight.shape[1], *[1] * (weight.dim() - 1))
        .transpose(0, 1)
    )

    return weight * (dora_scale / weight_norm).type(weight.dtype)


# Represent the model to patch.
ModelType = TypeVar("ModelType", bound=torch.nn.Module)
# Represent the sub-module of the model to patch.
ModuleType = TypeVar("ModuleType", bound=torch.nn.Module)


class PatchType(Enum):
    DIFF = "diff"
    LORA = "lora"
    LOKR = "lokr"
    LOHA = "loha"
    GLORA = "glora"


class LoRAWeight(NamedTuple):
    down: torch.Tensor
    up: torch.Tensor
    alpha_scale: Optional[float] = None
    # locon mid weights
    mid: Optional[torch.Tensor] = None
    dora_scale: Optional[torch.Tensor] = None


WeightPatchWeight = Union[torch.Tensor, LoRAWeight, Tuple[torch.Tensor, ...]]


def _unimplemented_func(*args, **kwargs):
    raise NotImplementedError("Not implemented.")


CastToDeviceFunc = Callable[[torch.Tensor, torch.device, torch.dtype], torch.Tensor]


class WeightPatch(BaseModel):
    """Patch to apply on model weight."""

    class Config:
        arbitrary_types_allowed = True
        extra = "ignore"

    cls_logger: ClassVar[logging.Logger] = logging.Logger("WeightPatch")
    cls_cast_to_device: ClassVar[CastToDeviceFunc] = lambda t, device, dtype: t.to(
        device, dtype
    )

    weight: WeightPatchWeight
    patch_type: PatchType = PatchType.DIFF
    # The scale applied on patch weight value.
    alpha: float = 1.0
    # The scale applied on the model weight value.
    strength_model: float = 1.0

    def apply(
        self, model_weight: torch.Tensor, key: Optional[str] = None
    ) -> torch.Tensor:
        """Apply the patch to model weight."""
        if self.strength_model != 1.0:
            model_weight *= self.strength_model

        try:
            if self.patch_type == PatchType.DIFF:
                assert isinstance(self.weight, torch.Tensor)
                return self._patch_diff(model_weight)
            elif self.patch_type == PatchType.LORA:
                assert isinstance(self.weight, LoRAWeight)
                return self._patch_lora(model_weight)
            else:
                raise NotImplementedError(
                    f"Patch type {self.patch_type} is not implemented."
                )
        except ValueError as e:
            logging.error("ERROR {} {} {}".format(self.patch_type, key, e))
            return model_weight

    def _patch_diff(self, model_weight: torch.Tensor) -> torch.Tensor:
        """Apply the diff patch to model weight."""
        if self.alpha != 0.0:
            if self.weight.shape != model_weight.shape:
                raise ValueError(
                    "WARNING SHAPE MISMATCH WEIGHT NOT MERGED {} != {}".format(
                        self.weight.shape, model_weight.shape
                    )
                )
            else:
                return model_weight + self.alpha * self.weight.to(model_weight.device)
        return model_weight

    def _patch_lora(self, model_weight: torch.Tensor) -> torch.Tensor:
        """Apply the lora/locon patch to model weight."""
        v: LoRAWeight = self.weight
        alpha = self.alpha
        weight = model_weight

        mat1 = WeightPatch.cls_cast_to_device(v.down, weight.device, torch.float32)
        mat2 = WeightPatch.cls_cast_to_device(v.up, weight.device, torch.float32)
        dora_scale = v.dora_scale

        if v.alpha_scale is not None:
            alpha *= v.alpha_scale / mat2.shape[0]
        if v.mid is not None:
            # locon mid weights, hopefully the math is fine because I didn't properly test it
            mat3 = WeightPatch.cls_cast_to_device(v.mid, weight.device, torch.float32)
            final_shape = [mat2.shape[1], mat2.shape[0], mat3.shape[2], mat3.shape[3]]
            mat2 = (
                torch.mm(
                    mat2.transpose(0, 1).flatten(start_dim=1),
                    mat3.transpose(0, 1).flatten(start_dim=1),
                )
                .reshape(final_shape)
                .transpose(0, 1)
            )
        weight += (
            (alpha * torch.mm(mat1.flatten(start_dim=1), mat2.flatten(start_dim=1)))
            .reshape(weight.shape)
            .type(weight.dtype)
        )
        if dora_scale is not None:
            weight = apply_weight_decompose(
                WeightPatch.cls_cast_to_device(
                    dora_scale, weight.device, torch.float32
                ),
                weight,
            )
        return weight


class ModulePatch(BaseModel, Generic[ModuleType]):
    """Patch to replace a module in the model."""

    apply_func: Callable[[ModuleType], ModuleType]


class ModelPatcher(BaseModel, Generic[ModelType]):
    class Config:
        arbitrary_types_allowed = True
        extra = "ignore"

    cls_logger: ClassVar[logging.Logger] = logging.Logger("ModelPatcher")
    cls_strict: ClassVar[bool] = False

    # The managed model of the model patcher.
    model: ModelType = Field(immutable=True)
    # The device to run inference on.
    load_device: torch.device = Field(immutable=True)
    # The device to offload the model to.
    offload_device: torch.device = Field(immutable=True)
    # Whether to update weight in place.
    weight_inplace_update: bool = Field(immutable=True)

    # The current device the model is stored on.
    current_device: torch.device = None

    @validator("current_device", pre=True, always=True)
    def set_current_device(cls, v, values):
        return values.get("offload_device") if v is None else v

    # The size of the model in number of bytes.
    model_size: int = None

    @validator("model_size", pre=True, always=True)
    def set_model_size(cls, v, values):
        model: ModelType = values.get("model")
        return module_size(model) if v is None else v

    model_keys: Set[str] = None

    @validator("model_keys", pre=True, always=True)
    def set_model_keys(cls, v, values):
        model: ModelType = values.get("model")
        return set(model.state_dict().keys()) if v is None else v

    # Patches applied to module weights.
    weight_patches: Dict[str, List[WeightPatch]] = defaultdict(list)
    # Store weights before patching.
    _weight_backup: Dict[str, torch.Tensor] = {}

    # Patches applied to model's torch modules.
    module_patches: Dict[str, List[ModulePatch]] = defaultdict(list)
    # Store modules before patching.
    _module_backup: Dict[str, torch.nn.Module] = {}

    def add_weight_patch(self, key: str, weight_patch: WeightPatch) -> bool:
        if key not in self.model_keys:
            if self.cls_strict:
                raise ValueError(f"Key {key} not found in model.")
            else:
                return False
        self.weight_patches[key].append(weight_patch)

    def add_weight_patches(self, weight_patches: Dict[str, WeightPatch]) -> List[str]:
        return [
            key
            for key, weight_patch in weight_patches.items()
            if self.add_weight_patch(key, weight_patch)
        ]

    def add_patches(
        self,
        patches: Dict[str, Union[Tuple[torch.Tensor], Tuple[str, torch.Tensor]]],
        strength_patch: float = 1.0,
        strength_model: float = 1.0,
    ):
        """@Deprecated
        Legacy function interface from ComfyUI ModelPatcher.
        """

        def parse_value(
            v: Union[Tuple[torch.Tensor], Tuple[str, torch.Tensor]]
        ) -> Tuple[torch.Tensor, PatchType]:
            if len(v) == 1:
                return v, PatchType.DIFF
            else:
                assert len(v) == 2, f"Invalid patch value {v}."
                return v[1], PatchType(v[0])

        return self.add_weight_patches(
            {
                key: WeightPatch(
                    *parse_value(value),
                    alpha=strength_patch,
                    strength_model=strength_model,
                )
                for key, value in patches.items()
            }
        )

    def to(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.model.to(device, dtype)
        self.current_device = device

    def get_attr(self, key: str) -> Optional[Any]:
        return get_attr(self.model, key)

    def set_attr(self, key: str, value: Any) -> Any:
        return set_attr(self.model, key, value)

    def set_attr_param(self, attr, value):
        return self.set_attr(attr, torch.nn.Parameter(value, requires_grad=False))

    def copy_to_param(self, attr, value):
        """inplace update tensor instead of replacing it"""
        attrs = attr.split(".")
        obj = self.model
        for name in attrs[:-1]:
            obj = getattr(obj, name)
        prev = getattr(obj, attrs[-1])
        prev.data.copy_(value)

    def add_module_patch(self, key: str, module_patch: ModulePatch) -> bool:
        target_module = self.get_attr(key)
        if target_module is None:
            return False

        self.module_patches[key].append(module_patch)
        return True

    def _patch_modules(self):
        for key, module_patches in self.module_patches.items():
            old_module = self.get_attr(key)
            self._module_backup[key] = old_module
            for module_patch in module_patches:
                self.set_attr(key, module_patch.apply_func(old_module))

    def _patch_weights(self):
        for key, weight_patches in self.weight_patches.items():
            assert key in self.model_keys, f"Key {key} not found in model."
            old_weight = self.get_attr(key)
            self._weight_backup[key] = old_weight

            new_weight = old_weight
            for weight_patch in weight_patches:
                new_weight = weight_patch.apply(new_weight, key)

            if self.weight_inplace_update:
                self.set_attr_param(key, new_weight)
            else:
                self.copy_to_param(key, new_weight)

    def patch_model(self, patch_weights: bool = True):
        self._patch_modules()
        if patch_weights:
            self._patch_weights()
        return self.model

    def _unpatch_weights(self):
        for k, v in self._weight_backup.items():
            if self.weight_inplace_update:
                self.copy_to_param(k, v)
            else:
                self.set_attr_param(k, v)
        self._weight_backup.clear()

    def _unpatch_modules(self):
        for k, v in self._module_backup.items():
            self.set_attr(k, v)
        self._module_backup.clear()

    def unpatch_model(self, unpatch_weights=True):
        if unpatch_weights:
            self._unpatch_weights()
        self._unpatch_modules()
