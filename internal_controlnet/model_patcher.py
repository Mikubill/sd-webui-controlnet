# Original version from:
# https://github.com/comfyanonymous/ComfyUI/blob/ffc4b7c30e35eb2773ace52a0b00e0ca5c1f4362/comfy/model_patcher.py

from __future__ import annotations
from collections import defaultdict
from enum import Enum
from typing import (
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

    def get_module(self, key: str) -> torch.nn.Module:
        pass

    def add_module_patch(self, key: str, module_patch: ModulePatch) -> bool:
        pass

    def patch_weight_to_device(self, key, device_to=None):
        if key not in self.patches:
            return

        weight = comfy.utils.get_attr(self.model, key)

        inplace_update = self.weight_inplace_update

        if key not in self.backup:
            self.backup[key] = weight.to(
                device=self.offload_device, copy=inplace_update
            )

        if device_to is not None:
            temp_weight = self.cls_cast_to_device(
                weight, device_to, torch.float32, copy=True
            )
        else:
            temp_weight = weight.to(torch.float32, copy=True)
        out_weight = self.calculate_weight(self.patches[key], temp_weight, key).to(
            weight.dtype
        )
        if inplace_update:
            comfy.utils.copy_to_param(self.model, key, out_weight)
        else:
            comfy.utils.set_attr_param(self.model, key, out_weight)

    def patch_model(self, device_to=None, patch_weights=True):
        for k in self.object_patches:
            old = comfy.utils.set_attr(self.model, k, self.object_patches[k])
            if k not in self.object_patches_backup:
                self.object_patches_backup[k] = old

        if patch_weights:
            model_sd = self.model_state_dict()
            for key in self.patches:
                if key not in model_sd:
                    logging.warning(
                        "could not patch. key doesn't exist in model: {}".format(key)
                    )
                    continue

                self.patch_weight_to_device(key, device_to)

            if device_to is not None:
                self.model.to(device_to)
                self.current_device = device_to

        return self.model

    def patch_model_lowvram(
        self, device_to=None, lowvram_model_memory=0, force_patch_weights=False
    ):
        self.patch_model(device_to, patch_weights=False)

        logging.info(
            "loading in lowvram mode {}".format(lowvram_model_memory / (1024 * 1024))
        )

        class LowVramPatch:
            def __init__(self, key, model_patcher):
                self.key = key
                self.model_patcher = model_patcher

            def __call__(self, weight):
                return self.model_patcher.calculate_weight(
                    self.model_patcher.patches[self.key], weight, self.key
                )

        mem_counter = 0
        patch_counter = 0
        for n, m in self.model.named_modules():
            lowvram_weight = False
            if hasattr(m, "comfy_cast_weights"):
                module_mem = comfy.model_management.module_size(m)
                if mem_counter + module_mem >= lowvram_model_memory:
                    lowvram_weight = True

            weight_key = "{}.weight".format(n)
            bias_key = "{}.bias".format(n)

            if lowvram_weight:
                if weight_key in self.patches:
                    if force_patch_weights:
                        self.patch_weight_to_device(weight_key)
                    else:
                        m.weight_function = LowVramPatch(weight_key, self)
                        patch_counter += 1
                if bias_key in self.patches:
                    if force_patch_weights:
                        self.patch_weight_to_device(bias_key)
                    else:
                        m.bias_function = LowVramPatch(bias_key, self)
                        patch_counter += 1

                m.prev_comfy_cast_weights = m.comfy_cast_weights
                m.comfy_cast_weights = True
            else:
                if hasattr(m, "weight"):
                    self.patch_weight_to_device(weight_key, device_to)
                    self.patch_weight_to_device(bias_key, device_to)
                    m.to(device_to)
                    mem_counter += comfy.model_management.module_size(m)
                    logging.debug("lowvram: loaded module regularly {}".format(m))

        self.model_lowvram = True
        self.lowvram_patch_counter = patch_counter
        return self.model

    def calculate_weight(self, patches, weight, key):
        for p in patches:
            alpha = p[0]
            v = p[1]
            strength_model = p[2]

            if strength_model != 1.0:
                weight *= strength_model

            if isinstance(v, list):
                v = (self.calculate_weight(v[1:], v[0].clone(), key),)

            if len(v) == 1:
                patch_type = "diff"
            elif len(v) == 2:
                patch_type = v[0]
                v = v[1]

            if patch_type == "diff":
                w1 = v[0]
                if alpha != 0.0:
                    if w1.shape != weight.shape:
                        logging.warning(
                            "WARNING SHAPE MISMATCH {} WEIGHT NOT MERGED {} != {}".format(
                                key, w1.shape, weight.shape
                            )
                        )
                    else:
                        weight += alpha * self.cls_cast_to_device(
                            w1, weight.device, weight.dtype
                        )
            elif patch_type == "lora":  # lora/locon
                mat1 = self.cls_cast_to_device(v[0], weight.device, torch.float32)
                mat2 = self.cls_cast_to_device(v[1], weight.device, torch.float32)
                dora_scale = v[4]
                if v[2] is not None:
                    alpha *= v[2] / mat2.shape[0]
                if v[3] is not None:
                    # locon mid weights, hopefully the math is fine because I didn't properly test it
                    mat3 = self.cls_cast_to_device(v[3], weight.device, torch.float32)
                    final_shape = [
                        mat2.shape[1],
                        mat2.shape[0],
                        mat3.shape[2],
                        mat3.shape[3],
                    ]
                    mat2 = (
                        torch.mm(
                            mat2.transpose(0, 1).flatten(start_dim=1),
                            mat3.transpose(0, 1).flatten(start_dim=1),
                        )
                        .reshape(final_shape)
                        .transpose(0, 1)
                    )
                try:
                    weight += (
                        (
                            alpha
                            * torch.mm(
                                mat1.flatten(start_dim=1), mat2.flatten(start_dim=1)
                            )
                        )
                        .reshape(weight.shape)
                        .type(weight.dtype)
                    )
                    if dora_scale is not None:
                        weight = apply_weight_decompose(
                            self.cls_cast_to_device(
                                dora_scale, weight.device, torch.float32
                            ),
                            weight,
                        )
                except Exception as e:
                    logging.error("ERROR {} {} {}".format(patch_type, key, e))
            elif patch_type == "lokr":
                w1 = v[0]
                w2 = v[1]
                w1_a = v[3]
                w1_b = v[4]
                w2_a = v[5]
                w2_b = v[6]
                t2 = v[7]
                dora_scale = v[8]
                dim = None

                if w1 is None:
                    dim = w1_b.shape[0]
                    w1 = torch.mm(
                        self.cls_cast_to_device(w1_a, weight.device, torch.float32),
                        self.cls_cast_to_device(w1_b, weight.device, torch.float32),
                    )
                else:
                    w1 = self.cls_cast_to_device(w1, weight.device, torch.float32)

                if w2 is None:
                    dim = w2_b.shape[0]
                    if t2 is None:
                        w2 = torch.mm(
                            self.cls_cast_to_device(w2_a, weight.device, torch.float32),
                            self.cls_cast_to_device(w2_b, weight.device, torch.float32),
                        )
                    else:
                        w2 = torch.einsum(
                            "i j k l, j r, i p -> p r k l",
                            self.cls_cast_to_device(t2, weight.device, torch.float32),
                            self.cls_cast_to_device(w2_b, weight.device, torch.float32),
                            self.cls_cast_to_device(w2_a, weight.device, torch.float32),
                        )
                else:
                    w2 = self.cls_cast_to_device(w2, weight.device, torch.float32)

                if len(w2.shape) == 4:
                    w1 = w1.unsqueeze(2).unsqueeze(2)
                if v[2] is not None and dim is not None:
                    alpha *= v[2] / dim

                try:
                    weight += alpha * torch.kron(w1, w2).reshape(weight.shape).type(
                        weight.dtype
                    )
                    if dora_scale is not None:
                        weight = apply_weight_decompose(
                            self.cls_cast_to_device(
                                dora_scale, weight.device, torch.float32
                            ),
                            weight,
                        )
                except Exception as e:
                    logging.error("ERROR {} {} {}".format(patch_type, key, e))
            elif patch_type == "loha":
                w1a = v[0]
                w1b = v[1]
                if v[2] is not None:
                    alpha *= v[2] / w1b.shape[0]
                w2a = v[3]
                w2b = v[4]
                dora_scale = v[7]
                if v[5] is not None:  # cp decomposition
                    t1 = v[5]
                    t2 = v[6]
                    m1 = torch.einsum(
                        "i j k l, j r, i p -> p r k l",
                        self.cls_cast_to_device(t1, weight.device, torch.float32),
                        self.cls_cast_to_device(w1b, weight.device, torch.float32),
                        self.cls_cast_to_device(w1a, weight.device, torch.float32),
                    )

                    m2 = torch.einsum(
                        "i j k l, j r, i p -> p r k l",
                        self.cls_cast_to_device(t2, weight.device, torch.float32),
                        self.cls_cast_to_device(w2b, weight.device, torch.float32),
                        self.cls_cast_to_device(w2a, weight.device, torch.float32),
                    )
                else:
                    m1 = torch.mm(
                        self.cls_cast_to_device(w1a, weight.device, torch.float32),
                        self.cls_cast_to_device(w1b, weight.device, torch.float32),
                    )
                    m2 = torch.mm(
                        self.cls_cast_to_device(w2a, weight.device, torch.float32),
                        self.cls_cast_to_device(w2b, weight.device, torch.float32),
                    )

                try:
                    weight += (alpha * m1 * m2).reshape(weight.shape).type(weight.dtype)
                    if dora_scale is not None:
                        weight = apply_weight_decompose(
                            self.cls_cast_to_device(
                                dora_scale, weight.device, torch.float32
                            ),
                            weight,
                        )
                except Exception as e:
                    logging.error("ERROR {} {} {}".format(patch_type, key, e))
            elif patch_type == "glora":
                if v[4] is not None:
                    alpha *= v[4] / v[0].shape[0]

                dora_scale = v[5]

                a1 = self.cls_cast_to_device(
                    v[0].flatten(start_dim=1), weight.device, torch.float32
                )
                a2 = self.cls_cast_to_device(
                    v[1].flatten(start_dim=1), weight.device, torch.float32
                )
                b1 = self.cls_cast_to_device(
                    v[2].flatten(start_dim=1), weight.device, torch.float32
                )
                b2 = self.cls_cast_to_device(
                    v[3].flatten(start_dim=1), weight.device, torch.float32
                )

                try:
                    weight += (
                        (
                            (
                                torch.mm(b2, b1)
                                + torch.mm(
                                    torch.mm(weight.flatten(start_dim=1), a2), a1
                                )
                            )
                            * alpha
                        )
                        .reshape(weight.shape)
                        .type(weight.dtype)
                    )
                    if dora_scale is not None:
                        weight = apply_weight_decompose(
                            self.cls_cast_to_device(
                                dora_scale, weight.device, torch.float32
                            ),
                            weight,
                        )
                except Exception as e:
                    logging.error("ERROR {} {} {}".format(patch_type, key, e))
            else:
                logging.warning(
                    "patch type not recognized {} {}".format(patch_type, key)
                )

        return weight

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if unpatch_weights:
            if self.model_lowvram:
                for m in self.model.modules():
                    if hasattr(m, "prev_comfy_cast_weights"):
                        m.comfy_cast_weights = m.prev_comfy_cast_weights
                        del m.prev_comfy_cast_weights
                    m.weight_function = None
                    m.bias_function = None

                self.model_lowvram = False
                self.lowvram_patch_counter = 0

            keys = list(self.backup.keys())

            if self.weight_inplace_update:
                for k in keys:
                    comfy.utils.copy_to_param(self.model, k, self.backup[k])
            else:
                for k in keys:
                    comfy.utils.set_attr_param(self.model, k, self.backup[k])

            self.backup.clear()

            if device_to is not None:
                self.model.to(device_to)
                self.current_device = device_to

        keys = list(self.object_patches_backup.keys())
        for k in keys:
            comfy.utils.set_attr(self.model, k, self.object_patches_backup[k])

        self.object_patches_backup.clear()
