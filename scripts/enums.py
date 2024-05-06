from enum import Enum
from typing import List, NamedTuple
from functools import lru_cache


class UnetBlockType(Enum):
    INPUT = "input"
    OUTPUT = "output"
    MIDDLE = "middle"


class TransformerID(NamedTuple):
    block_type: UnetBlockType
    # The id of the block the transformer is in. Not all blocks have cross attn.
    block_id: int
    # The index of transformer within the block.
    # A block can have multiple transformers in SDXL.
    block_index: int
    # The call index of transformer if in a single step of diffusion.
    transformer_index: int


class TransformerIDResult(NamedTuple):
    input_ids: List[TransformerID]
    output_ids: List[TransformerID]
    middle_ids: List[TransformerID]

    def get(self, idx: int) -> TransformerID:
        return self.to_list()[idx]

    def to_list(self) -> List[TransformerID]:
        return sorted(
            self.input_ids + self.output_ids + self.middle_ids,
            key=lambda i: i.transformer_index,
        )


class StableDiffusionVersion(Enum):
    """The version family of stable diffusion model."""

    UNKNOWN = 0
    SD1x = 1
    SD2x = 2
    SDXL = 3

    @staticmethod
    def detect_from_model_name(model_name: str) -> "StableDiffusionVersion":
        """Based on the model name provided, guess what stable diffusion version it is.
        This might not be accurate without actually inspect the file content.
        """
        if any(f"sd{v}" in model_name.lower() for v in ("14", "15", "16")):
            return StableDiffusionVersion.SD1x

        if "sd21" in model_name or "2.1" in model_name:
            return StableDiffusionVersion.SD2x

        if "xl" in model_name.lower():
            return StableDiffusionVersion.SDXL

        return StableDiffusionVersion.UNKNOWN

    def encoder_block_num(self) -> int:
        if self in (
            StableDiffusionVersion.SD1x,
            StableDiffusionVersion.SD2x,
            StableDiffusionVersion.UNKNOWN,
        ):
            return 12
        else:
            return 9  # SDXL

    def controlnet_layer_num(self) -> int:
        return self.encoder_block_num() + 1

    @property
    def transformer_block_num(self) -> int:
        """Number of blocks that has cross attn transformers in unet."""
        if self in (
            StableDiffusionVersion.SD1x,
            StableDiffusionVersion.SD2x,
            StableDiffusionVersion.UNKNOWN,
        ):
            return 16
        else:
            return 11  # SDXL

    @property
    @lru_cache(maxsize=None)
    def transformer_ids(self) -> List[TransformerID]:
        """id of blocks that have cross attention"""
        if self in (
            StableDiffusionVersion.SD1x,
            StableDiffusionVersion.SD2x,
            StableDiffusionVersion.UNKNOWN,
        ):
            transformer_index = 0
            input_ids = []
            for block_id in [1, 2, 4, 5, 7, 8]:
                input_ids.append(
                    TransformerID(UnetBlockType.INPUT, block_id, 0, transformer_index)
                )
                transformer_index += 1
            middle_id = TransformerID(UnetBlockType.MIDDLE, 0, 0, transformer_index)
            transformer_index += 1
            output_ids = []
            for block_id in [3, 4, 5, 6, 7, 8, 9, 10, 11]:
                input_ids.append(
                    TransformerID(UnetBlockType.OUTPUT, block_id, 0, transformer_index)
                )
                transformer_index += 1
            return TransformerIDResult(input_ids, output_ids, [middle_id])
        else:
            # SDXL
            transformer_index = 0
            input_ids = []
            for block_id in [4, 5, 7, 8]:
                block_indices = (
                    range(2) if block_id in [4, 5] else range(10)
                )  # transformer_depth
                for index in block_indices:
                    input_ids.append(
                        TransformerID(
                            UnetBlockType.INPUT, block_id, index, transformer_index
                        )
                    )
                transformer_index += 1

            middle_ids = [
                TransformerID(UnetBlockType.MIDDLE, 0, index, transformer_index)
                for index in range(10)
            ]
            transformer_index += 1

            output_ids = []
            for block_id in range(6):
                block_indices = (
                    range(2) if block_id in [3, 4, 5] else range(10)
                )  # transformer_depth
                for index in block_indices:
                    output_ids.append(
                        TransformerID(
                            UnetBlockType.OUTPUT, block_id, index, transformer_index
                        )
                    )
                transformer_index += 1
            return TransformerIDResult(input_ids, output_ids, middle_ids)

    def is_compatible_with(self, other: "StableDiffusionVersion") -> bool:
        """Incompatible only when one of version is SDXL and other is not."""
        return (
            any(v == StableDiffusionVersion.UNKNOWN for v in [self, other])
            or sum(v == StableDiffusionVersion.SDXL for v in [self, other]) != 1
        )


class ControlModelType(Enum):
    """
    The type of Control Models (supported or not).
    """

    ControlNet = "ControlNet, Lvmin Zhang"
    T2I_Adapter = "T2I_Adapter, Chong Mou"
    T2I_StyleAdapter = "T2I_StyleAdapter, Chong Mou"
    T2I_CoAdapter = "T2I_CoAdapter, Chong Mou"
    MasaCtrl = "MasaCtrl, Mingdeng Cao"
    GLIGEN = "GLIGEN, Yuheng Li"
    AttentionInjection = "AttentionInjection, Lvmin Zhang"  # A simple attention injection written by Lvmin
    StableSR = "StableSR, Jianyi Wang"
    PromptDiffusion = "PromptDiffusion, Zhendong Wang"
    ControlLoRA = "ControlLoRA, Wu Hecong"
    ReVision = "ReVision, Stability"
    IPAdapter = "IPAdapter, Hu Ye"
    Controlllite = "Controlllite, Kohya"
    InstantID = "InstantID, Qixun Wang"
    SparseCtrl = "SparseCtrl, Yuwei Guo"

    @property
    def is_controlnet(self) -> bool:
        """Returns whether the control model should be treated as ControlNet."""
        return self in (
            ControlModelType.ControlNet,
            ControlModelType.ControlLoRA,
            ControlModelType.InstantID,
        )

    @property
    def allow_context_sharing(self) -> bool:
        """Returns whether this control model type allows the same PlugableControlModel
        object map to multiple ControlNetUnit.
        Both IPAdapter and Controlllite have unit specific input (clip/image) stored
        on the model object during inference. Sharing the context means that the input
        set earlier gets lost.
        """
        return self not in (
            ControlModelType.IPAdapter,
            ControlModelType.Controlllite,
        )

    @property
    def supports_effective_region_mask(self) -> bool:
        return (
            self
            in {
                ControlModelType.IPAdapter,
                ControlModelType.T2I_Adapter,
            }
            or self.is_controlnet
        )


# Written by Lvmin
class AutoMachine(Enum):
    """
    Lvmin's algorithm for Attention/AdaIn AutoMachine States.
    """

    Read = "Read"
    Write = "Write"
    StyleAlign = "StyleAlign"


class HiResFixOption(Enum):
    BOTH = "Both"
    LOW_RES_ONLY = "Low res only"
    HIGH_RES_ONLY = "High res only"


class InputMode(Enum):
    # Single image to a single ControlNet unit.
    SIMPLE = "simple"
    # Input is a directory. N generations. Each generation takes 1 input image
    # from the directory.
    BATCH = "batch"
    # Input is a directory. 1 generation. Each generation takes N input image
    # from the directory.
    MERGE = "merge"


class PuLIDMode(Enum):
    FIDELITY = "Fidelity"
    STYLE = "Extremely style"


class ControlMode(Enum):
    """
    The improved guess mode.
    """

    BALANCED = "Balanced"
    PROMPT = "My prompt is more important"
    CONTROL = "ControlNet is more important"


class BatchOption(Enum):
    DEFAULT = "All ControlNet units for all images in a batch"
    SEPARATE = "Each ControlNet unit for each image in a batch"


class ResizeMode(Enum):
    """
    Resize modes for ControlNet input images.
    """

    RESIZE = "Just Resize"
    INNER_FIT = "Crop and Resize"
    OUTER_FIT = "Resize and Fill"

    def int_value(self):
        if self == ResizeMode.RESIZE:
            return 0
        elif self == ResizeMode.INNER_FIT:
            return 1
        elif self == ResizeMode.OUTER_FIT:
            return 2
        assert False, "NOTREACHED"
