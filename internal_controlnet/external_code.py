import base64
import io
from dataclasses import dataclass
from enum import Enum
from copy import copy
from typing import List, Any, Optional, Union, Tuple, Dict
import torch
import numpy as np

from modules import scripts, processing, shared
from modules.safe import unsafe_torch_load
from scripts import global_state
from scripts.logging import logger
from scripts.enums import HiResFixOption, PuLIDMode, ControlMode, ResizeMode
from scripts.supported_preprocessor import Preprocessor, PreprocessorParameter

from modules.api import api


def get_api_version() -> int:
    return 2


resize_mode_aliases = {
    "Inner Fit (Scale to Fit)": "Crop and Resize",
    "Outer Fit (Shrink to Fit)": "Resize and Fill",
    "Scale to Fit (Inner Fit)": "Crop and Resize",
    "Envelope (Outer Fit)": "Resize and Fill",
}


def resize_mode_from_value(value: Union[str, int, ResizeMode]) -> ResizeMode:
    if isinstance(value, str):
        return ResizeMode(resize_mode_aliases.get(value, value))
    elif isinstance(value, int):
        assert value >= 0
        if value == 3:  # 'Just Resize (Latent upscale)'
            return ResizeMode.RESIZE

        if value >= len(ResizeMode):
            logger.warning(
                f"Unrecognized ResizeMode int value {value}. Fall back to RESIZE."
            )
            return ResizeMode.RESIZE

        return [e for e in ResizeMode][value]
    else:
        return value


def control_mode_from_value(value: Union[str, int, ControlMode]) -> ControlMode:
    if isinstance(value, str):
        return ControlMode(value)
    elif isinstance(value, int):
        return [e for e in ControlMode][value]
    else:
        return value


def visualize_inpaint_mask(img):
    if img.ndim == 3 and img.shape[2] == 4:
        result = img.copy()
        mask = result[:, :, 3]
        mask = 255 - mask // 2
        result[:, :, 3] = mask
        return np.ascontiguousarray(result.copy())
    return img


def pixel_perfect_resolution(
    image: np.ndarray,
    target_H: int,
    target_W: int,
    resize_mode: ResizeMode,
) -> int:
    """
    Calculate the estimated resolution for resizing an image while preserving aspect ratio.

    The function first calculates scaling factors for height and width of the image based on the target
    height and width. Then, based on the chosen resize mode, it either takes the smaller or the larger
    scaling factor to estimate the new resolution.

    If the resize mode is OUTER_FIT, the function uses the smaller scaling factor, ensuring the whole image
    fits within the target dimensions, potentially leaving some empty space.

    If the resize mode is not OUTER_FIT, the function uses the larger scaling factor, ensuring the target
    dimensions are fully filled, potentially cropping the image.

    After calculating the estimated resolution, the function prints some debugging information.

    Args:
        image (np.ndarray): A 3D numpy array representing an image. The dimensions represent [height, width, channels].
        target_H (int): The target height for the image.
        target_W (int): The target width for the image.
        resize_mode (ResizeMode): The mode for resizing.

    Returns:
        int: The estimated resolution after resizing.
    """
    raw_H, raw_W, _ = image.shape

    k0 = float(target_H) / float(raw_H)
    k1 = float(target_W) / float(raw_W)

    if resize_mode == ResizeMode.OUTER_FIT:
        estimation = min(k0, k1) * float(min(raw_H, raw_W))
    else:
        estimation = max(k0, k1) * float(min(raw_H, raw_W))

    logger.debug("Pixel Perfect Computation:")
    logger.debug(f"resize_mode = {resize_mode}")
    logger.debug(f"raw_H = {raw_H}")
    logger.debug(f"raw_W = {raw_W}")
    logger.debug(f"target_H = {target_H}")
    logger.debug(f"target_W = {target_W}")
    logger.debug(f"estimation = {estimation}")

    return int(np.round(estimation))


InputImage = Union[np.ndarray, str]
InputImage = Union[Dict[str, InputImage], Tuple[InputImage, InputImage], InputImage]


@dataclass
class ControlNetUnit:
    """
    Represents an entire ControlNet processing unit.
    """

    enabled: bool = True
    module: str = "none"
    model: str = "None"
    weight: float = 1.0
    image: Optional[Union[InputImage, List[InputImage]]] = None
    resize_mode: Union[ResizeMode, int, str] = ResizeMode.INNER_FIT
    low_vram: bool = False
    processor_res: int = -1
    threshold_a: float = -1
    threshold_b: float = -1
    guidance_start: float = 0.0
    guidance_end: float = 1.0
    pixel_perfect: bool = False
    control_mode: Union[ControlMode, int, str] = ControlMode.BALANCED
    # Whether to crop input image based on A1111 img2img mask. This flag is only used when `inpaint area`
    # in A1111 is set to `Only masked`. In API, this correspond to `inpaint_full_res = True`.
    inpaint_crop_input_image: bool = True
    # If hires fix is enabled in A1111, how should this ControlNet unit be applied.
    # The value is ignored if the generation is not using hires fix.
    hr_option: Union[HiResFixOption, int, str] = HiResFixOption.BOTH

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

    # The weight mode for PuLID.
    # https://github.com/ToTheBeginning/PuLID
    pulid_mode: PuLIDMode = PuLIDMode.FIDELITY

    # The tensor input for ipadapter. When this field is set in the API,
    # the base64string will be interpret by torch.load to reconstruct ipadapter
    # preprocessor output.
    # Currently the option is only accessible in API calls.
    ipadapter_input: Optional[List[Any]] = None

    def __eq__(self, other):
        if not isinstance(other, ControlNetUnit):
            return False

        return vars(self) == vars(other)

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

    @staticmethod
    def infotext_excluded_fields() -> List[str]:
        return [
            "image",
            "enabled",
            # API-only fields.
            "advanced_weighting",
            "ipadapter_input",
            # End of API-only fields.
            # Note: "inpaint_crop_image" is img2img inpaint only flag, which does not
            # provide much information when restoring the unit.
            "inpaint_crop_input_image",
            "effective_region_mask",
            "pulid_mode",
        ]

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

    def bound_check_params(self) -> None:
        """
        Checks and corrects negative parameters in ControlNetUnit 'unit' in place.
        Parameters 'processor_res', 'threshold_a', 'threshold_b' are reset to
        their default values if negative.
        """
        preprocessor = Preprocessor.get_preprocessor(self.module)
        for unit_param, param in zip(
            ("processor_res", "threshold_a", "threshold_b"),
            ("slider_resolution", "slider_1", "slider_2"),
        ):
            value = getattr(self, unit_param)
            cfg: PreprocessorParameter = getattr(preprocessor, param)
            if value < 0:
                setattr(self, unit_param, cfg.value)
                logger.info(
                    f"[{self.module}.{unit_param}] Invalid value({value}), using default value {cfg.value}."
                )

    def get_actual_preprocessor(self) -> Preprocessor:
        if self.module == "ip-adapter-auto":
            return Preprocessor.get_preprocessor(self.module).get_preprocessor_by_model(
                self.model
            )
        return Preprocessor.get_preprocessor(self.module)


def to_base64_nparray(encoding: str):
    """
    Convert a base64 image into the image type the extension uses
    """

    return np.array(api.decode_base64_to_image(encoding)).astype("uint8")


def get_all_units_in_processing(
    p: processing.StableDiffusionProcessing,
) -> List[ControlNetUnit]:
    """
    Fetch ControlNet processing units from a StableDiffusionProcessing.
    """

    return get_all_units(p.scripts, p.script_args)


def get_all_units(
    script_runner: scripts.ScriptRunner, script_args: List[Any]
) -> List[ControlNetUnit]:
    """
    Fetch ControlNet processing units from an existing script runner.
    Use this function to fetch units from the list of all scripts arguments.
    """

    cn_script = find_cn_script(script_runner)
    if cn_script:
        return get_all_units_from(script_args[cn_script.args_from : cn_script.args_to])

    return []


def get_all_units_from(script_args: List[Any]) -> List[ControlNetUnit]:
    """
    Fetch ControlNet processing units from ControlNet script arguments.
    Use `external_code.get_all_units` to fetch units from the list of all scripts arguments.
    """

    def is_stale_unit(script_arg: Any) -> bool:
        """Returns whether the script_arg is potentially an stale version of
        ControlNetUnit created before module reload."""
        return "ControlNetUnit" in type(script_arg).__name__ and not isinstance(
            script_arg, ControlNetUnit
        )

    def is_controlnet_unit(script_arg: Any) -> bool:
        """Returns whether the script_arg is ControlNetUnit or anything that
        can be treated like ControlNetUnit."""
        return isinstance(script_arg, (ControlNetUnit, dict)) or (
            hasattr(script_arg, "__dict__")
            and set(vars(ControlNetUnit()).keys()).issubset(
                set(vars(script_arg).keys())
            )
        )

    all_units = [
        to_processing_unit(script_arg)
        for script_arg in script_args
        if is_controlnet_unit(script_arg)
    ]
    if not all_units:
        logger.warning(
            "No ControlNetUnit detected in args. It is very likely that you are having an extension conflict."
            f"Here are args received by ControlNet: {script_args}."
        )
    if any(is_stale_unit(script_arg) for script_arg in script_args):
        logger.debug(
            "Stale version of ControlNetUnit detected. The ControlNetUnit received"
            "by ControlNet is created before the newest load of ControlNet extension."
            "They will still be used by ControlNet as long as they provide same fields"
            "defined in the newest version of ControlNetUnit."
        )

    return all_units


def get_single_unit_from(
    script_args: List[Any], index: int = 0
) -> Optional[ControlNetUnit]:
    """
    Fetch a single ControlNet processing unit from ControlNet script arguments.
    The list must not contain script positional arguments. It must only contain processing units.
    """

    i = 0
    while i < len(script_args) and index >= 0:
        if index == 0 and script_args[i] is not None:
            return to_processing_unit(script_args[i])
        i += 1

        index -= 1

    return None


def get_max_models_num():
    """
    Fetch the maximum number of allowed ControlNet models.
    """

    max_models_num = shared.opts.data.get("control_net_unit_count", 3)
    return max_models_num


def to_processing_unit(unit: Union[Dict[str, Any], ControlNetUnit]) -> ControlNetUnit:
    """
    Convert different types to processing unit.
    If `unit` is a dict, alternative keys are supported. See `ext_compat_keys` in implementation for details.
    """

    ext_compat_keys = {
        "guessmode": "guess_mode",
        "guidance": "guidance_end",
        "lowvram": "low_vram",
        "input_image": "image",
    }

    if isinstance(unit, dict):
        unit = {ext_compat_keys.get(k, k): v for k, v in unit.items()}

        # Handle mask
        mask = None
        if "mask" in unit:
            mask = unit["mask"]
            del unit["mask"]

        if "mask_image" in unit:
            mask = unit["mask_image"]
            del unit["mask_image"]

        if "image" in unit and not isinstance(unit["image"], dict):
            unit["image"] = (
                {"image": unit["image"], "mask": mask}
                if mask is not None
                else unit["image"] if unit["image"] else None
            )

        # Parse ipadapter_input
        if "ipadapter_input" in unit and unit["ipadapter_input"] is not None:

            def decode_base64(b: str) -> torch.Tensor:
                decoded_bytes = base64.b64decode(b)
                return unsafe_torch_load(io.BytesIO(decoded_bytes))

            if isinstance(unit["ipadapter_input"], str):
                unit["ipadapter_input"] = [unit["ipadapter_input"]]

            unit["ipadapter_input"] = [
                decode_base64(b) for b in unit["ipadapter_input"]
            ]

        if unit.get("effective_region_mask", None) is not None:
            base64img = unit["effective_region_mask"]
            assert isinstance(base64img, str)
            unit["effective_region_mask"] = to_base64_nparray(base64img)

        if "guess_mode" in unit:
            logger.warning(
                "Guess Mode is removed since 1.1.136. Please use Control Mode instead."
            )

        for k in unit.keys():
            if k not in vars(ControlNetUnit):
                logger.warn(f"Received unrecognized key '{k}' in API.")

        unit = ControlNetUnit(
            **{k: v for k, v in unit.items() if k in vars(ControlNetUnit).keys()}
        )

    # temporary, check #602
    # assert isinstance(unit, ControlNetUnit), f'bad argument to controlnet extension: {unit}\nexpected Union[dict[str, Any], ControlNetUnit]'
    return unit


def update_cn_script_in_processing(
    p: processing.StableDiffusionProcessing,
    cn_units: List[ControlNetUnit],
    **_kwargs,  # for backwards compatibility
):
    """
    Update the arguments of the ControlNet script in `p.script_args` in place, reading from `cn_units`.
    `cn_units` and its elements are not modified. You can call this function repeatedly, as many times as you want.

    Does not update `p.script_args` if any of the folling is true:
    - ControlNet is not present in `p.scripts`
    - `p.script_args` is not filled with script arguments for scripts that are processed before ControlNet
    """
    p.script_args = update_cn_script(p.scripts, p.script_args_value, cn_units)


def update_cn_script(
    script_runner: scripts.ScriptRunner,
    script_args: Union[Tuple[Any], List[Any]],
    cn_units: List[ControlNetUnit],
) -> Union[Tuple[Any], List[Any]]:
    """
    Returns: The updated `script_args` with given `cn_units` used as ControlNet
    script args.

    Does not update `script_args` if any of the folling is true:
    - ControlNet is not present in `script_runner`
    - `script_args` is not filled with script arguments for scripts that are
    processed before ControlNet
    """
    script_args_type = type(script_args)
    assert script_args_type in (tuple, list), script_args_type
    updated_script_args = list(copy(script_args))

    cn_script = find_cn_script(script_runner)

    if cn_script is None or len(script_args) < cn_script.args_from:
        return script_args

    # fill in remaining parameters to satisfy max models, just in case script needs it.
    max_models = shared.opts.data.get("control_net_unit_count", 3)
    cn_units = cn_units + [ControlNetUnit(enabled=False)] * max(
        max_models - len(cn_units), 0
    )

    cn_script_args_diff = 0
    for script in script_runner.alwayson_scripts:
        if script is cn_script:
            cn_script_args_diff = len(cn_units) - (
                cn_script.args_to - cn_script.args_from
            )
            updated_script_args[script.args_from : script.args_to] = cn_units
            script.args_to = script.args_from + len(cn_units)
        else:
            script.args_from += cn_script_args_diff
            script.args_to += cn_script_args_diff

    return script_args_type(updated_script_args)


def update_cn_script_in_place(
    script_runner: scripts.ScriptRunner,
    script_args: List[Any],
    cn_units: List[ControlNetUnit],
    **_kwargs,  # for backwards compatibility
):
    """
    @Deprecated(Raises assertion error if script_args passed in is Tuple)

    Update the arguments of the ControlNet script in `script_args` in place, reading from `cn_units`.
    `cn_units` and its elements are not modified. You can call this function repeatedly, as many times as you want.

    Does not update `script_args` if any of the folling is true:
    - ControlNet is not present in `script_runner`
    - `script_args` is not filled with script arguments for scripts that are processed before ControlNet
    """
    assert isinstance(script_args, list), type(script_args)

    cn_script = find_cn_script(script_runner)
    if cn_script is None or len(script_args) < cn_script.args_from:
        return

    # fill in remaining parameters to satisfy max models, just in case script needs it.
    max_models = shared.opts.data.get("control_net_unit_count", 3)
    cn_units = cn_units + [ControlNetUnit(enabled=False)] * max(
        max_models - len(cn_units), 0
    )

    cn_script_args_diff = 0
    for script in script_runner.alwayson_scripts:
        if script is cn_script:
            cn_script_args_diff = len(cn_units) - (
                cn_script.args_to - cn_script.args_from
            )
            script_args[script.args_from : script.args_to] = cn_units
            script.args_to = script.args_from + len(cn_units)
        else:
            script.args_from += cn_script_args_diff
            script.args_to += cn_script_args_diff


def get_models(update: bool = False) -> List[str]:
    """
    Fetch the list of available models.
    Each value is a valid candidate of `ControlNetUnit.model`.

    Keyword arguments:
    update -- Whether to refresh the list from disk. (default False)
    """

    if update:
        global_state.update_cn_models()

    return list(global_state.cn_models_names.values())


def get_modules(alias_names: bool = False) -> List[str]:
    """
    Fetch the list of available preprocessors.
    Each value is a valid candidate of `ControlNetUnit.module`.

    Keyword arguments:
    alias_names -- Whether to get the ui alias names instead of internal keys
    """
    return [
        (p.label if alias_names else p.name)
        for p in Preprocessor.get_sorted_preprocessors()
    ]


def get_modules_detail(alias_names: bool = False) -> Dict[str, Any]:
    """
    get the detail of all preprocessors including
    sliders: the slider config in Auto1111 webUI

    Keyword arguments:
    alias_names -- Whether to get the module detail with alias names instead of internal keys
    """

    _module_detail = {}
    _module_list = get_modules(False)
    _module_list_alias = get_modules(True)

    _output_list = _module_list if not alias_names else _module_list_alias
    for module_name in _output_list:
        preprocessor = Preprocessor.get_preprocessor(module_name)
        assert preprocessor is not None
        _module_detail[module_name] = dict(
            model_free=preprocessor.do_not_need_model,
            sliders=[
                s.api_json
                for s in (
                    preprocessor.slider_resolution,
                    preprocessor.slider_1,
                    preprocessor.slider_2,
                    preprocessor.slider_3,
                )
                if s.visible
            ],
        )

    return _module_detail


def find_cn_script(script_runner: scripts.ScriptRunner) -> Optional[scripts.Script]:
    """
    Find the ControlNet script in `script_runner`. Returns `None` if `script_runner` does not contain a ControlNet script.
    """

    if script_runner is None:
        return None

    for script in script_runner.alwayson_scripts:
        if is_cn_script(script):
            return script


def is_cn_script(script: scripts.Script) -> bool:
    """
    Determine whether `script` is a ControlNet script.
    """

    return script.title().lower() == "controlnet"
