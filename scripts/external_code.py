from typing import List, Any, Optional, Union, Tuple, Dict
from modules import scripts, processing, shared
from scripts.controlnet import ResizeMode, update_cn_models, cn_models_names, PARAM_COUNT
import numpy as np


"""
Resize modes for ControlNet input images.
"""
ResizeMode = ResizeMode


class ControlNetUnit:
    """
    Represents an entire ControlNet processing unit.
    """

    def __init__(
        self,
        enabled: bool=True,
        module: Optional[str]=None,
        model: Optional[str]=None,
        weight: float=1.0,
        image: Optional[Union[Dict[str, np.ndarray], Tuple[np.ndarray, np.ndarray], np.ndarray]]=None,
        invert_image: bool=False,
        resize_mode: Union[ResizeMode, int, str]=ResizeMode.INNER_FIT,
        rgbbgr_mode: bool=False,
        low_vram: bool=False,
        processor_res: int=64,
        threshold_a: float=64,
        threshold_b: float=64,
        guidance_start: float=0.0,
        guidance_end: float=1.0,
        guess_mode: bool=True,
    ):
        if image is not None:
            if isinstance(image, tuple):
                image = {'image': image[0], 'mask': image[1]}
            elif isinstance(image, np.ndarray):
                image = {'image': image, 'mask': np.zeros_like(image, dtype=np.uint8)}

            while len(image['mask'].shape) < 3:
                image['mask'] = image['mask'][..., np.newaxis]

        self.enabled = enabled
        self.module = module
        self.model = model
        self.weight = weight
        self.image = image
        self.invert_image = invert_image
        self.resize_mode = resize_mode
        self.rgbbgr_mode = rgbbgr_mode
        self.low_vram = low_vram
        self.processor_res = processor_res
        self.threshold_a = threshold_a
        self.threshold_b = threshold_b
        self.guidance_start = guidance_start
        self.guidance_end = guidance_end
        self.guess_mode = guess_mode


def get_all_units_in_processing(p: processing.StableDiffusionProcessing) -> List[ControlNetUnit]:
    """
    Fetch ControlNet processing units from a StableDiffusionProcessing.
    """

    return get_all_units(p.scripts, p.script_args)


def get_all_units(script_runner: scripts.ScriptRunner, script_args: List[Any]) -> List[ControlNetUnit]:
    """
    Fetch ControlNet processing units from an existing script runner.
    Use this function to fetch units from the list of all scripts arguments.
    """

    cn_script = find_cn_script(script_runner)
    if cn_script:
        return get_all_units_from(script_args[cn_script.args_from:cn_script.args_to])

    return []


def get_all_units_from(script_args: List[Any], strip_positional_args=True) -> List[ControlNetUnit]:
    """
    Fetch ControlNet processing units from ControlNet script arguments.
    Use `external_code.get_all_units` to fetch units from the list of all scripts arguments.

    Keyword arguments:
    strip_positional_args -- Whether positional arguments are present in `script_args`. (default True)
    """

    if strip_positional_args:
        script_args = script_args[2:]

    res = []
    for i in range(len(script_args) // PARAM_COUNT):
        res.append(get_single_unit_from(script_args, i))

    return res


def get_single_unit_from(script_args: List[Any], index: int=0) -> ControlNetUnit:
    """
    Fetch a single ControlNet processing unit from ControlNet script arguments.
    The list must not contain script positional arguments. It must only consist of flattened processing unit parameters.
    """

    index_from = index * PARAM_COUNT
    index_to = index_from + PARAM_COUNT
    return ControlNetUnit(*script_args[index_from:index_to])


def update_cn_script_in_processing(
    p: processing.StableDiffusionProcessing,
    cn_units: List[ControlNetUnit],
    is_img2img: Optional[bool] = None,
    is_ui: Optional[bool] = None
):
    """
    Update the arguments of the ControlNet script in `p.script_args` in place, reading from `cn_units`.
    `cn_units` and its elements are not modified. You can call this function repeatedly, as many times as you want.

    Does not update `p.script_args` if any of the folling is true:
    - ControlNet is not present in `p.scripts`
    - `p.script_args` is not filled with script arguments for scripts that are processed before ControlNet

    Keyword arguments:
    is_img2img -- whether to run the script as img2img. In general, this should be set to the appropriate value depending on the `StableDiffusionProcessing` subclass used for generating. If set to None, do not change existing value. (default None)
    is_ui -- whether to run the script as if from the gradio interface. If set to None, do not change existing value. (default None)
    """

    cn_units_type = type(cn_units) if type(cn_units) in (list, tuple) else list
    script_args = list(p.script_args)
    update_cn_script_in_place(p.scripts, script_args, cn_units, is_img2img, is_ui)
    p.script_args = cn_units_type(script_args)


def update_cn_script_in_place(
    script_runner: scripts.ScriptRunner,
    script_args: List[Any],
    cn_units: List[ControlNetUnit],
    is_img2img: Optional[bool] = None,
    is_ui: Optional[bool] = None,
):
    """
    Update the arguments of the ControlNet script in `script_args` in place, reading from `cn_units`.
    `cn_units` and its elements are not modified. You can call this function repeatedly, as many times as you want.

    Does not update `script_args` if any of the folling is true:
    - ControlNet is not present in `script_runner`
    - `script_args` is not filled with script arguments for scripts that are processed before ControlNet

    Keyword arguments:
    is_img2img -- whether to run the script as img2img. In general, this should be set to the appropriate value depending on the `StableDiffusionProcessing` subclass used for generating. If set to None, do not change existing value. (default None)
    is_ui -- whether to run the script as if from the gradio interface. If set to None, do not change existing value. (default None)
    """

    cn_script = find_cn_script(script_runner)
    if cn_script is None or len(script_args) < cn_script.args_from:
        return

    cn_script_has_args = len(script_args[cn_script.args_from:cn_script.args_to]) > 0
    if is_img2img is None:
        is_img2img = script_args[cn_script.args_from] if cn_script_has_args else False
    if is_ui is None:
        is_ui = script_args[cn_script.args_from + 1] if cn_script_has_args else False

    # fill in remaining parameters to satisfy max models, just in case script needs it.
    max_models = shared.opts.data.get("control_net_max_models_num", 1)
    cn_units = cn_units + [ControlNetUnit(enabled=False)] * max(max_models - len(cn_units), 0)

    flattened_cn_args: List[Any] = [is_img2img, is_ui]
    for unit in cn_units:
        flattened_cn_args.extend((
            unit.enabled,
            unit.module if unit.module is not None else "none",
            unit.model if unit.model is not None else "None",
            unit.weight,
            unit.image,
            unit.invert_image,
            unit.resize_mode,
            unit.rgbbgr_mode,
            unit.low_vram,
            unit.processor_res,
            unit.threshold_a,
            unit.threshold_b,
            unit.guidance_start,
            unit.guidance_end,
            unit.guess_mode))

    cn_script_args_diff = 0
    for script in script_runner.alwayson_scripts:
        if script is cn_script:
            cn_script_args_diff = len(flattened_cn_args) - (cn_script.args_to - cn_script.args_from)
            script_args[script.args_from:script.args_to] = flattened_cn_args
            script.args_to = script.args_from + len(flattened_cn_args)
        else:
            script.args_from += cn_script_args_diff
            script.args_to += cn_script_args_diff


def get_models(update: bool=False) -> List[str]:
    """
    Fetch the list of available models.
    Each value is a valid candidate of `ControlNetUnit.model`.

    Keyword arguments:
    update -- Whether to refresh the list from disk. (default False)
    """

    if update:
        update_cn_models()

    return list(cn_models_names.values())


def find_cn_script(script_runner: scripts.ScriptRunner) -> Optional[scripts.Script]:
    """
    Find the ControlNet script in `script_runner`. Returns `None` if `script_runner` does not contain a ControlNet script.
    """

    for script in script_runner.alwayson_scripts:
        if is_cn_script(script):
            return script


def is_cn_script(script: scripts.Script) -> bool:
    """
    Determine whether `script` is a ControlNet script.
    """

    return script.title().lower() == 'controlnet'
