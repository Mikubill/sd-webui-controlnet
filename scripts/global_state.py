import os.path
import stat
from collections import OrderedDict

from modules import shared, scripts, sd_models
from modules.paths import models_path

from scripts.enums import StableDiffusionVersion
from scripts.supported_preprocessor import Preprocessor

from typing import Dict, Tuple, List

CN_MODEL_EXTS = [".pt", ".pth", ".ckpt", ".safetensors", ".bin"]
cn_models_dir = os.path.join(models_path, "ControlNet")
cn_models_dir_old = os.path.join(scripts.basedir(), "models")
cn_models = OrderedDict()      # "My_Lora(abcd1234)" -> C:/path/to/model.safetensors
cn_models_names = {}  # "my_lora" -> "My_Lora(abcd1234)"


default_detectedmap_dir = os.path.join("detected_maps")
script_dir = scripts.basedir()

os.makedirs(cn_models_dir, exist_ok=True)


def traverse_all_files(curr_path, model_list):
    f_list = [
        (os.path.join(curr_path, entry.name), entry.stat())
        for entry in os.scandir(curr_path)
        if os.path.isdir(curr_path)
    ]
    for f_info in f_list:
        fname, fstat = f_info
        if os.path.splitext(fname)[1] in CN_MODEL_EXTS:
            model_list.append(f_info)
        elif stat.S_ISDIR(fstat.st_mode):
            model_list = traverse_all_files(fname, model_list)
    return model_list


def get_all_models(sort_by, filter_by, path):
    res = OrderedDict()
    fileinfos = traverse_all_files(path, [])
    filter_by = filter_by.strip(" ")
    if len(filter_by) != 0:
        fileinfos = [x for x in fileinfos if filter_by.lower()
                     in os.path.basename(x[0]).lower()]
    if sort_by == "name":
        fileinfos = sorted(fileinfos, key=lambda x: os.path.basename(x[0]))
    elif sort_by == "date":
        fileinfos = sorted(fileinfos, key=lambda x: -x[1].st_mtime)
    elif sort_by == "path name":
        fileinfos = sorted(fileinfos)

    for finfo in fileinfos:
        filename = finfo[0]
        name = os.path.splitext(os.path.basename(filename))[0]
        # Prevent a hypothetical "None.pt" from being listed.
        if name != "None":
            res[name + f" [{sd_models.model_hash(filename)}]"] = filename

    return res


def update_cn_models():
    cn_models.clear()
    ext_dirs = (shared.opts.data.get("control_net_models_path", None), getattr(shared.cmd_opts, 'controlnet_dir', None))
    extra_lora_paths = (extra_lora_path for extra_lora_path in ext_dirs
                if extra_lora_path is not None and os.path.exists(extra_lora_path))
    paths = [cn_models_dir, cn_models_dir_old, *extra_lora_paths]

    for path in paths:
        sort_by = shared.opts.data.get(
            "control_net_models_sort_models_by", "name")
        filter_by = shared.opts.data.get("control_net_models_name_filter", "")
        found = get_all_models(sort_by, filter_by, path)
        cn_models.update({**found, **cn_models})

    # insert "None" at the beginning of `cn_models` in-place
    cn_models_copy = OrderedDict(cn_models)
    cn_models.clear()
    cn_models.update({**{"None": None}, **cn_models_copy})

    cn_models_names.clear()
    for name_and_hash, filename in cn_models.items():
        if filename is None:
            continue
        name = os.path.splitext(os.path.basename(filename))[0].lower()
        cn_models_names[name] = name_and_hash


def get_sd_version() -> StableDiffusionVersion:
    if hasattr(shared.sd_model, 'is_sdxl'):
        if shared.sd_model.is_sdxl:
            return StableDiffusionVersion.SDXL
        elif shared.sd_model.is_sd2:
            return StableDiffusionVersion.SD2x
        elif shared.sd_model.is_sd1:
            return StableDiffusionVersion.SD1x
        else:
            return StableDiffusionVersion.UNKNOWN

    # backward compability for webui < 1.5.0
    else:
        if hasattr(shared.sd_model, 'conditioner'):
            return StableDiffusionVersion.SDXL
        elif hasattr(shared.sd_model.cond_stage_model, 'model'):
            return StableDiffusionVersion.SD2x
        else:
            return StableDiffusionVersion.SD1x



def select_control_type(
    control_type: str,
    sd_version: StableDiffusionVersion = StableDiffusionVersion.UNKNOWN,
    cn_models: Dict = cn_models, # Override or testing
) -> Tuple[List[str], List[str], str, str]:
    pattern = control_type.lower()
    all_models = list(cn_models.keys())

    if pattern == "all":
        return [
            [p.label for p in Preprocessor.get_sorted_preprocessors()],
            all_models,
            'none', #default option
            "None"  #default model
        ]

    filtered_model_list = [
        model for model in all_models
        if model.lower() == "none" or
        ((
            pattern in model.lower() or
            any(a in model.lower() for a in Preprocessor.tag_to_filters(control_type))
        ) and (
            sd_version.is_compatible_with(StableDiffusionVersion.detect_from_model_name(model))
        ))
    ]
    assert len(filtered_model_list) > 0, "'None' model should always be available."
    if len(filtered_model_list) == 1:
        default_model = "None"
    else:
        default_model = filtered_model_list[1]
        for x in filtered_model_list:
            if "11" in x.split("[")[0]:
                default_model = x
                break

    return (
        [p.label for p in Preprocessor.get_filtered_preprocessors(control_type)],
        filtered_model_list,
        Preprocessor.get_default_preprocessor(control_type).label,
        default_model
    )


