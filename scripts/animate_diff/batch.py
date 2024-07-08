from scripts.external_code import ControlNetUnit
from scripts.logging import logger
from modules.processing import StableDiffusionProcessing
from modules import shared


def add_animate_diff_batch_input(
    p: StableDiffusionProcessing, unit: ControlNetUnit
) -> ControlNetUnit:
    """AnimateDiff + ControlNet batch processing."""
    assert unit.is_animate_diff_batch

    batch_parameters = unit.batch_images.split("\n")
    batch_image_dir = batch_parameters[0]
    logger.info(
        f"AnimateDiff + ControlNet {unit.module} receive the following parameters:"
    )
    logger.info(f"\tbatch control images: {batch_image_dir}")
    for ad_cn_batch_parameter in batch_parameters[1:]:
        if ad_cn_batch_parameter.startswith("mask:"):
            unit.batch_mask_dir = ad_cn_batch_parameter[len("mask:") :].strip()
            logger.info(f"\tbatch control mask: {unit.batch_mask_dir}")
        elif ad_cn_batch_parameter.startswith("keyframe:"):
            unit.batch_keyframe_idx = ad_cn_batch_parameter[len("keyframe:") :].strip()
            unit.batch_keyframe_idx = [
                int(b_i.strip()) for b_i in unit.batch_keyframe_idx.split(",")
            ]
            logger.info(f"\tbatch control keyframe index: {unit.batch_keyframe_idx}")
    batch_image_files = shared.listfiles(batch_image_dir)
    for batch_modifier in getattr(unit, "batch_modifiers", []):
        batch_image_files = batch_modifier(batch_image_files, p)
    unit.batch_image_files = batch_image_files
    unit.image = []
    for idx, image_path in enumerate(batch_image_files):
        mask_path = None
        if getattr(unit, "batch_mask_dir", None) is not None:
            batch_mask_files = shared.listfiles(unit.batch_mask_dir)
            if len(batch_mask_files) >= len(batch_image_files):
                mask_path = batch_mask_files[idx]
            else:
                mask_path = batch_mask_files[0]
        unit.image.append(
            {
                "image": image_path,
                "mask": mask_path,
            }
        )
    return unit
