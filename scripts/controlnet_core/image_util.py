import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps

from modules.processing import StableDiffusionProcessing


def resize_and_pad(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # Ensure mask is boolean
    mask = mask > 0

    # Find bounding box of the mask
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    # Crop and resize image to the mask's bounding box
    cropped_image = image[ymin : ymax + 1, xmin : xmax + 1]
    mask_height, mask_width = ymax - ymin + 1, xmax - xmin + 1

    # Resize image to fit the mask's bounding box
    resized_image = cv2.resize(cropped_image, (mask_width, mask_height))

    # Create a new image with the same size as the mask, filled with zeros (black)
    new_image = np.zeros_like(image)

    # Place the resized image in the bounding box of the mask
    new_image[ymin : ymax + 1, xmin : xmax + 1] = resized_image

    return new_image


def prepare_mask(mask: Image.Image, p: StableDiffusionProcessing) -> Image.Image:
    """
    Prepare an image mask for the inpainting process.

    This function takes as input a PIL Image object and an instance of the
    StableDiffusionProcessing class, and performs the following steps to prepare the mask:

    1. Convert the mask to grayscale (mode "L").
    2. If the 'inpainting_mask_invert' attribute of the processing instance is True,
       invert the mask colors.
    3. If the 'mask_blur' attribute of the processing instance is greater than 0,
       apply a Gaussian blur to the mask with a radius equal to 'mask_blur'.

    Args:
        mask (Image.Image): The input mask as a PIL Image object.
        p (processing.StableDiffusionProcessing): An instance of the StableDiffusionProcessing class
                                                   containing the processing parameters.

    Returns:
        mask (Image.Image): The prepared mask as a PIL Image object.
    """
    mask = mask.convert("L")
    if getattr(p, "inpainting_mask_invert", False):
        mask = ImageOps.invert(mask)

    if hasattr(p, "mask_blur_x"):
        if getattr(p, "mask_blur_x", 0) > 0:
            np_mask = np.array(mask)
            kernel_size = 2 * int(2.5 * p.mask_blur_x + 0.5) + 1
            np_mask = cv2.GaussianBlur(np_mask, (kernel_size, 1), p.mask_blur_x)
            mask = Image.fromarray(np_mask)
        if getattr(p, "mask_blur_y", 0) > 0:
            np_mask = np.array(mask)
            kernel_size = 2 * int(2.5 * p.mask_blur_y + 0.5) + 1
            np_mask = cv2.GaussianBlur(np_mask, (1, kernel_size), p.mask_blur_y)
            mask = Image.fromarray(np_mask)
    else:
        if getattr(p, "mask_blur", 0) > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(p.mask_blur))

    return mask
