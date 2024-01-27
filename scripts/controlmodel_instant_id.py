import torch
from typing import NamedTuple


from scripts.controlmodel_ipadapter import ImageEmbed


class InstantIdControlNetInput(NamedTuple):
    """The ControlNet input for InstantID control model type. Unlike normal
    ControlNet which accepts text prompt as ControlNet's crossattn condition,
    InstantID's ControlNet accepts projected face embedding as ControlNet's
    crossattn condition."""

    resized_keypoints: torch.Tensor
    projected_embedding: ImageEmbed
