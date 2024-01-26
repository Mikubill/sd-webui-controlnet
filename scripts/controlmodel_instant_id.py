import torch
import numpy as np
from typing import NamedTuple


from scripts.controlmodel_ipadapter import ImageEmbed


class RawInstantIdInput(NamedTuple):
    """Raw input from insightface."""

    keypoints: np.ndarray
    embedding: torch.Tensor


class ResizedInstantIdInput(NamedTuple):
    """keypoints image get resized and convert to torch.Tensor."""

    resized_keypoints: torch.Tensor
    embedding: torch.Tensor


class InstantIdInput(NamedTuple):
    """embedding get projected in IPAdapter."""

    resized_keypoints: torch.Tensor
    projected_embedding: ImageEmbed
