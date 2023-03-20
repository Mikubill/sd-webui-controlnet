# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import numpy as np
import torch
from annotator.mmpkg.mmcv.runner import load_checkpoint
from torch import Tensor, nn

from .builder import FILTERS
from .filter import TemporalFilter


class SmoothNetResBlock(nn.Module):
    """Residual block module used in SmoothNet.

    Args:
        in_channels (int): Input channel number.
        hidden_channels (int): The hidden feature channel number.
        dropout (float): Dropout probability. Default: 0.5

    Shape:
        Input: (*, in_channels)
        Output: (*, in_channels)
    """

    def __init__(self, in_channels, hidden_channels, dropout=0.5):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, in_channels)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(p=dropout, inplace=True)

    def forward(self, x):
        identity = x
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.lrelu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = self.lrelu(x)

        out = x + identity
        return out


class SmoothNet(nn.Module):
    """SmoothNet is a plug-and-play temporal-only network to refine human
    poses. It works for 2d/3d/6d pose smoothing.

    "SmoothNet: A Plug-and-Play Network for Refining Human Poses in Videos",
    arXiv'2021. More details can be found in the `paper
    <https://arxiv.org/abs/2112.13715>`__ .

    Note:
        N: The batch size
        T: The temporal length of the pose sequence
        C: The total pose dimension (e.g. keypoint_number * keypoint_dim)

    Args:
        window_size (int): The size of the input window.
        output_size (int): The size of the output window.
        hidden_size (int): The hidden feature dimension in the encoder,
            the decoder and between residual blocks. Default: 512
        res_hidden_size (int): The hidden feature dimension inside the
            residual blocks. Default: 256
        num_blocks (int): The number of residual blocks. Default: 3
        dropout (float): Dropout probability. Default: 0.5

    Shape:
        Input: (N, C, T) the original pose sequence
        Output: (N, C, T) the smoothed pose sequence
    """

    def __init__(self,
                 window_size: int,
                 output_size: int,
                 hidden_size: int = 512,
                 res_hidden_size: int = 256,
                 num_blocks: int = 3,
                 dropout: float = 0.5):
        super().__init__()
        self.window_size = window_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.res_hidden_size = res_hidden_size
        self.num_blocks = num_blocks
        self.dropout = dropout

        assert output_size <= window_size, (
            'The output size should be less than or equal to the window size.',
            f' Got output_size=={output_size} and window_size=={window_size}')

        # Build encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(window_size, hidden_size),
            nn.LeakyReLU(0.1, inplace=True))

        # Build residual blocks
        res_blocks = []
        for _ in range(num_blocks):
            res_blocks.append(
                SmoothNetResBlock(
                    in_channels=hidden_size,
                    hidden_channels=res_hidden_size,
                    dropout=dropout))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Build decoder layers
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        N, C, T = x.shape
        num_windows = T - self.window_size + 1

        assert T >= self.window_size, (
            'Input sequence length must be no less than the window size. ',
            f'Got x.shape[2]=={T} and window_size=={self.window_size}')

        # Unfold x to obtain input sliding windows
        # [N, C, num_windows, window_size]
        x = x.unfold(2, self.window_size, 1)

        # Forward layers
        x = self.encoder(x)
        x = self.res_blocks(x)
        x = self.decoder(x)  # [N, C, num_windows, output_size]

        # Accumulate output ensembles
        out = x.new_zeros(N, C, T)
        count = x.new_zeros(T)

        for t in range(num_windows):
            out[..., t:t + self.output_size] += x[:, :, t]
            count[t:t + self.output_size] += 1.0

        return out.div(count)


@FILTERS.register_module(name=['SmoothNetFilter', 'SmoothNet', 'smoothnet'])
class SmoothNetFilter(TemporalFilter):
    """Apply SmoothNet filter.

    "SmoothNet: A Plug-and-Play Network for Refining Human Poses in Videos",
    arXiv'2021. More details can be found in the `paper
    <https://arxiv.org/abs/2112.13715>`__ .

    Args:
        window_size (int): The size of the filter window. It's also the
            window_size of SmoothNet model.
        output_size (int): The output window size of SmoothNet model.
        checkpoint (str): The checkpoint file of the pretrained SmoothNet
            model. Please note that `checkpoint` should be matched with
            `window_size` and `output_size`.
        hidden_size (int): SmoothNet argument. See :class:`SmoothNet` for
            details. Default: 512
        hidden_res_size (int): SmoothNet argument. See :class:`SmoothNet`
            for details. Default: 256
        num_blocks (int): SmoothNet argument. See :class:`SmoothNet` for
            details. Default: 3
        device (str): Device for model inference. Default: 'cpu'
        root_index (int, optional): If not None, relative keypoint coordinates
            will be calculated as the SmoothNet input, by centering the
            keypoints around the root point. The model output will be
            converted back to absolute coordinates. Default: None
    """

    def __init__(
        self,
        window_size: int,
        output_size: int,
        checkpoint: Optional[str] = None,
        hidden_size: int = 512,
        res_hidden_size: int = 256,
        num_blocks: int = 3,
        device: str = 'cpu',
        root_index: Optional[int] = None,
    ):
        super().__init__(window_size)
        self.device = device
        self.root_index = root_index
        self.smoothnet = SmoothNet(window_size, output_size, hidden_size,
                                   res_hidden_size, num_blocks)
        if checkpoint:
            load_checkpoint(self.smoothnet, checkpoint)
        self.smoothnet.to(device)
        self.smoothnet.eval()

        for p in self.smoothnet.parameters():
            p.requires_grad_(False)

    def __call__(self, x: np.ndarray):
        assert x.ndim == 3, ('Input should be an array with shape [T, K, C]'
                             f', but got invalid shape {x.shape}')

        root_index = self.root_index
        if root_index is not None:
            x_root = x[:, root_index:root_index + 1]
            x = np.delete(x, root_index, axis=1)
            x = x - x_root

        T, K, C = x.shape

        if T < self.window_size:
            # Skip smoothing if the input length is less than the window size
            smoothed = x
        else:
            dtype = x.dtype

            # Convert to tensor and forward the model
            with torch.no_grad():
                x = torch.tensor(x, dtype=torch.float32, device=self.device)
                x = x.view(1, T, K * C).permute(0, 2, 1)  # to [1, KC, T]
                smoothed = self.smoothnet(x)  # in shape [1, KC, T]

            # Convert model output back to input shape and format
            smoothed = smoothed.permute(0, 2, 1).view(T, K, C)  # to [T, K, C]
            smoothed = smoothed.cpu().numpy().astype(dtype)  # to numpy.ndarray

        if root_index is not None:
            smoothed += x_root
            smoothed = np.concatenate(
                (smoothed[:, :root_index], x_root, smoothed[:, root_index:]),
                axis=1)

        return smoothed
