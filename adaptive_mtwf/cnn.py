"""A masked, variable-length CNN for generic numerical sequences."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class CNNConfig:
    num_classes: int
    in_channels: int = 2
    widths: tuple[int, ...] = (32, 64, 128, 256)
    kernels: tuple[int, ...] = (7, 5, 3, 3)
    hidden_size: int = 128
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if not self.widths or len(self.widths) != len(self.kernels):
            raise ValueError("Provide one kernel for each convolution width.")
        for value in (self.num_classes, self.in_channels, self.hidden_size, *self.widths, *self.kernels):
            if type(value) is not int or value < 1:
                raise ValueError("Channels, classes, kernels, and hidden size must be positive integers.")
        if any(kernel % 2 == 0 for kernel in self.kernels):
            raise ValueError("Odd kernels are required for centered, length-preserving convolution.")
        if not 0 <= self.dropout < 1:
            raise ValueError("Dropout must be in [0, 1).")


class _MaskedConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, dropout: float) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel, padding=kernel // 2)
        self.norm = nn.LayerNorm(out_channels)
        self.pool = nn.MaxPool1d(2, stride=2, ceil_mode=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        features = self.conv(inputs)
        # Normalize channels independently at each time position.
        features = self.norm(features.transpose(1, 2)).transpose(1, 2).relu()
        pooled = self.pool(features.masked_fill(~mask[:, None, :], -torch.inf))
        # A left-aligned pair is valid iff its first position is valid for a
        # prefix mask. Ceil pooling retains the final singleton position.
        pooled_mask = mask[:, ::2]
        pooled = pooled.masked_fill(~pooled_mask[:, None, :], 0)
        return self.dropout(pooled), pooled_mask


class VariableLengthCNN(nn.Module):
    """Accept float channels and a boolean right-padded prefix mask, return logits."""

    def __init__(self, config: CNNConfig) -> None:
        super().__init__()
        self.config = config
        inputs = (config.in_channels, *config.widths[:-1])
        self.blocks = nn.ModuleList(_MaskedConvBlock(left, right, kernel, config.dropout)
                                    for left, right, kernel in zip(inputs, config.widths, config.kernels))
        self.head = nn.Sequential(nn.Linear(2 * config.widths[-1], config.hidden_size), nn.ReLU(),
                                  nn.Dropout(config.dropout), nn.Linear(config.hidden_size, config.num_classes))

    def encode(self, inputs: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Return downsampled features and their valid-position mask."""
        if inputs.ndim != 3 or inputs.shape[1] != self.config.in_channels or min(inputs.shape) < 1:
            raise ValueError("Input must have nonempty shape (batch, configured channels, length).")
        if not inputs.is_floating_point():
            raise ValueError("Input channels must be floating point.")
        if mask.shape != (inputs.shape[0], inputs.shape[2]) or mask.dtype != torch.bool or mask.device != inputs.device:
            raise ValueError("Mask must be boolean with shape (batch, length) on the input device.")
        if not mask.any(dim=1).all().item():
            raise ValueError("All-padding samples are invalid.")
        if (mask[:, 1:] & ~mask[:, :-1]).any().item():
            raise ValueError("Mask must mark a contiguous valid prefix for each sample.")
        features = inputs.masked_fill(~mask[:, None, :], 0)
        if not features.isfinite().all().item():
            raise ValueError("Valid inputs must be finite.")
        for block in self.blocks:
            features, mask = block(features, mask)
        return features, mask

    def forward(self, inputs: Tensor, mask: Tensor) -> Tensor:
        features, mask = self.encode(inputs, mask)
        valid = mask[:, None, :]
        mean = features.masked_fill(~valid, 0).sum(dim=2) / mask.sum(dim=1, keepdim=True)
        maximum = features.masked_fill(~valid, -torch.inf).amax(dim=2)
        return self.head(torch.cat((mean, maximum), dim=1))


def make_bce_loss(num_classes: int, *, pos_weight=None, device: torch.device | str = "cpu") -> nn.BCEWithLogitsLoss:
    """Construct BCE with optional finite, nonnegative weights supplied by the caller."""
    weights = None
    if pos_weight is not None:
        weights = torch.as_tensor(pos_weight, dtype=torch.float32, device=device).detach().clone()
        if weights.shape != (num_classes,) or not (weights.isfinite() & (weights >= 0)).all().item():
            raise ValueError("pos_weight must contain one finite nonnegative value per class.")
    return nn.BCEWithLogitsLoss(pos_weight=weights)
