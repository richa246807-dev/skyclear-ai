"""PatchGAN discriminator used during Model 1 training."""

from __future__ import annotations

import torch
from torch import nn


class PatchGANDiscriminator(nn.Module):
    """PatchGAN discriminator for optical image realism."""

    def __init__(self, in_channels: int = 4, base_channels: int = 32) -> None:
        """Initialize the discriminator.

        Args:
            in_channels: Number of optical input channels.
            base_channels: Width multiplier for the network.
        """

        super().__init__()
        if in_channels <= 0 or base_channels <= 0:
            raise ValueError("Channel counts must be positive.")

        def block(
            block_in_channels: int,
            block_out_channels: int,
            stride: int,
            normalize: bool = True,
        ) -> nn.Sequential:
            layers: list[nn.Module] = [
                nn.Conv2d(
                    block_in_channels,
                    block_out_channels,
                    kernel_size=4,
                    stride=stride,
                    padding=1,
                    bias=not normalize,
                )
            ]
            if normalize:
                layers.append(nn.BatchNorm2d(block_out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.layers = nn.Sequential(
            block(in_channels, base_channels, stride=2, normalize=False),
            block(base_channels, base_channels * 2, stride=2),
            block(base_channels * 2, base_channels * 4, stride=2),
            block(base_channels * 4, base_channels * 8, stride=1),
            nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return patch-level logits."""

        if tensor.ndim != 4:
            raise ValueError("Discriminator input must have shape (batch, channels, height, width).")
        return self.layers(tensor)


def build_discriminator(in_channels: int = 4, base_channels: int = 32) -> PatchGANDiscriminator:
    """Build the default PatchGAN discriminator."""

    return PatchGANDiscriminator(in_channels=in_channels, base_channels=base_channels)

