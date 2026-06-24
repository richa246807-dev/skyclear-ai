"""U-Net generator used for SAR-fused optical reconstruction."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Two-layer convolution block with batch normalization and ReLU activations."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize the block.

        Args:
            in_channels: Number of input feature channels.
            out_channels: Number of output feature channels.
        """

        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Run the convolution block."""

        return self.layers(tensor)


class DownBlock(nn.Module):
    """Downsampling block for the U-Net encoder."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize the block."""

        super().__init__()
        self.layers = nn.Sequential(nn.MaxPool2d(kernel_size=2), ConvBlock(in_channels, out_channels))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Run downsampling and convolution."""

        return self.layers(tensor)


class UpBlock(nn.Module):
    """Upsampling block for the U-Net decoder."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        """Initialize the block."""

        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, tensor: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Upsample ``tensor`` and concatenate the encoder skip connection."""

        tensor = self.up(tensor)
        if tensor.shape[-2:] != skip.shape[-2:]:
            tensor = nn.functional.interpolate(
                tensor,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return self.conv(torch.cat([skip, tensor], dim=1))


class UNetGenerator(nn.Module):
    """U-Net generator for four-band optical output from optical, SAR, and mask input."""

    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 4,
        base_channels: int = 32,
    ) -> None:
        """Initialize the generator.

        Args:
            in_channels: Expected input channels. The sprint default is 7.
            out_channels: Reconstructed optical bands. The sprint default is 4.
            base_channels: Width multiplier for the network.
        """

        super().__init__()
        if in_channels <= 0 or out_channels <= 0 or base_channels <= 0:
            raise ValueError("Channel counts must be positive.")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inc = ConvBlock(in_channels, base_channels)
        self.down1 = DownBlock(base_channels, base_channels * 2)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8)
        self.bridge = DownBlock(base_channels * 8, base_channels * 8)
        self.up1 = UpBlock(base_channels * 8, base_channels * 8, base_channels * 4)
        self.up2 = UpBlock(base_channels * 4, base_channels * 4, base_channels * 2)
        self.up3 = UpBlock(base_channels * 2, base_channels * 2, base_channels)
        self.up4 = UpBlock(base_channels, base_channels, base_channels)
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Run the generator and return optical bands in [0, 1]."""

        if tensor.ndim != 4:
            raise ValueError("Generator input must have shape (batch, channels, height, width).")
        if tensor.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, received {tensor.shape[1]}.")
        x1 = self.inc(tensor)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bridge(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return torch.sigmoid(self.outc(x))


def build_generator(
    in_channels: int = 7,
    out_channels: int = 4,
    base_channels: int = 32,
) -> UNetGenerator:
    """Build the default SkyClearAI generator."""

    return UNetGenerator(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
    )

