"""2D U-Net for medical image segmentation (PyTorch).

Architecture: Ronneberger et al., 2015 — adapted for binary segmentation
on greyscale slices. Designed to run on GTX 1650 (4 GB VRAM) with
FP32 and small batch sizes.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(in_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = _DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:])
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """2D U-Net with 4 encoder / 4 decoder levels.

    Parameters
    ----------
    in_channels: number of input channels (1 for greyscale MRI slices)
    out_channels: number of output classes (1 for binary tumour mask)
    base_features: feature map width at the first encoder level
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_features: int = 32,
    ) -> None:
        super().__init__()
        f = base_features
        self.enc1 = _DoubleConv(in_channels, f)
        self.enc2 = _Down(f, f * 2)
        self.enc3 = _Down(f * 2, f * 4)
        self.enc4 = _Down(f * 4, f * 8)
        self.bottleneck = _Down(f * 8, f * 16)
        self.dec4 = _Up(f * 16, f * 8)
        self.dec3 = _Up(f * 8, f * 4)
        self.dec2 = _Up(f * 4, f * 2)
        self.dec1 = _Up(f * 2, f)
        self.head = nn.Conv2d(f, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bottleneck(e4)
        d = self.dec4(b, e4)
        d = self.dec3(d, e3)
        d = self.dec2(d, e2)
        d = self.dec1(d, e1)
        return self.head(d)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
