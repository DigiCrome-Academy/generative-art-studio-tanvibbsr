"""Wasserstein GAN (WGAN / WGAN-GP) critic.

Learning objective (docs/PROJECT_BRIEF.md, Phase 2):
"Implement Wasserstein GAN for training stability."

A WGAN "critic" has the *same architecture* as a DCGAN discriminator minus
the final Sigmoid (it outputs an unbounded real-valued score, not a
probability). The Wasserstein loss and gradient-penalty term live in
`src/generative_art_studio/training/losses.py`.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 4,
    stride: int = 2,
    padding: int = 1,
    use_instance_norm: bool = True,
    first_block: bool = False,
) -> nn.Module:
    """Build a convolutional block for the WGANCritic.

    This mirrors the DCGAN discriminator stack while avoiding BatchNorm in
    the critic to keep the WGAN gradient penalty valid for each sample.
    """
    layers: list[nn.Module] = [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_instance_norm,
        )
    ]

    if use_instance_norm:
        layers.append(nn.InstanceNorm2d(out_channels, affine=True))
    elif not first_block:
        layers.append(nn.BatchNorm2d(out_channels))

    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


class WGANCritic(nn.Module):
    """Mirrors DCGANDiscriminator's architecture but has no final Sigmoid.

    TODO(Phase 2 - WGAN): build `self.net` with the same 5-block
    conv stack as `DCGANDiscriminator` in `dcgan.py`, but:
      * end with `Conv2d(feat*8, 1, 4, 1, 0)` and **no Sigmoid** — the
        critic's output is an unbounded score, not a probability.
      * if you plan to use gradient penalty (WGAN-GP) rather than weight
        clipping, use `nn.InstanceNorm2d` instead of `nn.BatchNorm2d`
        (BatchNorm couples samples within a batch, which conflicts with
        the per-sample gradient penalty).
    """

    def __init__(self, img_channels: int = 3, feature_maps: int = 64, use_instance_norm: bool = True):
        super().__init__()
        self.feature_maps = feature_maps
        self.net = nn.Sequential(
            conv_block(
                img_channels,
                feature_maps,
                use_instance_norm=use_instance_norm,
                first_block=True,
            ),
            conv_block(feature_maps, feature_maps * 2, use_instance_norm=use_instance_norm),
            conv_block(feature_maps * 2, feature_maps * 4, use_instance_norm=use_instance_norm),
            conv_block(feature_maps * 4, feature_maps * 8, use_instance_norm=use_instance_norm),
            nn.Conv2d(feature_maps * 8, 1, kernel_size=4, stride=1, padding=0),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        out = self.net(img)
        return out.view(-1, 1)


def clip_weights(critic: nn.Module, clip_value: float = 0.01) -> None:
    """Weight clipping for the original (non-gradient-penalty) WGAN.

    Fully implemented — call after each critic optimizer step when *not*
    using gradient penalty:
        `clip_weights(critic, config.WGAN_CLIP_VALUE)`
    """
    for p in critic.parameters():
        p.data.clamp_(-clip_value, clip_value)
