"""V -- the convolutional VAE.

A faithful reimplementation of the V model in Ha & Schmidhuber, *World Models*
(2018), Section 2.1: four strided convolutions down to a 1024-d bottleneck, a
diagonal-Gaussian latent of 32 dimensions, and a mirrored deconvolutional
decoder.  At 64x64x3 with the paper's channel widths this is ~4.35 M
parameters, which matches the count reported in their Appendix table.

**This module is frozen across the comparison.**  Both the baseline and Supreme
consume latents produced by the *same* trained V checkpoint (see
``scripts/encode_latents.py``).  Retraining V per model would let VAE seed noise
leak into the memory comparison, and the effect size we are chasing is smaller
than that noise.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VAEOutput:
    recon: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor
    z: torch.Tensor


class ConvVAE(nn.Module):
    """64x64x3 <-> latent_dim Gaussian VAE.

    Parameters
    ----------
    latent_dim:
        Size of z.  32 in the paper; also the input width of M.
    channels:
        Encoder channel progression.  Decoder mirrors it.
    """

    def __init__(self, latent_dim: int = 32, channels: tuple[int, ...] = (32, 64, 128, 256),
                 image_size: int = 64, in_channels: int = 3):
        super().__init__()
        if image_size != 64:
            raise ValueError(
                "the encoder/decoder kernel schedule below is derived for 64x64; "
                "change image_size only together with the kernel sizes."
            )
        self.latent_dim = latent_dim
        self.image_size = image_size

        c1, c2, c3, c4 = channels
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=4, stride=2), nn.ReLU(inplace=True),  # 31
            nn.Conv2d(c1, c2, kernel_size=4, stride=2), nn.ReLU(inplace=True),           # 14
            nn.Conv2d(c2, c3, kernel_size=4, stride=2), nn.ReLU(inplace=True),           # 6
            nn.Conv2d(c3, c4, kernel_size=4, stride=2), nn.ReLU(inplace=True),           # 2
        )
        self.flat_dim = c4 * 2 * 2  # 1024 with the paper's widths
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, self.flat_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.flat_dim, c3, kernel_size=5, stride=2), nn.ReLU(inplace=True),  # 5
            nn.ConvTranspose2d(c3, c2, kernel_size=5, stride=2), nn.ReLU(inplace=True),             # 13
            nn.ConvTranspose2d(c2, c1, kernel_size=6, stride=2), nn.ReLU(inplace=True),             # 30
            nn.ConvTranspose2d(c1, in_channels, kernel_size=6, stride=2),                            # 64
        )

    # ------------------------------------------------------------ pieces --
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, 3, 64, 64) in [0,1] -> (mu, logvar)."""
        h = self.encoder(x).flatten(1)
        # Clamp keeps sigma in a numerically sane band; without it the low-entropy
        # feedback frames of the memory envs drive logvar to -inf.
        return self.fc_mu(h), self.fc_logvar(h).clamp(-8.0, 8.0)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z).view(-1, self.flat_dim, 1, 1)
        return torch.sigmoid(self.decoder(h))

    def forward(self, x: torch.Tensor) -> VAEOutput:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return VAEOutput(recon=self.decode(z), mu=mu, logvar=logvar, z=z)


def vae_loss(
    out: VAEOutput,
    target: torch.Tensor,
    *,
    beta: float = 1.0,
    free_bits: float = 0.5,
    recon: str = "l2_sum",
) -> dict[str, torch.Tensor]:
    """VAE objective, summed over pixels and averaged over the batch.

    ``free_bits`` puts a per-dimension floor on the KL term.  Ha & Schmidhuber
    clamp KL for their VizDoom run for the same reason we need it here: the
    synthetic memory frames are extremely low-entropy (a flat green or red
    screen), so an unconstrained KL collapses most latent dimensions to the
    prior and the cue becomes unrecoverable from z -- which would cripple *both*
    memory models and mask the effect we are measuring.
    """
    batch = target.shape[0]
    if recon == "l2_sum":
        recon_loss = F.mse_loss(out.recon, target, reduction="sum") / batch
    elif recon == "bce_sum":
        recon_loss = F.binary_cross_entropy(out.recon, target, reduction="sum") / batch
    else:
        raise ValueError(f"unknown recon loss {recon!r}")

    kl_per_dim = -0.5 * (1 + out.logvar - out.mu.pow(2) - out.logvar.exp())
    kl_free = torch.clamp(kl_per_dim, min=free_bits)
    kl = kl_free.sum(dim=1).mean()

    return {
        "loss": recon_loss + beta * kl,
        "recon": recon_loss.detach(),
        "kl": kl.detach(),
        "kl_raw": kl_per_dim.sum(dim=1).mean().detach(),
        "active_dims": (kl_per_dim.mean(dim=0) > 0.01).sum().detach().float(),
    }
