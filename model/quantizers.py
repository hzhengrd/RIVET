from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class ContinuousQuantizer(nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return x, {"quantizer_loss": x.new_zeros(())}


class GroupedFSQ(nn.Module):
    def __init__(self, dim: int, levels: list[int]):
        super().__init__()
        self.levels = levels
        self.encoder = nn.Linear(dim, len(levels))
        self.decoder = nn.Linear(len(levels), dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        z = torch.tanh(self.encoder(x))
        quantized = []
        for index, level in enumerate(self.levels):
            scale = (level - 1) / 2
            q = torch.round(z[..., index] * scale) / max(scale, 1)
            quantized.append(q)
        hard = torch.stack(quantized, dim=-1)
        straight = z + (hard - z).detach()
        recon = self.decoder(straight)
        return recon, {"quantizer_loss": F.mse_loss(recon, x.detach()), "codes": hard.detach()}


class VectorQuantizer(nn.Module):
    def __init__(self, dim: int, codebook_size: int, commitment: float = 0.25, latent_dim: int | None = None):
        super().__init__()
        self.commitment = commitment
        latent_dim = latent_dim or min(64, dim)
        self.encoder = nn.Linear(dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, dim)
        self.codebook = nn.Embedding(codebook_size, latent_dim)
        nn.init.uniform_(self.codebook.weight, -1 / codebook_size, 1 / codebook_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        z = self.encoder(x)
        flat = z.reshape(-1, z.shape[-1])
        weight = self.codebook.weight
        distance = flat.square().sum(1, keepdim=True) + weight.square().sum(1) - 2 * flat @ weight.t()
        indices = distance.argmin(1)
        hard = self.codebook(indices).reshape_as(z)
        codebook_loss = F.mse_loss(hard, z.detach()) + self.commitment * F.mse_loss(hard.detach(), z)
        straight = z + (hard - z).detach()
        recon = self.decoder(straight)
        loss = codebook_loss + F.mse_loss(recon, x.detach())
        return recon, {"quantizer_loss": loss, "codes": indices.reshape(x.shape[:-1]).detach()}


class LearnedLatent(nn.Module):
    def __init__(self, token_count: int, dim: int):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(token_count, dim) * 0.02)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.tokens.unsqueeze(0).expand(x.shape[0], -1, -1), {"quantizer_loss": x.new_zeros(())}


class GaussianControl(nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return torch.randn_like(x), {"quantizer_loss": x.new_zeros(())}


def build_quantizer(kind: str, dim: int, token_count: int, fsq_levels: list[int], vq_size: int, commitment: float,
                    vq_latent_dim: int = 64) -> nn.Module:
    if kind == "continuous":
        return ContinuousQuantizer()
    if kind == "fsq":
        return GroupedFSQ(dim, fsq_levels)
    if kind == "vq":
        return VectorQuantizer(dim, vq_size, commitment, vq_latent_dim)
    if kind == "learned":
        return LearnedLatent(token_count, dim)
    if kind == "gaussian":
        return GaussianControl()
    raise ValueError(kind)
