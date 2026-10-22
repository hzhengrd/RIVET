"""Spatially grounded evidence injector.

Hidden-conditioned cross-attention over a positional evidence grid, with a
utility gate, a slot auxiliary head and a learnable injection scale.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .utils import SLOTS


class GridPositional(nn.Module):
    """Separable learned time/row/column embeddings for a (B,T,H,W,D) grid.

    Separable rather than a single flat table so it generalises to clips with
    fewer keyframes than the maximum (short clips yield T<10)."""

    def __init__(self, dim: int, max_t: int = 32, max_h: int = 16, max_w: int = 16):
        super().__init__()
        self.time = nn.Parameter(torch.randn(max_t, dim) * 0.02)
        self.row = nn.Parameter(torch.randn(max_h, dim) * 0.02)
        self.col = nn.Parameter(torch.randn(max_w, dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, h, w, d = x.shape
        # clamp instead of erroring: a clip may carry more keyframes than max_t
        ti = torch.arange(t, device=x.device).clamp_max(self.time.shape[0] - 1)
        ri = torch.arange(h, device=x.device).clamp_max(self.row.shape[0] - 1)
        ci = torch.arange(w, device=x.device).clamp_max(self.col.shape[0] - 1)
        pos = (self.time[ti][:, None, None, :]
               + self.row[ri][None, :, None, :]
               + self.col[ci][None, None, :, :])
        return x + pos.unsqueeze(0).to(x.dtype)


class SpatialEvidenceAdapter(nn.Module):
    """Hidden-conditioned cross-attention over a positional evidence grid."""

    def __init__(self, d_model: int, feature_dim: int, cfg: dict[str, Any],
                 slot_vocab: dict[str, list[str]] | None = None):
        super().__init__()
        self.cfg = cfg
        self.enabled = True
        self.gate_override: float | None = None
        self.bank_mode = cfg.get("bank_mode", "evidence")
        dim = int(cfg["token_dim"])
        count = int(cfg["token_count"])

        self.project = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, dim))
        self.pos = GridPositional(dim)
        self.grid_norm = nn.LayerNorm(dim)
        self.key_value = nn.Linear(dim, d_model)
        self.query_norm = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(d_model, int(cfg.get("phase_heads", 8)),
                                                     batch_first=True)
        # summary tokens drive the gate and the slot-aux head only
        self.summary_queries = nn.Parameter(torch.randn(count, dim) * 0.02)
        self.utility = nn.Sequential(
            nn.LayerNorm(d_model + dim), nn.Linear(d_model + dim, dim // 2), nn.GELU(),
            nn.Linear(dim // 2, 1),
        )
        # positive init so the adapter contributes from step 0
        self.alpha = nn.Parameter(torch.full((), float(cfg.get("alpha_init", 0.1))))
        self.learned_bank = nn.Parameter(torch.randn(1, max(2, count // 2), 4, 4, feature_dim) * 0.02)
        self._grid: torch.Tensor | None = None
        self.last: dict[str, torch.Tensor] = {}
        self.slot_vocab = slot_vocab or {}
        self.slot_heads = nn.ModuleDict()
        if cfg.get("slot_aux", True) and slot_vocab:
            for slot in SLOTS:
                self.slot_heads[slot] = nn.Linear(dim, max(2, len(slot_vocab.get(slot, []))))

    def set_grid(self, grid: torch.Tensor | None) -> None:
        self._grid = grid

    def _summary_tokens(self, cells: torch.Tensor) -> torch.Tensor:
        """(B,N,dim) -> (B,count,dim) by plain query pooling; gate/slot-aux only."""
        logits = torch.einsum("md,bnd->bmn", self.summary_queries, cells) / cells.shape[-1] ** 0.5
        return torch.einsum("bmn,bnd->bmd", torch.softmax(logits, dim=-1), cells)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            self.last = {}
            return hidden
        if self.bank_mode == "learned" or self._grid is None:
            grid = self.learned_bank.expand(hidden.shape[0], -1, -1, -1, -1).to(hidden.dtype)
        else:
            grid = self._grid.to(hidden.device, dtype=hidden.dtype)
            if self.bank_mode == "shuffled" and grid.shape[1] > 1:
                grid = grid[:, torch.randperm(grid.shape[1], device=grid.device)]
        if grid.shape[0] != hidden.shape[0]:
            grid = grid.expand(hidden.shape[0], *grid.shape[1:])

        cells = self.grid_norm(self.pos(self.project(grid)))          # (B,T,H,W,dim)
        b, t, h, w, d = cells.shape
        cells = cells.reshape(b, t * h * w, d)                         # (B,N,dim)

        drop = float(self.cfg.get("token_dropout", 0.0))
        if self.training and drop > 0:
            keep = (torch.rand(cells.shape[:2], device=cells.device) >= drop).to(cells.dtype)
            cells = cells * keep.unsqueeze(-1) / (1.0 - drop)

        # the LM's own hidden states are the queries -> what gets read out depends
        # on which slot is currently being generated
        key_value = self.key_value(cells)
        attended, attn = self.cross_attention(self.query_norm(hidden), key_value, key_value,
                                              need_weights=True, average_attn_weights=True)
        summary_tokens = self._summary_tokens(cells)
        summary = hidden.mean(1)
        gate_logit = self.utility(torch.cat([summary, summary_tokens.mean(1)], dim=-1)).squeeze(-1)
        gate = torch.sigmoid(gate_logit) if self.cfg.get("utility_gate", True) \
            else torch.ones_like(gate_logit)
        if self.gate_override is not None:
            gate = torch.full_like(gate, self.gate_override)
        scale = torch.tanh(self.alpha) * gate
        self.last = {
            "gate": gate, "gate_logit": gate_logit, "tokens": summary_tokens,
            "action": summary_tokens, "predicted": summary_tokens,
            "residual": summary_tokens, "attn": attn.detach(),
            "n_cells": torch.tensor(float(cells.shape[1])),
        }
        return hidden + scale[:, None, None] * attended

    def slot_aux_loss(self, gold_slots: dict[str, str]) -> torch.Tensor:
        if not self.slot_heads or "tokens" not in self.last:
            return torch.zeros((), device=next(self.parameters()).device)
        pooled = self.last["tokens"].mean(1)
        total = pooled.new_zeros(())
        count = 0
        for slot, head in self.slot_heads.items():
            vocab = self.slot_vocab.get(slot, [])
            value = gold_slots.get(slot, "")
            if not vocab or value not in vocab:
                continue
            target = torch.full((pooled.shape[0],), vocab.index(value),
                                device=pooled.device, dtype=torch.long)
            total = total + F.cross_entropy(head(pooled), target)
            count += 1
        return total / max(1, count)
