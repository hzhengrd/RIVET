"""Spatially grounded evidence injector.

Drop-in replacement for EvidenceActionAdapter, motivated by measurements on the
trained r_sft_e_L0 run rather than by intuition.

WHAT THE DIAGNOSTICS SHOWED
---------------------------
1. The bank contributes nothing. evidence / learned / shuffled / 96-d histogram
   all score the same (0.6408 / 0.6425 / 0.6408; McNemar p=1.0, 97-99% of
   predictions byte-identical). So the pathway carries no usable signal today.
2. It is not the gate: gate is open (0.88 train, 0.72 infer), so this is NOT the
   old "gate closed the pathway" failure.
3. It is not temporal coverage: the 10 keyframes linearly interpolate a dense
   32-frame feature trajectory to within frame-to-frame noise (0.052 vs 0.059).
4. The residual error is dominated by target_object: 44 of 71 exact failures are
   target-only, and its accuracy on active clips is 0.661 vs 0.949 for tool.

TWO CONCRETE DEFECTS THIS FIXES
-------------------------------
a) **No spatial position anywhere.** PhaseTokenReducer flattens the (T,H,W) grid
   with `grid.reshape(batch, time*height*width, -1)` and adds a Gaussian prior
   over *frame time only*. Nothing tells the adapter which cell a feature came
   from, so it cannot represent "the screw went into the upper-left hole" -- the
   exact distinction target_object needs. Here the grid gets separable learned
   time/row/column embeddings before any attention.

b) **A fixed-query bottleneck placed before the question is known.** 160 grid
   positions are squeezed into 6 tokens by 6 *learned constant* queries, and only
   then does the LM cross-attend to those 6. The compression therefore decides
   what to keep without knowing whether the model is currently emitting `verb` or
   `target_object`. Here the LM hidden states attend **directly over the full
   positional grid**, so the query -- which is conditioned on what is being
   generated -- selects the relevant cells itself. 160 keys instead of 6 is
   negligible compute.

Everything else is deliberately unchanged (utility gate, slot-aux head, residual
option, bank_mode semantics, injection site) so a v1-vs-v2 comparison isolates
the reducer/positional change.
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
        # summary tokens are kept only to drive the gate and the slot-aux head,
        # NOT as the bottleneck the LM has to read through
        self.summary_queries = nn.Parameter(torch.randn(count, dim) * 0.02)
        self.utility = nn.Sequential(
            nn.LayerNorm(d_model + dim), nn.Linear(d_model + dim, dim // 2), nn.GELU(),
            nn.Linear(dim // 2, 1),
        )
        # NOT zero-initialised. v1 starts alpha at 0 so the injector begins as a
        # no-op and has to learn its way open -- but the main path is already
        # saturated (task_loss ~0.005 on the trained r_sft_e_L0), so there is no
        # residual error to push it. Measured after 12384 steps: alpha = -0.00077,
        # i.e. an injection scale of tanh(alpha)*gate ~= -0.0006. The pathway was
        # effectively closed regardless of what the bank contained. Starting at a
        # small positive value makes the adapter contribute from step 0 and lets
        # the utility gate learn *when* to use it instead.
        self.alpha = nn.Parameter(torch.full((), float(cfg.get("alpha_init", 0.1))))
        self.learned_bank = nn.Parameter(torch.randn(1, max(2, count // 2), 4, 4, feature_dim) * 0.02)
        self._grid: torch.Tensor | None = None
        self.last: dict[str, torch.Tensor] = {}
        self.slot_vocab = slot_vocab or {}
        self.slot_heads = nn.ModuleDict()
        if cfg.get("slot_aux", True) and slot_vocab:
            for slot in SLOTS:
                self.slot_heads[slot] = nn.Linear(dim, max(2, len(slot_vocab.get(slot, []))))

    # same external contract as EvidenceActionAdapter
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
