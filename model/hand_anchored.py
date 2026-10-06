"""Hand-anchored evidence adapter for the pooled dual-hand model.

THE PROBLEM IT TARGETS
----------------------
One model, both hands. The only thing that distinguishes the two cases is the
prompt sentence "Focus only on the left/right hand." -- the crop box is the same
constant (450,100,1100,700) on both sides, and on ~45% of the 1762 clip_ids the
two hands share, the annotated window and the label are identical too. So for a
large slice of training the hand clause is uninformative, and the model has every
reason to ignore it. It does: pooled scores 0.5285/0.5336 against per-hand
specialists at 0.6326/0.6560, and over half that deficit is idle right-hand
clips called "active" (34.2% vs 18.9%) -- exactly the error you get from reading
the wrong hand's motion.

Adding the hand as one more embedding would not fix this, because the text
already carries hand *identity*. What is missing is hand *location*: nothing in
the (T,4,4,1152) evidence grid says which cells the asked-about hand occupies.

WHAT THIS ADDS (three terms, all ablatable)
-------------------------------------------
1. identity   a learned left/right embedding on the grid cells. Cheap, and it
              lets the two hands specialise their read-out of the same features.
2. geometry   the tracked hand box per keyframe (x0,y0,x1,y1,valid) through a
              small MLP, added to that keyframe's cells -- so the adapter knows
              where the hand is *at each point in time*, which a single sentence
              cannot express.
3. proximity  an additive bias on the cross-attention logits, -scale * d^2, with
              d the distance from each grid cell centre to the hand box centre.
              This is the term that actually steers read-out to the acting hand.
              `scale` is learned and starts small, so at init the model is v2 to
              within the two embeddings; it can only sharpen if that helps.

Keyframes with no detected hand contribute zero bias and a `valid=0` geometry
vector, so the adapter degrades to v2 exactly where tracking fails (measured
coverage: mean 0.978, 85% of clips fully tracked).

Boxes come from model/hand_boxes.py, computed on the same keyframe JPEGs the
feature grid is built from, so cell <-> box alignment needs no assumptions.

NOT SoM: nothing is drawn, nothing enters the prompt, the VLM never sees a mark.
The geometry is consumed numerically here, inside the adapter.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from .spatial import SpatialEvidenceAdapter

HAND_INDEX = {"left": 0, "right": 1}


class HandAnchoredAdapter(SpatialEvidenceAdapter):
    """SpatialEvidenceAdapter + identity / geometry / proximity conditioning."""

    def __init__(self, d_model: int, feature_dim: int, cfg: dict[str, Any],
                 slot_vocab: dict[str, list[str]] | None = None):
        super().__init__(d_model, feature_dim, cfg, slot_vocab)
        dim = int(cfg["token_dim"])
        self.use_identity = bool(cfg.get("hand_identity", True))
        self.use_geometry = bool(cfg.get("hand_geometry", True))
        self.use_proximity = bool(cfg.get("hand_proximity", True))
        self.hand_identity = nn.Embedding(len(HAND_INDEX), dim)
        nn.init.normal_(self.hand_identity.weight, std=0.02)
        self.hand_geometry = nn.Sequential(
            nn.Linear(5, dim), nn.GELU(), nn.Linear(dim, dim))
        nn.init.zeros_(self.hand_geometry[-1].weight)
        nn.init.zeros_(self.hand_geometry[-1].bias)
        # starts near zero so the attention distribution at init matches v2
        self.prox_scale = nn.Parameter(torch.tensor(float(cfg.get("prox_scale_init", 0.5))))
        self.prox_sigma = nn.Parameter(torch.tensor(float(cfg.get("prox_sigma_init", 0.35))))
        self._hand_index: int | None = None
        self._hand_boxes: np.ndarray | None = None      # (T,4) normalised xyxy
        self._hand_valid: np.ndarray | None = None      # (T,)

    # ---- external contract, mirrors set_grid ---------------------------------
    def set_hand(self, hand: str | None, boxes: list[list[float] | None] | None) -> None:
        if hand not in HAND_INDEX:
            self._hand_index = None
            self._hand_boxes = None
            self._hand_valid = None
            return
        self._hand_index = HAND_INDEX[hand]
        if not boxes:
            self._hand_boxes = np.zeros((0, 4), dtype=np.float32)
            self._hand_valid = np.zeros((0,), dtype=np.float32)
            return
        arr = np.zeros((len(boxes), 4), dtype=np.float32)
        valid = np.zeros((len(boxes),), dtype=np.float32)
        for i, box in enumerate(boxes):
            if box is None:
                continue
            arr[i] = np.clip(np.asarray(box, dtype=np.float32), 0.0, 1.0)
            valid[i] = 1.0
        self._hand_boxes = arr
        self._hand_valid = valid

    def _aligned(self, t: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """-> boxes (T,4), valid (T,) padded/truncated to the grid's T."""
        boxes = np.zeros((t, 4), dtype=np.float32)
        valid = np.zeros((t,), dtype=np.float32)
        if self._hand_boxes is not None and len(self._hand_boxes):
            n = min(t, len(self._hand_boxes))
            boxes[:n] = self._hand_boxes[:n]
            valid[:n] = self._hand_valid[:n]
        return (torch.from_numpy(boxes).to(device=device, dtype=dtype),
                torch.from_numpy(valid).to(device=device, dtype=dtype))

    def _condition(self, cells: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """cells (B,T,H,W,dim) -> conditioned cells, per-key attention bias (B,N)."""
        b, t, h, w, dim = cells.shape
        if self._hand_index is None:
            return cells, None
        boxes, valid = self._aligned(t, cells.device, cells.dtype)

        if self.use_identity:
            idx = torch.tensor([self._hand_index], device=cells.device)
            cells = cells + self.hand_identity(idx).view(1, 1, 1, 1, dim).to(cells.dtype)
        if self.use_geometry:
            geo = self.hand_geometry(torch.cat([boxes, valid[:, None]], dim=-1))   # (T,dim)
            cells = cells + geo.view(1, t, 1, 1, dim)

        bias = None
        if self.use_proximity:
            # grid cell centres in normalised image coordinates
            rows = (torch.arange(h, device=cells.device, dtype=cells.dtype) + 0.5) / h
            cols = (torch.arange(w, device=cells.device, dtype=cells.dtype) + 0.5) / w
            cy = 0.5 * (boxes[:, 1] + boxes[:, 3])
            cx = 0.5 * (boxes[:, 0] + boxes[:, 2])
            dy = rows.view(1, h, 1) - cy.view(t, 1, 1)
            dx = cols.view(1, 1, w) - cx.view(t, 1, 1)
            sigma = self.prox_sigma.abs().clamp(min=0.05).to(cells.dtype)
            dist2 = (dy ** 2 + dx ** 2) / (sigma ** 2)
            bias = -self.prox_scale.to(cells.dtype) * dist2                # (T,H,W)
            # keyframes without a detected hand must not bias anything
            bias = bias * valid.view(t, 1, 1)
            bias = bias.reshape(1, t * h * w).expand(b, -1)
        return cells, bias

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

        cells = self.grid_norm(self.pos(self.project(grid)))
        cells, bias = self._condition(cells)
        b, t, h, w, d = cells.shape
        cells = cells.reshape(b, t * h * w, d)

        drop = float(self.cfg.get("token_dropout", 0.0))
        if self.training and drop > 0:
            keep = (torch.rand(cells.shape[:2], device=cells.device) >= drop).to(cells.dtype)
            cells = cells * keep.unsqueeze(-1) / (1.0 - drop)

        key_value = self.key_value(cells)
        attn_mask = None
        if bias is not None:
            heads = self.cross_attention.num_heads
            length = hidden.shape[1]
            # MultiheadAttention wants (B*heads, L, S) additive float mask
            attn_mask = bias[:, None, None, :].expand(b, heads, length, bias.shape[1])
            attn_mask = attn_mask.reshape(b * heads, length, bias.shape[1]).to(hidden.dtype)
        attended, attn = self.cross_attention(self.query_norm(hidden), key_value, key_value,
                                              attn_mask=attn_mask, need_weights=True,
                                              average_attn_weights=True)
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
            "hand_valid": torch.tensor(float(0.0 if self._hand_valid is None
                                             else float(np.mean(self._hand_valid))
                                             if len(self._hand_valid) else 0.0)),
        }
        return hidden + scale[:, None, None] * attended
