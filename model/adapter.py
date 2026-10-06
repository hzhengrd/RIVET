"""Evidence-aligned action adapter.

The evidence grid lives in Qwen vision space. Residualization is optional.
The reliability gate is trained on corruption labels. A slot auxiliary head
targets exact structured accuracy.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .quantizers import build_quantizer
from .utils import SLOTS


class PhaseTokenReducer(nn.Module):
    def __init__(self, input_dim: int, token_dim: int, token_count: int):
        super().__init__()
        self.token_count = token_count
        self.input = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, token_dim))
        self.queries = nn.Parameter(torch.randn(token_count, token_dim) * 0.02)
        self.log_width = nn.Parameter(torch.full((token_count,), -1.5))
        self.output = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, token_dim))

    def forward(self, grid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, time, height, width, _ = grid.shape
        features = self.input(grid.reshape(batch, time * height * width, -1))
        logits = torch.einsum("md,bnd->bmn", self.queries, features) / features.shape[-1] ** 0.5
        token_time = torch.linspace(0, 1, self.token_count, device=grid.device, dtype=features.dtype)
        frame_time = torch.linspace(0, 1, time, device=grid.device, dtype=features.dtype).repeat_interleave(height * width)
        width_scale = self.log_width.exp().clamp_min(0.03)
        prior = -0.5 * ((frame_time[None, :] - token_time[:, None]) / width_scale[:, None]).square()
        weights = torch.softmax(logits + prior.unsqueeze(0), dim=-1)
        tokens = self.output(torch.einsum("bmn,bnd->bmd", weights, features))
        return tokens, weights


class EvidenceActionAdapter(nn.Module):
    """Mid-layer evidence injector with optional residualization and slot aux."""

    def __init__(self, d_model: int, feature_dim: int, cfg: dict[str, Any],
                 slot_vocab: dict[str, list[str]] | None = None):
        super().__init__()
        self.cfg = cfg
        self.enabled = True
        self.gate_override: float | None = None
        self.bank_mode = cfg.get("bank_mode", "evidence")  # evidence | learned | shuffled
        count, dim = int(cfg["token_count"]), int(cfg["token_dim"])
        self.reducer = PhaseTokenReducer(feature_dim, dim, count)
        self.predictor = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, dim * count))
        self.quantizer = build_quantizer(
            cfg.get("quantizer", "continuous"), dim, count,
            cfg.get("fsq_levels", [8, 8, 8, 6, 5]),
            int(cfg.get("vq_codebook_size", 256)),
            float(cfg.get("vq_commitment", 0.25)),
            int(cfg.get("vq_latent_dim", 64)),
        )
        self.key_value = nn.Linear(dim, d_model)
        self.query_norm = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(d_model, int(cfg.get("phase_heads", 8)), batch_first=True)
        self.utility = nn.Sequential(
            nn.LayerNorm(d_model + dim), nn.Linear(d_model + dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1),
        )
        self.alpha = nn.Parameter(torch.zeros(()))
        self.learned_bank = nn.Parameter(torch.randn(1, max(2, count // 2), 4, 4, feature_dim) * 0.02)
        self._grid: torch.Tensor | None = None
        self.last: dict[str, torch.Tensor] = {}
        self.slot_vocab = slot_vocab or {}
        self.slot_heads = nn.ModuleDict()
        if cfg.get("slot_aux", True) and slot_vocab:
            for slot in SLOTS:
                n = max(2, len(slot_vocab.get(slot, [])))
                self.slot_heads[slot] = nn.Linear(dim, n)

    def set_grid(self, grid: torch.Tensor | None) -> None:
        self._grid = grid

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            self.last = {}
            return hidden
        if self.bank_mode == "learned" or self._grid is None:
            grid = self.learned_bank.expand(hidden.shape[0], -1, -1, -1, -1).to(hidden.dtype)
        else:
            grid = self._grid.to(hidden.device, dtype=hidden.dtype)
            if self.bank_mode == "shuffled" and grid.shape[1] > 1:
                perm = torch.randperm(grid.shape[1], device=grid.device)
                grid = grid[:, perm]
        action, phase_weights = self.reducer(grid)
        summary = hidden.mean(1)
        predicted = self.predictor(summary).reshape(action.shape)
        residualize = bool(self.cfg.get("residualize", False))
        residual = action - predicted if residualize else action
        tokens, qdiag = self.quantizer(residual)
        drop_probability = float(self.cfg.get("token_dropout", 0.0))
        if self.training and drop_probability > 0:
            keep = (torch.rand(tokens.shape[:2], device=tokens.device) >= drop_probability).to(tokens.dtype)
            tokens = tokens * keep.unsqueeze(-1) / (1.0 - drop_probability)
        key_value = self.key_value(tokens)
        attended, _ = self.cross_attention(self.query_norm(hidden), key_value, key_value, need_weights=False)
        gate_logit = self.utility(torch.cat([summary, tokens.mean(1)], dim=-1)).squeeze(-1)
        if self.cfg.get("utility_gate", True):
            gate = torch.sigmoid(gate_logit)
        else:
            gate = torch.ones_like(gate_logit)
        if self.gate_override is not None:
            gate = torch.full_like(gate, self.gate_override)
        scale = torch.tanh(self.alpha) * gate
        self.last = {
            "gate": gate, "gate_logit": gate_logit, "action": action, "predicted": predicted,
            "residual": residual, "tokens": tokens, "phase_weights": phase_weights, **qdiag,
        }
        return hidden + scale[:, None, None] * attended

    def slot_aux_loss(self, gold_slots: dict[str, str]) -> torch.Tensor:
        if not self.slot_heads or "tokens" not in self.last:
            return torch.zeros((), device=next(self.parameters()).device)
        pooled = self.last["tokens"].mean(1)
        batch = pooled.shape[0]
        total = pooled.new_zeros(())
        count = 0
        for slot, head in self.slot_heads.items():
            vocab = self.slot_vocab.get(slot, [])
            if not vocab:
                continue
            value = gold_slots.get(slot, "")
            if value not in vocab:
                continue
            target = torch.full((batch,), vocab.index(value), device=pooled.device, dtype=torch.long)
            total = total + F.cross_entropy(head(pooled), target)
            count += 1
        return total / max(1, count)


def locate_decoder_layers(model: nn.Module) -> nn.ModuleList:
    candidates = [(name, module) for name, module in model.named_modules()
                  if isinstance(module, nn.ModuleList) and len(module) >= 8]
    if not candidates:
        raise RuntimeError("Could not locate VLM decoder ModuleList")
    name, layers = max(candidates, key=lambda item: len(item[1]))
    return layers


class AdapterHook:
    def __init__(self, model: nn.Module, adapter: EvidenceActionAdapter, layer_index: int):
        self.adapter = adapter
        layers = locate_decoder_layers(model)
        if not (-len(layers) <= layer_index < len(layers)):
            raise IndexError(f"layer {layer_index}, available={len(layers)}")
        self.layer_count = len(layers)
        self.handle = layers[layer_index].register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if isinstance(output, tuple):
            return (self.adapter(output[0]), *output[1:])
        return self.adapter(output)

    def remove(self) -> None:
        self.handle.remove()


def adapter_config(cfg: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    value = dict(cfg.get("rapt", {}))
    value.update({key: val for key, val in overrides.items() if val is not None})
    return value


def hidden_size(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    if config is None:
        return 4096
    if getattr(config, "hidden_size", None):
        return int(config.hidden_size)
    text = getattr(config, "text_config", None)
    if text is not None and getattr(text, "hidden_size", None):
        return int(text.hidden_size)
    return 4096
