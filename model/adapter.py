"""Adapter infrastructure: hook, config, and the adapter-class anchor.

The adapter itself lives in model/spatial.py (SpatialEvidenceAdapter) and
model/hand_anchored.py (HandAnchoredAdapter). Runners rebind `AdapterClass`
before calling train/infer, so the stock entry points construct whichever
adapter the launcher selects.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn


AdapterClass: type[nn.Module] | None = None


def locate_decoder_layers(model: nn.Module) -> nn.ModuleList:
    candidates = [(name, module) for name, module in model.named_modules()
                  if isinstance(module, nn.ModuleList) and len(module) >= 8]
    if not candidates:
        raise RuntimeError("Could not locate VLM decoder ModuleList")
    name, layers = max(candidates, key=lambda item: len(item[1]))
    return layers


class AdapterHook:
    def __init__(self, model: nn.Module, adapter: nn.Module, layer_index: int):
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
