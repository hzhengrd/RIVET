"""Run training / inference with the hand-anchored adapter (model/hand_anchored.py).

Two rebinds inside the shipped trainer's / inferencer's module namespace, no edit
to either file:

  EvidenceActionAdapter -> HandAnchoredAdapter   (the adapter itself)
  make_messages         -> wrapper               (delivers the per-clip hand box)

The second one is needed because the adapter's only per-sample input channel is
`set_grid`, which carries features and nothing else. `make_messages(cfg, row, ...)`
is the one call made exactly once per item immediately before each forward pass
(model.train and model.infer) and it has the row, so
wrapping it is the natural place to push the hand and its boxes onto the adapter.

If a clip has no hand_boxes file the wrapper pushes `None`, and the adapter
falls back to plain v2 behaviour for that sample rather than failing.

    python -m model.run_hand_anchored train --config ... --variant ... --seed 17
    python -m model.run_hand_anchored infer --config ... --variant ... --split test_lh
"""
from __future__ import annotations

import json
import sys
from typing import Any

from .hand_anchored import HandAnchoredAdapter

_LIVE: dict[str, Any] = {"adapter": None, "cache": {}, "hits": 0, "misses": 0}


class _RegisteredHandAdapter(HandAnchoredAdapter):
    """Identical to HandAnchoredAdapter; records the instance so the patched
    make_messages can reach it (exactly one adapter exists per run)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _LIVE["adapter"] = self


def _boxes_for(cfg: dict[str, Any], clip_id: str) -> list | None:
    cache = _LIVE["cache"]
    if clip_id in cache:
        return cache[clip_id]
    from .config import artifact_path
    dirname = cfg.get("features", {}).get("hand_boxes_dirname", "hand_boxes")
    path = artifact_path(cfg, dirname, f"{clip_id}.json")
    boxes = None
    if path.exists():
        try:
            boxes = json.loads(path.read_text()).get("boxes")
        except (ValueError, OSError):
            boxes = None
    cache[clip_id] = boxes
    if boxes is None:
        _LIVE["misses"] += 1
    else:
        _LIVE["hits"] += 1
    return boxes


def _patch(module) -> None:
    module.EvidenceActionAdapter = _RegisteredHandAdapter
    original = module.make_messages

    def wrapped(cfg, row, spec, *args, **kwargs):
        adapter = _LIVE["adapter"]
        if adapter is not None and hasattr(adapter, "set_hand"):
            hand = row.get("target_hand")
            adapter.set_hand(hand, _boxes_for(cfg, row["clip_id"]))
        return original(cfg, row, spec, *args, **kwargs)

    module.make_messages = wrapped


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "infer"):
        raise SystemExit("usage: python -m model.run_hand_anchored {train|infer} [args...]")
    mode = sys.argv.pop(1)
    if mode == "train":
        from . import train as target
    else:
        from . import infer as target
    _patch(target)
    print(f"[run_hand_anchored] {mode}: adapter = HandAnchoredAdapter")
    try:
        target.main()
    finally:
        print(f"[run_hand_anchored] hand_boxes: {_LIVE['hits']} clips found, "
              f"{_LIVE['misses']} missing")


if __name__ == "__main__":
    main()
