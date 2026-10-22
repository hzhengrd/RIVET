"""Mixed-reliability evidence corruption for the reliability gate.

Matches the paper's five modes (Section 3.2.3): an intact packet (correct), a
shuffled keyframe order (shuffled), a partially removed packet (partial),
substitution by another clip's packet (wrong), and complete removal (absent).
Each mode carries a binary arbitration target: only an intact packet still
depicts the queried clip with usable structure.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

CorruptionMode = Literal["correct", "shuffled", "partial", "wrong", "absent"]

DEFAULT_SCHEDULE: dict[str, float] = {
    "correct": 0.50,
    "shuffled": 0.11,
    "partial": 0.17,
    "wrong": 0.11,
    "absent": 0.11,
}

RELIABILITY_MODES: list[str] = ["correct", "shuffled", "partial", "wrong", "absent"]

# Binary arbitration target of Eq. (arbitration): a(m) = 1 only for an intact packet.
GATE_TARGET: dict[str, float] = {
    "correct": 1.0,
    "shuffled": 0.0,
    "partial": 0.0,
    "wrong": 0.0,
    "absent": 0.0,
}


@dataclass
class CorruptionDraw:
    mode: CorruptionMode
    evidence_clip_id: str | None
    shuffle_order: list[int] | None
    notes: dict[str, Any]


def sample_mode(rng: np.random.Generator, schedule: dict[str, float] | None = None) -> CorruptionMode:
    sched = schedule or DEFAULT_SCHEDULE
    keys = list(sched.keys())
    probs = np.asarray([sched[k] for k in keys], dtype=np.float64)
    probs = probs / probs.sum()
    return str(rng.choice(keys, p=probs))  # type: ignore[return-value]


def draw_corruption(row: dict[str, Any], rows: list[dict[str, Any]], index: int,
                    rng: np.random.Generator, mode: CorruptionMode | None = None,
                    schedule: dict[str, float] | None = None,
                    n_keyframes: int = 10) -> CorruptionDraw:
    mode = mode or sample_mode(rng, schedule)
    clip_id = row["clip_id"]
    if mode == "absent":
        return CorruptionDraw("absent", None, None, {"usable": False})
    if mode == "wrong":
        alt = next((rows[(index + offset) % len(rows)]
                    for offset in range(1, len(rows))
                    if rows[(index + offset) % len(rows)]["label"] != row["label"]),
                   rows[(index + 1) % len(rows)])
        return CorruptionDraw("wrong", alt["clip_id"], None,
                              {"usable": False, "donor": alt["clip_id"]})
    if mode == "shuffled":
        order = list(map(int, rng.permutation(n_keyframes)))
        return CorruptionDraw("shuffled", clip_id, order, {"usable": False})
    if mode == "partial":
        remove_idx = sorted(map(int, rng.choice(n_keyframes, size=max(1, n_keyframes // 2), replace=False)))
        return CorruptionDraw("partial", clip_id, None, {"usable": False, "remove_idx": remove_idx})
    return CorruptionDraw("correct", clip_id, None, {"usable": True})


def arbitration_label(draw: CorruptionDraw) -> str:
    return "yes" if draw.notes.get("usable") else "no"


def gate_target(mode: str | None) -> float:
    """Reliability gate binary target: 1 open, 0 closed."""
    return GATE_TARGET.get(mode, 0.0)


def apply_path_corruption(paths: list[str], draw: CorruptionDraw,
                          evidence_root: Path, donor_paths: list[str] | None = None) -> list[str]:
    if draw.mode == "absent":
        return []
    if draw.mode == "shuffled" and draw.shuffle_order is not None and paths:
        order = draw.shuffle_order
        return [paths[i % len(paths)] for i in order[:len(paths)]]
    if draw.mode == "partial" and paths:
        remove = set(draw.notes.get("remove_idx", []))
        return [p for i, p in enumerate(paths) if i not in remove]
    if draw.mode == "wrong" and donor_paths:
        return list(donor_paths[:len(paths)])
    return paths


def harm_levels() -> list[CorruptionMode]:
    return ["correct", "shuffled", "partial", "wrong", "absent"]
