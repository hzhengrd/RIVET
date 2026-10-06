"""Mixed-reliability evidence corruption for parent checkpoints and gate targets."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

CorruptionMode = Literal[
    "correct", "shuffled", "partial", "wrong", "absent", "mark_jitter", "badge_perm"
]

DEFAULT_SCHEDULE: dict[str, float] = {
    "correct": 0.45,
    "shuffled": 0.10,
    "partial": 0.15,
    "mark_jitter": 0.05,
    "badge_perm": 0.05,
    "wrong": 0.10,
    "absent": 0.10,
}

RELIABILITY_MODES: list[str] = [
    "correct", "shuffled", "partial", "mark_jitter", "badge_perm", "wrong", "absent",
]

ARBITRATION_LABELS: dict[str, str | None] = {
    "correct": "yes",
    "mark_jitter": "yes",
    "shuffled": "no",
    "partial": "no",
    "badge_perm": "no",
    "wrong": "no",
    "absent": None,
}

# Gate should open on modes that still depict the same clip with usable structure.
GATE_OPEN_MODES = {"correct", "mark_jitter", "partial", "shuffled", "badge_perm"}


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
        return CorruptionDraw("absent", None, None, {"matches_video": False})
    if mode == "wrong":
        alt = next((rows[(index + offset) % len(rows)]
                    for offset in range(1, len(rows))
                    if rows[(index + offset) % len(rows)]["label"] != row["label"]),
                   rows[(index + 1) % len(rows)])
        return CorruptionDraw("wrong", alt["clip_id"], None,
                              {"matches_video": False, "donor": alt["clip_id"]})
    if mode in {"shuffled", "badge_perm"}:
        order = list(map(int, rng.permutation(n_keyframes)))
        return CorruptionDraw(mode, clip_id, order, {"matches_video": False})
    if mode == "partial":
        donor = rows[(index + 1 + int(rng.integers(0, max(1, len(rows) - 1)))) % len(rows)]
        swap_idx = sorted(map(int, rng.choice(n_keyframes, size=max(1, n_keyframes // 2), replace=False)))
        return CorruptionDraw("partial", clip_id, None,
                              {"matches_video": False, "donor": donor["clip_id"], "swap_idx": swap_idx})
    if mode == "mark_jitter":
        return CorruptionDraw("mark_jitter", clip_id, None,
                              {"matches_video": True, "jitter_px": int(rng.integers(8, 40))})
    return CorruptionDraw("correct", clip_id, None, {"matches_video": True})


def arbitration_label(draw: CorruptionDraw) -> str:
    return "yes" if draw.notes.get("matches_video") else "no"


def gate_target(mode: str | None) -> float:
    """Reliability gate soft target: 1 open, 0 closed."""
    if mode is None or mode in GATE_OPEN_MODES:
        return 1.0 if mode in {"correct", "mark_jitter", None} else 0.5
    return 0.0


def apply_path_corruption(paths: list[str], draw: CorruptionDraw,
                          evidence_root: Path, donor_paths: list[str] | None = None) -> list[str]:
    if draw.mode == "absent":
        return []
    if draw.mode in {"shuffled", "badge_perm"} and draw.shuffle_order is not None and paths:
        kf = [p for p in paths if "som" not in Path(p).name]
        som = [p for p in paths if "som" in Path(p).name]
        order = draw.shuffle_order
        if len(order) >= len(kf):
            kf = [kf[i] for i in order[:len(kf)]]
        else:
            kf = [kf[i % len(kf)] for i in order] + kf[len(order):]
        return kf + som
    if draw.mode == "partial" and donor_paths and paths:
        kf = [p for p in paths if "som" not in Path(p).name]
        som = [p for p in paths if "som" in Path(p).name]
        donor_kf = [p for p in donor_paths if "som" not in Path(p).name]
        swap = set(draw.notes.get("swap_idx", []))
        out = []
        for i, p in enumerate(kf):
            if i in swap and i < len(donor_kf):
                out.append(donor_kf[i])
            else:
                out.append(p)
        return out + som
    return paths


def harm_levels() -> list[CorruptionMode]:
    return ["correct", "shuffled", "partial", "wrong", "absent"]
