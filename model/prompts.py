"""Prompt and message construction (main, necessity, arbitration)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import artifact_path, evidence_root, repo_path
from .corruption import (
    DEFAULT_SCHEDULE,
    apply_path_corruption,
    arbitration_label,
    draw_corruption,
)
from .utils import structured_answer, subset_indices


def hand_focus(target: str) -> str:
    """Hand-focus clause, trailing space when non-empty. Mirrors
    data.common.hand_focus: datasets with no acting-hand annotation
    ("both") get no hand clause -- the question is about the action, not a hand."""
    return "" if target == "both" else f"Focus only on the {target} hand. "

MAIN_PROMPT = (
    "{hand_focus}The video shows the full clip. "
    "{evidence_sentence}"
    "Identify the assembly state and the ordered slots. "
    "Return only JSON with keys status, action_verb, manipulated_object, target_object, tool. "
    "Use status active or null. For null, use action_verb null, no tool, and not applicable objects."
)

BADGE_SENTENCE = (
    "The additional images are keyframes in temporal order; each carries a badge "
    "'k/N' (top-left) and a yellow progress bar (bottom) showing its time position in the clip. "
)
PLAIN_KF_SENTENCE = "The additional images are keyframes sampled across the clip in temporal order. "
SOM_SENTENCE = (
    "The last image marks candidate object regions with numbered colored circles; "
    "the numbers are reference options only and may include irrelevant regions. "
)
CALIBRATION_SENTENCE = (
    "Evidence may be incomplete or mismatched; if the images disagree with the video, "
    "trust the video and ignore the images. "
)

NECESSITY_PROMPTS = {
    "change": ("Look at the badge numbers on the keyframes. Between which keyframe and the next one "
               "does the largest visible change in the workspace occur? Answer with the badge number "
               "of the earlier keyframe only."),
    "latest": ("Look at the badge numbers and progress bars on the keyframes. Which badge number "
               "corresponds to the latest moment in the clip? Answer with the number only."),
    "mark": ("Look at the numbered circle marks in the last image. Which mark number is closest to the "
             "{hand} hand? Answer with the number only."),
}

ARBITRATION_PROMPT = (
    "Look at the video and the additional evidence images. "
    "Do the evidence images come from the same clip as the video "
    "(same workspace, same action sequence)? Answer yes or no only."
)


def load_meta(cfg: dict[str, Any], clip_id: str, spec: dict[str, Any] | None = None) -> dict[str, Any]:
    path = evidence_root(cfg, spec) / clip_id / "meta.json"
    if not path.exists():
        path = artifact_path(cfg, "evidence", clip_id, "meta.json")
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _kf_paths(root: Path, meta: dict[str, Any], use_badges: bool) -> list[str]:
    kf_assets = sorted((a for a in meta["assets"] if a["type"] == "kf"), key=lambda a: a["ordinal"])
    return [str(root / (a["path"] if use_badges else f"plain_{a['path']}")) for a in kf_assets]


def evidence_images(cfg: dict[str, Any], meta: dict[str, Any], spec: dict[str, Any],
                    rng: np.random.Generator | None = None,
                    corruption: dict[str, Any] | None = None) -> list[str]:
    root = evidence_root(cfg, spec) / meta["clip_id"]
    if not root.exists():
        root = artifact_path(cfg, "evidence", meta["clip_id"])
    use_badges = spec.get("use_badges", True)
    paths = _kf_paths(root, meta, use_badges)
    subset = spec.get("keyframe_subset")
    if subset and len(paths) > int(subset):
        paths = [paths[i] for i in subset_indices(len(paths), int(subset))]
    if corruption and corruption.get("mode") in {"shuffled", "badge_perm"} and corruption.get("shuffle_order"):
        order = corruption["shuffle_order"]
        paths = [paths[i % len(paths)] for i in order[:len(paths)]] if paths else paths
    if corruption and corruption.get("mode") == "partial" and corruption.get("donor"):
        donor_meta = load_meta(cfg, corruption["donor"], spec)
        donor_root = evidence_root(cfg, spec) / donor_meta["clip_id"]
        donor_paths = _kf_paths(donor_root, donor_meta, use_badges)
        from .corruption import CorruptionDraw
        draw = CorruptionDraw("partial", meta["clip_id"], None, corruption)
        paths = apply_path_corruption(paths, draw, evidence_root(cfg, spec), donor_paths)
    if spec.get("inference_evidence") == "shuffled" and rng is not None and paths:
        paths = [paths[i] for i in rng.permutation(len(paths))]
    if spec.get("use_som", True) and any(a["type"] == "som" for a in meta["assets"]):
        paths.append(str(root / "som.jpg"))
    if corruption and corruption.get("mode") == "absent":
        return []
    return paths[: int(cfg["vlm"]["max_images"])]


def evidence_sentence(spec: dict[str, Any], has_som: bool, calibrated: bool = False) -> str:
    parts = []
    if spec.get("use_badges", True):
        parts.append(BADGE_SENTENCE)
    else:
        parts.append(PLAIN_KF_SENTENCE)
    if spec.get("use_som", True) and has_som:
        parts.append(SOM_SENTENCE)
    if calibrated or spec.get("reliability_mix") or spec.get("arbitration"):
        parts.append(CALIBRATION_SENTENCE)
    return "".join(parts)


def single_frame_fallback(cfg: dict[str, Any], row: dict[str, Any], path: Path, tag: str) -> list[str] | None:
    import cv2
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total >= 2:
        cap.release()
        return None
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Unreadable video: {path}")
    frame_path = artifact_path(cfg, "evidence", row["clip_id"], f"single_frame_{tag}.jpg")
    if not frame_path.exists():
        alt = evidence_root(cfg) / row["clip_id"] / f"single_frame_{tag}.jpg"
        if alt.exists():
            return [str(alt), str(alt)]
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(frame_path), bgr)
    return [str(frame_path), str(frame_path)]


def video_content(cfg: dict[str, Any], row: dict[str, Any], degraded: bool = False) -> dict[str, Any]:
    if degraded:
        path = evidence_root(cfg) / row["clip_id"] / "degraded.mp4"
        if not path.exists():
            path = artifact_path(cfg, "evidence", row["clip_id"], "degraded.mp4")
    else:
        path = repo_path(cfg, row[cfg["vlm"].get("video_source", "video")])
    max_pixels = int(cfg["vlm"]["video_max_pixels"])
    frames = single_frame_fallback(cfg, row, path, "degraded" if degraded else "main")
    if frames is not None:
        return {"type": "video", "video": frames,
                "min_pixels": min(28 * 28 * 4, max_pixels), "max_pixels": max_pixels}
    return {"type": "video", "video": str(path),
            "min_pixels": min(28 * 28 * 4, max_pixels), "max_pixels": max_pixels,
            "fps": float(cfg["vlm"].get("video_fps", 4.0)), "min_frames": 2,
            "max_frames": int(cfg["vlm"]["video_max_frames"])}


def make_messages(cfg: dict[str, Any], row: dict[str, Any], spec: dict[str, Any],
                  answer: str | None, task: str = "main", degraded: bool = False,
                  evidence_row: dict[str, Any] | None = None,
                  rng: np.random.Generator | None = None,
                  corruption: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [video_content(cfg, row, degraded=degraded)]
    meta = None
    use_evidence = spec.get("evidence", False) and not (corruption and corruption.get("mode") == "absent")
    if use_evidence:
        eid = (evidence_row or row)["clip_id"]
        if corruption and corruption.get("evidence_clip_id"):
            eid = corruption["evidence_clip_id"]
        meta = load_meta(cfg, eid, spec)
        for path in evidence_images(cfg, meta, spec, rng, corruption=corruption):
            content.append({"type": "image", "image": path})
    has_som = bool(meta and any(a["type"] == "som" for a in meta.get("assets", [])))
    calibrated = bool(spec.get("reliability_mix") or spec.get("arbitration"))
    if task == "main":
        sentence = evidence_sentence(spec, has_som, calibrated) if use_evidence or calibrated else ""
        if not use_evidence and calibrated:
            sentence = CALIBRATION_SENTENCE
        prompt = MAIN_PROMPT.format(hand_focus=hand_focus(row["target_hand"]),
                                    evidence_sentence=sentence)
    elif task == "arbitration":
        prompt = evidence_sentence(spec, has_som, calibrated=True) + ARBITRATION_PROMPT
    elif task in NECESSITY_PROMPTS:
        prompt = evidence_sentence(spec, has_som) + NECESSITY_PROMPTS[task].format(hand=row["target_hand"])
    else:
        raise KeyError(f"Unknown task {task!r}")
    content.append({"type": "text", "text": prompt})
    messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages


def necessity_items(cfg: dict[str, Any], row: dict[str, Any], spec: dict[str, Any],
                    rng: np.random.Generator) -> list[dict[str, Any]]:
    if not spec.get("necessity", False):
        return []
    try:
        meta = load_meta(cfg, row["clip_id"], spec)
    except FileNotFoundError:
        return []
    necessity = meta.get("necessity", {})
    fraction = float(cfg["sft"]["necessity_fraction"])
    items = []
    if rng.random() < fraction:
        items.append({"row": row, "task": "change", "answer": str(necessity.get("max_change_after_kf", 1)),
                      "degraded": False, "corruption": None})
    if rng.random() < fraction:
        items.append({"row": row, "task": "latest", "answer": str(necessity.get("keyframe_count", 1)),
                      "degraded": False, "corruption": None})
    if spec.get("use_som", True) and "mark_nearest_hand" in necessity and rng.random() < fraction:
        items.append({"row": row, "task": "mark", "answer": str(necessity["mark_nearest_hand"]),
                      "degraded": False, "corruption": None})
    return items


def build_training_items(cfg: dict[str, Any], rows: list[dict[str, Any]], spec: dict[str, Any],
                         seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    items: list[dict[str, Any]] = []
    degrade_fraction = float(cfg["sft"]["degrade_fraction"]) if spec.get("degrade", False) else 0.0
    schedule = cfg.get("reliability", {}).get("schedule", DEFAULT_SCHEDULE)
    arb_frac = float(cfg.get("reliability", {}).get("arbitration_fraction", 0.3))
    n_kf = int(cfg.get("evidence", {}).get("num_keyframes", 10))
    for index, row in enumerate(rows):
        corruption = None
        if spec.get("reliability_mix"):
            draw = draw_corruption(row, rows, index, rng, schedule=schedule, n_keyframes=n_kf)
            corruption = {
                "mode": draw.mode,
                "evidence_clip_id": draw.evidence_clip_id,
                "shuffle_order": draw.shuffle_order,
                **draw.notes,
            }
        items.append({"row": row, "task": "main", "answer": structured_answer(row["gold_slots"]),
                      "degraded": False, "corruption": corruption})
        items.extend(necessity_items(cfg, row, spec, rng))
        if spec.get("arbitration") and rng.random() < arb_frac:
            if rng.random() < 0.5:
                draw = draw_corruption(row, rows, index, rng, mode="correct", n_keyframes=n_kf)
            else:
                draw = draw_corruption(row, rows, index, rng, mode="wrong", n_keyframes=n_kf)
            corr = {
                "mode": draw.mode,
                "evidence_clip_id": draw.evidence_clip_id,
                "shuffle_order": draw.shuffle_order,
                **draw.notes,
            }
            items.append({"row": row, "task": "arbitration", "answer": arbitration_label(draw),
                          "degraded": False, "corruption": corr})
        if degrade_fraction and rng.random() < degrade_fraction:
            deg = evidence_root(cfg, spec) / row["clip_id"] / "degraded.mp4"
            if deg.exists():
                items.append({"row": row, "task": "main", "answer": structured_answer(row["gold_slots"]),
                              "degraded": True, "corruption": None})
    order = rng.permutation(len(items))
    return [items[int(i)] for i in order]
