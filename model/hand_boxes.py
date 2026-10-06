"""Per-keyframe box of the *asked-about* hand, aligned to the vision feature grid.

WHY THIS EXISTS
---------------
In the pooled dual-hand setting the two hands are told apart by exactly one
thing: the sentence "Focus only on the left/right hand." Everything else is
identical -- the crop box is literally the same constant (450,100,1100,700) for
both hands (data/havid_side.py), and for ~45% of the 1762 clip_ids
shared between the hands the annotated time window is the same too, with the
same label. A text clause names the hand but never says *where it is*, so the
adapter's evidence grid carries no way to act on it. Measured consequence: the
pooled model over-calls "active" on idle right-hand clips (34.2% vs 18.9% for
the right-hand specialist), which alone is over half its 0.104 accuracy deficit.

This script supplies the missing signal. For every evidence keyframe it runs the
same MediaPipe HandLandmarker the evidence builder uses, applies the same
mirror correction, and follows the same nearest-centre track, then stores the
target hand's box in NORMALISED image coordinates.

Boxes are computed on the keyframe JPEGs, which are the exact images
features_vlm encodes into the (T,4,4,1152) grid, so box -> grid cell
is a direct mapping with no resampling assumption.

This is not SoM. Nothing is drawn and nothing enters the prompt; the geometry is
consumed numerically inside the adapter (model/hand_anchored.py).

    python -m model.hand_boxes \
        --config configs/havid.yaml \
        --ev_config configs/evidence_havid_side_right.yaml \
        --splits train,test_lh,test_rh --workers 12
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

_CFG: dict | None = None


def _load_ev_cfg(path: str) -> dict:
    from data.evidence_config import load_config
    return load_config(path)


def _init(ev_config: str) -> None:
    global _CFG
    _CFG = _load_ev_cfg(ev_config)


def track_boxes(image_paths: list[str], target_hand: str, cfg: dict
                ) -> tuple[list[list[float] | None], str]:
    """Same detector, mirror correction and nearest-centre track as
    data.evidence.hand_track, but over pre-extracted keyframe images."""
    from PIL import Image
    from data.evidence import _detect_hands_tasks_api, _detect_hands_legacy_api

    frames = [np.asarray(Image.open(p).convert("RGB")) for p in image_paths]
    per_frame = _detect_hands_tasks_api(frames, cfg)
    tracker = "mediapipe_tasks"
    if per_frame is None:
        per_frame = _detect_hands_legacy_api(frames, cfg)
        tracker = "mediapipe"
    if per_frame is None:
        return [None] * len(frames), "unavailable"

    mirrored = bool(cfg["evidence"].get("mediapipe_input_mirrored", False))
    track: list[list[float] | None] = []
    previous = None
    for detections in per_frame:
        candidates = []
        for box, label in detections:
            if not mirrored and label in {"left", "right"}:
                label = "right" if label == "left" else "left"
            centre = np.asarray([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            candidates.append((box, centre, label))
        if not candidates:
            track.append(None)
            continue
        matching = [c for c in candidates if c[2] == target_hand]
        if matching:
            candidates = matching
        if previous is None:
            chosen = max(candidates, key=lambda c: (c[0][2] - c[0][0]) * (c[0][3] - c[0][1]))
        else:
            chosen = min(candidates, key=lambda c: float(np.linalg.norm(c[1] - previous)))
        track.append([float(v) for v in chosen[0]])
        previous = chosen[1]
    return track, tracker


def process(job: tuple[str, str, str, str]) -> tuple[str, str]:
    clip_id, evidence_dir, target_hand, out_path = job
    if os.path.exists(out_path):
        return clip_id, "exists"
    meta_path = Path(evidence_dir) / "meta.json"
    if not meta_path.exists():
        return clip_id, "no_meta"
    meta = json.loads(meta_path.read_text())
    kf = sorted((a for a in meta["assets"] if a["type"] == "kf"), key=lambda a: a["ordinal"])
    # plain_* carries no badge overlay -- cleanest input for the detector. The
    # badged and plain frames are the same pixels underneath, so a box measured
    # on one is valid on the other.
    paths = []
    for a in kf:
        plain = Path(evidence_dir) / f"plain_{a['path']}"
        paths.append(str(plain if plain.exists() else Path(evidence_dir) / a["path"]))
    if not paths:
        return clip_id, "no_keyframes"

    boxes, tracker = track_boxes(paths, target_hand, _CFG)
    coverage = float(np.mean([b is not None for b in boxes])) if boxes else 0.0
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps({
        "clip_id": clip_id, "hand": target_hand, "tracker": tracker,
        "coverage": coverage, "n_keyframes": len(paths), "boxes": boxes}))
    return clip_id, "ok"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/havid.yaml")
    ap.add_argument("--ev_config", default="configs/evidence_havid_side_right.yaml")
    ap.add_argument("--splits", default="train,test_lh,test_rh")
    ap.add_argument("--evidence_dirname", default="evidence")
    ap.add_argument("--output_dirname", default="hand_boxes")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, os.getcwd())
    from model.config import load_config, artifact_path
    cfg = load_config(args.config)

    jobs, seen = [], set()
    for split in args.splits.split(","):
        path = artifact_path(cfg, "manifests", f"{split}.jsonl")
        if not path.exists():
            print(f"  skip missing split {path}")
            continue
        rows = [json.loads(l) for l in open(path)]
        for row in rows:
            clip_id = row["clip_id"]
            if clip_id in seen:
                continue
            seen.add(clip_id)
            hand = row.get("target_hand", "left")
            if hand not in {"left", "right"}:
                continue          # hand-agnostic clips have no asked-about hand
            jobs.append((clip_id,
                         str(artifact_path(cfg, args.evidence_dirname, clip_id)),
                         hand,
                         str(artifact_path(cfg, args.output_dirname, f"{clip_id}.json"))))
        print(f"  {split}: {len(rows)} rows")
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"  {len(jobs)} unique clips -> {artifact_path(cfg, args.output_dirname)}")

    counts: dict[str, int] = {}
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args.ev_config,)) as ex:
        futures = [ex.submit(process, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            _, status = fut.result()
            counts[status] = counts.get(status, 0) + 1
            if i % 500 == 0:
                print(f"  {i}/{len(jobs)}  {counts}", flush=True)
    print(f"  done: {counts}")

    out_root = artifact_path(cfg, args.output_dirname)
    cov = [json.loads(p.read_text())["coverage"] for p in sorted(out_root.glob("*.json"))]
    if cov:
        cov = np.asarray(cov)
        print(f"  coverage: mean={cov.mean():.3f}  zero={float((cov == 0).mean()):.3f}  "
              f"full={float((cov == 1).mean()):.3f}")


if __name__ == "__main__":
    main()
