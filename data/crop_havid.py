"""Derive ONE shared crop window for the view-1 pair (lh_v1 + rh_v1).

WHY IT HAS TO BE DERIVED, NOT REUSED
------------------------------------
View 0's box is the constant (450,100,1100,700) recovered by template-matching
lh_v0 against lh_v0_original. View 1 is a different camera, so that box means
nothing here; the workspace sits somewhere else in the frame.

WHY ONE BOX FOR BOTH HANDS
--------------------------
Exactly the reasoning that produced the view-0 pair: rh_v0 was cropped with the
lh_v0 box so that any left-vs-right difference could not be a difference of
framing (data/havid_side.py). lh_v1 and rh_v1 are the same recordings
from the same camera -- only the annotation differs -- so a single box is both
correct and required for the comparison to mean anything. Evidence is therefore
pooled across the two directories before the box is fitted.

HOW THE BOX IS CHOSEN
---------------------
Two independent signals, and the box must satisfy both:

  motion    frame-to-frame absolute difference accumulated over sampled frames of
            sampled clips -> where the assembly actually happens.
  hands     MediaPipe HandLandmarker boxes on the same frames -> where the hands
            actually go. A crop that clips a hand would silently damage the very
            signal the dual-hand model depends on, so hand coverage is a hard
            constraint, not a tiebreak.

The reported box is the union of (motion percentile box) and (hand box hull at
the requested coverage), padded, clamped, and rounded to even dimensions so
libx264 accepts it.

    python -m data.crop_havid --clips 80 --out_dir outputs/v1_crop
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())

DIRS = {"lh_v1": "artifacts/havid_clips/lh_v1",
        "rh_v1": "artifacts/havid_clips/rh_v1"}


def dirs_for(view: str) -> dict[str, str]:
    """Both hands of one view. They are the same footage under two annotations,
    so evidence is pooled across them before the box is fitted -- one box per
    view, never one per hand (see prep_rh_v0 for why framing must not differ)."""
    return {f"{h}_{view}": f"artifacts/havid_clips/{h}_{view}" for h in ("lh", "rh")}
LIST = {"train": "train_list_video.txt", "val": "val_list_video.txt"}
_HAND_CFG: dict | None = None


def read_list(root: str, name: str, drop_extended: bool = True) -> list[str]:
    out = []
    for line in open(os.path.join(root, name)):
        line = line.strip()
        if not line:
            continue
        rel = line.split()[0]
        if drop_extended and "_extended_" in rel:
            continue
        if os.path.basename(rel).startswith("._"):
            continue          # macOS AppleDouble stub, not a video
        out.append(rel)
    return out


def _init(ev_config: str) -> None:
    global _HAND_CFG
    from data.evidence_config import load_config
    _HAND_CFG = load_config(ev_config)


def _one(job: tuple[str, int]) -> dict | None:
    """-> {'motion': (H,W) float32, 'hands': [xyxy px, ...], 'shape': (H,W)}"""
    import cv2
    path, samples = job
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 1:
        cap.release()
        return None
    idx = np.unique(np.linspace(0, total - 1, min(samples, total)).astype(int))
    frames, prev, acc = [], None, None
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if acc is None:
            acc = np.zeros_like(gray)
        if prev is not None:
            acc += np.abs(gray - prev)
        prev = gray
        frames.append(rgb)
    cap.release()
    if acc is None or not frames:
        return None

    hands = []
    keep = frames[:: max(1, len(frames) // 8)][:8]
    try:
        from data.evidence import _detect_hands_tasks_api
        per_frame = _detect_hands_tasks_api(keep, _HAND_CFG)
    except Exception:
        per_frame = None
    if per_frame:
        h, w = acc.shape
        for dets in per_frame:
            for box, _lab in dets:
                hands.append([box[0] * w, box[1] * h, box[2] * w, box[3] * h])
    return {"motion": acc, "hands": hands, "shape": acc.shape}


def box_from_motion(acc: np.ndarray, pct: float) -> tuple[int, int, int, int]:
    """Tightest box holding `pct` percent of the accumulated motion mass."""
    col = acc.sum(0)
    row = acc.sum(1)

    def span(v):
        c = np.cumsum(v) / max(v.sum(), 1e-9)
        lo = float((1.0 - pct / 100.0) / 2.0)
        hi = 1.0 - lo
        return int(np.searchsorted(c, lo)), int(np.searchsorted(c, hi))

    x1, x2 = span(col)
    y1, y2 = span(row)
    return x1, y1, x2, y2


def box_from_hands(hands: np.ndarray, cover: float) -> tuple[int, int, int, int]:
    """Box containing `cover` fraction of all detected hand boxes entirely."""
    if not len(hands):
        return 0, 0, 0, 0
    lo = (1.0 - cover) / 2.0 * 100.0
    hi = 100.0 - lo
    return (int(np.percentile(hands[:, 0], lo)), int(np.percentile(hands[:, 1], lo)),
            int(np.percentile(hands[:, 2], hi)), int(np.percentile(hands[:, 3], hi)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--clips", type=int, default=80, help="clips sampled per directory")
    ap.add_argument("--samples", type=int, default=40, help="frames sampled per clip")
    ap.add_argument("--motion_pct", type=float, default=97.0)
    ap.add_argument("--hand_cover", type=float, default=0.99)
    ap.add_argument("--pad", type=float, default=0.04, help="fraction of frame size")
    ap.add_argument("--ev_config", default="configs/evidence_havid_side_right.yaml")
    ap.add_argument("--view", default="v1", help="v1, v2, ... -> lh_<view>/rh_<view>")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    dirs = dirs_for(args.view)
    args.out_dir = args.out_dir or f"outputs/{args.view}_crop"
    jobs = []
    for tag, root in dirs.items():
        rels = read_list(root, LIST["train"]) + read_list(root, LIST["val"])
        sub = "videos_train"
        picks = rng.sample(rels, min(args.clips, len(rels)))
        for rel in picks:
            for sub in ("videos_train", "videos_val"):
                p = os.path.join(root, sub, rel)
                if os.path.exists(p):
                    jobs.append((p, args.samples))
                    break
        print(f"  {tag}: {len(rels)} clips (extended dropped), sampled {len(picks)}")

    acc = None
    hands: list[list[float]] = []
    shape = None
    ok = 0
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args.ev_config,)) as ex:
        futures = [ex.submit(_one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            if r is None:
                continue
            ok += 1
            shape = r["shape"]
            acc = r["motion"] if acc is None else acc + r["motion"]
            hands.extend(r["hands"])
            if i % 40 == 0:
                print(f"  {i}/{len(jobs)} clips, {len(hands)} hand boxes", flush=True)

    H, W = shape
    hands_a = np.asarray(hands, dtype=np.float32) if hands else np.zeros((0, 4), np.float32)
    mb = box_from_motion(acc, args.motion_pct)
    hb = box_from_hands(hands_a, args.hand_cover)
    px, py = args.pad * W, args.pad * H
    x1 = int(max(0, min(mb[0], hb[0] if hands_a.size else mb[0]) - px))
    y1 = int(max(0, min(mb[1], hb[1] if hands_a.size else mb[1]) - py))
    x2 = int(min(W, max(mb[2], hb[2] if hands_a.size else mb[2]) + px))
    y2 = int(min(H, max(mb[3], hb[3] if hands_a.size else mb[3]) + py))
    x2 -= (x2 - x1) % 2
    y2 -= (y2 - y1) % 2

    inside = 0.0
    if hands_a.size:
        inside = float(np.mean((hands_a[:, 0] >= x1) & (hands_a[:, 1] >= y1)
                               & (hands_a[:, 2] <= x2) & (hands_a[:, 3] <= y2)))
    kept = float(acc[y1:y2, x1:x2].sum() / max(acc.sum(), 1e-9))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {"frame": [W, H], "clips_ok": ok, "hand_boxes": len(hands),
              "motion_box": list(mb), "hand_box": [int(v) for v in hb],
              "crop": [x1, y1, x2, y2], "crop_wh": [x2 - x1, y2 - y1],
              "area_frac": round((x2 - x1) * (y2 - y1) / (W * H), 4),
              "hands_fully_inside": round(inside, 4),
              "motion_energy_kept": round(kept, 4)}
    (out / "crop_report.json").write_text(json.dumps(report, indent=1))
    np.save(out / "motion_acc.npy", acc)
    print("\n" + json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
