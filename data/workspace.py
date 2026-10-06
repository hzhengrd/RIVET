"""Per-scan workspace ROI for IKEA ASM (and any full-scene external dataset).

WHY
---
HA-ViD's training input is a cropped workspace view (`lh_v0`). IKEA ASM's dev3
camera instead frames the whole room, so the person and table occupy a small
part of the frame. Measured on a real clip, MediaPipe finds **0 hands in 48
full frames**, but 9-10 detections in the same 16 frames after a workspace crop.
Feeding one dataset close-up workspace evidence and the other a wide room shot
makes any cross-dataset comparison a comparison of framing, not of method.

HOW
---
The camera is static per scan, so a single rectangle suffices for the whole
scan (a fixed rectangle also avoids frame-to-frame jitter that would corrupt
the wrist kinematics). The ROI is the bounding box of sustained motion --
where the person and the furniture parts actually are -- computed by
accumulating absolute frame differences over frames sampled across the scan,
thresholding, and padding.

    python -m data.workspace \
        --videos artifacts/ikea/videos --camera dev3 \
        --out artifacts/ikea/workspace_roi.json [--preview_dir /tmp/roi_preview]
"""
from __future__ import annotations
import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np


def motion_accumulator(path: str, samples: int = 60) -> np.ndarray | None:
    """Sum of |frame diff| over frames spread across the whole video."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total < 2:
        cap.release()
        return None
    idx = np.linspace(0, total - 1, min(samples, total)).astype(int)
    acc, prev = None, None
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            d = np.abs(g - prev)
            acc = d if acc is None else acc + d
        prev = g
    cap.release()
    return acc


def _centroid_window(acc: np.ndarray, max_frac: float, pad: float) -> tuple[int, int, int, int]:
    """Motion-mass-weighted centroid + a window of area `max_frac`, with the
    aspect ratio taken from the motion's second moments.

    Fallback for scans where the motion bounding box blows up -- floor assembly
    has the person moving over the whole floor, so the box of all above-threshold
    motion covers most of the frame and crops nothing useful. Measured on such
    scans: bbox 71-100% -> 0 hand detections, centroid window 27% -> 1-11."""
    H, W = acc.shape
    blur = cv2.GaussianBlur(acc, (0, 0), sigmaX=max(W, H) / 200.0)
    m = blur - blur.min()
    m = m / (m.sum() + 1e-9)
    ys, xs = np.mgrid[0:H, 0:W]
    cx, cy = (m * xs).sum(), (m * ys).sum()
    sx = np.sqrt((m * (xs - cx) ** 2).sum())
    sy = np.sqrt((m * (ys - cy) ** 2).sum())
    area = max_frac * W * H
    ar = max(sx, 1.0) / max(sy, 1.0)
    h = np.sqrt(area / ar) * (1 + pad)
    w = ar * np.sqrt(area / ar) * (1 + pad)
    x1, x2 = int(max(0, cx - w / 2)), int(min(W, cx + w / 2))
    y1, y2 = int(max(0, cy - h / 2)), int(min(H, cy + h / 2))
    x2 -= (x2 - x1) % 2
    y2 -= (y2 - y1) % 2
    return x1, y1, x2, y2


def roi_from_motion(acc: np.ndarray, pct: float = 96.0, pad: float = 0.12,
                    min_frac: float = 0.25, max_frac: float = 0.45) -> tuple[int, int, int, int]:
    """Bounding box of the top-(100-pct)% most-moving pixels, padded, clamped,
    and grown to cover at least `min_frac` of each dimension so a nearly static
    scan never yields a degenerate crop."""
    H, W = acc.shape
    blur = cv2.GaussianBlur(acc, (0, 0), sigmaX=max(W, H) / 200.0)
    thr = np.percentile(blur, pct)
    ys, xs = np.where(blur >= thr)
    if xs.size == 0:
        return 0, 0, W, H
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    px, py = int(pad * (x2 - x1)), int(pad * (y2 - y1))
    x1, x2 = max(0, x1 - px), min(W, x2 + px)
    y1, y2 = max(0, y1 - py), min(H, y2 + py)
    # enforce a minimum extent, centred on the motion box
    if (x2 - x1) < min_frac * W:
        cx, half = (x1 + x2) // 2, int(min_frac * W / 2)
        x1, x2 = max(0, cx - half), min(W, cx + half)
    if (y2 - y1) < min_frac * H:
        cy, half = (y1 + y2) // 2, int(min_frac * H / 2)
        y1, y2 = max(0, cy - half), min(H, cy + half)
    # A box that covers most of the frame has localized nothing; fall back to the
    # centroid window, which stays tight even when the motion is spread out.
    if (x2 - x1) * (y2 - y1) > max_frac * W * H:
        return _centroid_window(acc, max_frac * 0.5, pad)
    # ffmpeg's crop wants even dimensions for yuv420p
    x2 -= (x2 - x1) % 2
    y2 -= (y2 - y1) % 2
    return x1, y1, x2, y2


def _one(job):
    scan, path, samples, preview_dir = job
    try:
        acc = motion_accumulator(path, samples)
        if acc is None:
            return scan, None, "unreadable"
        roi = roi_from_motion(acc)
        if preview_dir:
            cap = cv2.VideoCapture(path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 2) // 2)
            ok, frame = cap.read()
            cap.release()
            if ok:
                x1, y1, x2, y2 = roi
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                os.makedirs(preview_dir, exist_ok=True)
                cv2.imwrite(os.path.join(preview_dir, scan.replace("/", "__") + ".jpg"), frame)
        return scan, roi, None
    except Exception as e:
        return scan, None, repr(e)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--videos", required=True, help="root holding <Furniture>/<scan>/<camera>/images/")
    ap.add_argument("--camera", default="dev3")
    ap.add_argument("--basename", default="scan_video.avi")
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=60)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--preview_dir", default="")
    args = ap.parse_args()

    pattern = os.path.join(args.videos, "*", "*", args.camera, "images", args.basename)
    paths = sorted(glob.glob(pattern))
    jobs = []
    for p in paths:
        parts = p.split(os.sep)
        scan = f"{parts[-5]}/{parts[-4]}"
        jobs.append((scan, p, args.samples, args.preview_dir))
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"deriving workspace ROI for {len(jobs)} scans (camera {args.camera})")

    rois, errors = {}, {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_one, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            scan, roi, err = f.result()
            if err:
                errors[scan] = err
            else:
                rois[scan] = list(roi)
            if i % 50 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)}  ok={len(rois)} err={len(errors)}", flush=True)

    fracs = [((r[2] - r[0]) * (r[3] - r[1])) for r in rois.values()]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"camera": args.camera, "rois": rois, "errors": errors}, open(args.out, "w"), indent=1)
    print(f"\n-> {args.out}   scans={len(rois)}  errors={len(errors)}")
    if fracs:
        a = np.array(fracs, float)
        print(f"   crop area px: min={a.min():.0f} median={np.median(a):.0f} max={a.max():.0f}")


if __name__ == "__main__":
    main()
