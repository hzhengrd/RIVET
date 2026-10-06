"""Apply a workspace crop to already-trimmed action clips, in place of re-cutting.

Assembly101 clips were cut temporally from the continuous recordings but left at
the full 1920x1080 frame, because the fixed view v4 already fills most of the
frame with the bench (96% reliable hand localisation without a crop, against 0%
for IKEA ASM's room-wide view before cropping). Adding a crop afterwards does
not require going back to the source recordings: the clips are already trimmed,
so a spatial crop of each clip file is sufficient and far cheaper.

The crop is derived exactly as for the HA-ViD side views
(data.crop_havid): accumulated frame-difference motion plus the
empirical quantiles of MediaPipe hand boxes, over a sample of clips, with the
candidates checked on rendered frames before adoption.

Manifests are rewritten to point at the cropped tree; every other field,
including clip_id, is preserved, so the existing splits, head cap and
augmentation plan all remain valid and downstream artefacts keyed by clip_id
line up unchanged.

    python -m data.crop \
        --manifests artifacts/assembly101/manifests/train_cap200.jsonl \
                    artifacts/assembly101/manifests/validation_nat.jsonl \
                    artifacts/assembly101/manifests/test.jsonl \
        --crop 440,10,1880,1050 --out_root artifacts/assembly101_videos/clips_c \
        --workers 24
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def crop_one(job: tuple[str, str, tuple[int, int, int, int]]) -> tuple[str, bool, str]:
    src, dst, crop = job
    if os.path.exists(dst) and os.path.getsize(dst) > 1024:
        return dst, True, "exists"
    x1, y1, x2, y2 = crop
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    tmp = dst + ".part.mp4"
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", src,
           "-vf", f"crop={x2 - x1}:{y2 - y1}:{x1}:{y1}",
           "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
           "-pix_fmt", "yuv420p", tmp]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(tmp) or os.path.getsize(tmp) < 1024:
        if os.path.exists(tmp):
            os.remove(tmp)
        return dst, False, (proc.stderr or "empty output")[:200]
    os.replace(tmp, dst)
    return dst, True, "ok"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--crop", required=True, help="x1,y1,x2,y2 in source pixels")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--src_root", default="artifacts/assembly101_videos/clips")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--suffix", default="_c", help="written manifest gets this suffix")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    crop = tuple(int(v) for v in args.crop.split(","))
    jobs: dict[str, tuple] = {}
    per_manifest: list[tuple[str, list[dict]]] = []
    for m in args.manifests:
        rows = [json.loads(l) for l in open(m)]
        for r in rows:
            src = r["video"]
            rel = os.path.relpath(src, args.src_root)
            dst = os.path.join(args.out_root, rel)
            jobs[src] = (src, dst, crop)
            r["video"] = dst
            r["full_video"] = dst
        per_manifest.append((m, rows))
        print(f"  {m}: {len(rows)} rows")

    todo = [j for j in jobs.values() if not (os.path.exists(j[1]) and os.path.getsize(j[1]) > 1024)]
    print(f"  unique source clips: {len(jobs)}   to crop now: {len(todo)}   crop={crop}")
    if args.dry_run:
        return

    ok = fail = 0
    if todo:
        with ProcessPoolExecutor(args.workers) as ex:
            futures = [ex.submit(crop_one, j) for j in todo]
            for i, fut in enumerate(as_completed(futures), 1):
                _, good, msg = fut.result()
                ok += good
                if not good:
                    fail += 1
                    if fail <= 5:
                        print(f"  crop failed: {msg}")
                if i % 2000 == 0:
                    print(f"  cropped {i}/{len(todo)}  failed={fail}", flush=True)
    print(f"  cropped ok={ok} failed={fail}")

    for m, rows in per_manifest:
        kept = [r for r in rows if os.path.exists(r["video"])]
        out = m.replace(".jsonl", f"{args.suffix}.jsonl")
        with open(out, "w") as fh:
            for r in kept:
                fh.write(json.dumps(r) + "\n")
        lost = len(rows) - len(kept)
        print(f"  {out}: {len(kept)} rows" + (f"   ({lost} dropped: crop failed)" if lost else ""))


if __name__ == "__main__":
    main()
