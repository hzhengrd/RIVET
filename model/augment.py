"""Long-tail augmentation for the pooled dual-hand split.

WHY THIS IS *NOT* OVERSAMPLING
------------------------------
The pooled build already ran the duplication experiment for us. Merging the two
hands roughly doubled every tail class's training count, and every accuracy
bucket got *worse* (right-hand test, epoch 4, pooled vs the right-hand
specialist):

    freq bucket   rh train -> pooled    specialist -> pooled
    <10             63 ->  113 (1.79x)    0.0952 -> 0.0476
    10-29          377 ->  849 (2.25x)    0.4333 -> 0.3417
    30-99          931 -> 1879 (2.02x)    0.6261 -> 0.5522

So more *copies* of the same footage does not help; ~45% of the 1762 shared
clip_ids are the same time window under the same crop box, i.e. near-identical
frames carrying the same label. Augmentation therefore has to produce genuinely
different pixels, which means new keyframes, new evidence and new vision
features -- not a reweighted sampler.

WHAT THE ERRORS SAY TO AUGMENT
------------------------------
On the 141 tail-class test clips the specialist's slot accuracies are

    tool 0.986 | status 0.894 | verb 0.731 | manipulated_object 0.560 | target_object 0.475

and only 17% of the failures are one-slot-off. Tail failure is not a fuzzy
decision boundary that more samples would sharpen; it is the model not resolving
*which object*. The two ops below are chosen for that:

  temporal sub-window  a different span of the clip -> the keyframe selector
                       lands on different frames -> genuinely different evidence
                       images and a different feature grid.
  spatial jitter       translate + rescale inside the fixed (450,100,1100,700)
                       workspace box -> the same part appears at a different
                       scale and image position, which is exactly the invariance
                       manipulated_object / target_object are missing.

NO HORIZONTAL FLIP. A mirror swaps which side of the frame each hand is on while
the prompt still says "Focus only on the left hand", so it would corrupt the one
signal the dual-hand model is supposed to learn.

K IS CAPPED
-----------
Filling every class to `--target` would demand 59 variants of a 1-clip class.
Past a handful of variants the crops overlap so heavily that the result is
oversampling with noise -- the thing shown above not to work. `--max_per_clip`
(default 4) caps it, so rare classes grow by a bounded factor and simply do not
reach the target.

    python -m model.augment \
        --manifest artifacts/havid/manifests/train.jsonl \
        --out_manifest artifacts/havid/manifests/train_aug_only.jsonl \
        --video_root artifacts/augmented_clips/tail --workers 24
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

# Ops are sampled from these ranges, seeded per (clip_id, k) so a rerun
# reproduces the same clips and `--resume` is meaningful.
WINDOW_FRAC = (0.78, 0.92)   # fraction of the clip duration kept
ZOOM_FRAC = (0.00, 0.08)     # fraction of width/height cropped away before rescale
MIN_SECONDS = 0.60           # never cut a clip below this
MIN_SOURCE_SECONDS = 0.80    # clips shorter than this get zoom-only variants


def probe(path: str) -> tuple[float, int, int] | None:
    """-> (duration_s, width, height)"""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-show_entries", "format=duration",
         "-of", "json", path], capture_output=True, text=True)
    if out.returncode != 0:
        return None
    try:
        blob = json.loads(out.stdout)
        stream = blob["streams"][0]
        return float(blob["format"]["duration"]), int(stream["width"]), int(stream["height"])
    except (KeyError, IndexError, ValueError):
        return None


def probe_all(paths: list[str], workers: int, cache_path: str | None = None) -> dict[str, tuple]:
    """Probe every source clip once, in parallel, with an on-disk cache.

    The planner needs (duration, width, height) per source clip. Calling probe()
    inside the planning loop is both serial and redundant -- the same clip is
    probed once per variant index k, and ffprobe is a subprocess whose cost is
    dominated by process setup rather than by the read. On HA-ViD (991
    candidates) that was invisible; on Assembly101, where the cap-200 split
    yields ~25.7k candidates, the planning phase alone ran for hours at ~1
    probe/s while the CPU sat idle waiting on subprocesses.

    Probing is I/O-bound, so a thread pool is enough, and the result is cached so
    that a re-run costs nothing."""
    cache: dict[str, tuple] = {}
    if cache_path and os.path.exists(cache_path):
        try:
            cache = {k: tuple(v) for k, v in json.load(open(cache_path)).items()}
        except (ValueError, OSError):
            cache = {}
    todo = [p for p in dict.fromkeys(paths) if p not in cache]
    if todo:
        print(f"  probing {len(todo)} clips with {workers} threads "
              f"({len(cache)} already cached)", flush=True)
        with ThreadPoolExecutor(workers) as ex:
            for i, (path, info) in enumerate(zip(todo, ex.map(probe, todo)), 1):
                if info is not None:
                    cache[path] = info
                if i % 2000 == 0:
                    print(f"    probed {i}/{len(todo)}", flush=True)
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            json.dump({k: list(v) for k, v in cache.items()}, open(cache_path, "w"))
    return cache


def plan_op(clip_id: str, k: int, duration: float, width: int, height: int) -> dict:
    """Deterministic (temporal window, spatial box) for the k-th variant."""
    rng = random.Random(f"{clip_id}#{k}")
    zoom = rng.uniform(*ZOOM_FRAC)
    crop_w = max(16, int(round(width * (1.0 - zoom))) // 2 * 2)
    crop_h = max(16, int(round(height * (1.0 - zoom))) // 2 * 2)
    x = rng.randint(0, max(0, width - crop_w))
    y = rng.randint(0, max(0, height - crop_h))

    if duration < MIN_SOURCE_SECONDS:
        start, length = 0.0, duration          # too short to trim; zoom only
    else:
        frac = rng.uniform(*WINDOW_FRAC)
        length = max(MIN_SECONDS, duration * frac)
        length = min(length, duration)
        start = rng.uniform(0.0, max(0.0, duration - length))
    return {"start": round(start, 3), "length": round(length, 3),
            "crop": (crop_w, crop_h, x, y), "out_w": width, "out_h": height}


def render(job: tuple[str, str, dict]) -> tuple[str, bool, str]:
    src, dst, op = job
    if os.path.exists(dst) and os.path.getsize(dst) > 1024:
        return dst, True, "exists"
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    cw, ch, x, y = op["crop"]
    vf = f"crop={cw}:{ch}:{x}:{y},scale={op['out_w']}:{op['out_h']}"
    tmp = dst + ".part.mp4"
    cmd = ["ffmpeg", "-y", "-v", "error",
           "-ss", str(op["start"]), "-t", str(op["length"]), "-i", src,
           "-vf", vf, "-an", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "20", "-pix_fmt", "yuv420p", tmp]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(tmp) or os.path.getsize(tmp) < 1024:
        if os.path.exists(tmp):
            os.remove(tmp)
        return dst, False, (proc.stderr or "empty output")[:200]
    os.replace(tmp, dst)
    return dst, True, "ok"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", default="artifacts/havid/manifests/train.jsonl")
    ap.add_argument("--out_manifest", default="artifacts/havid/manifests/train_aug_only.jsonl")
    ap.add_argument("--merged_manifest", default="artifacts/havid/manifests/train_aug.jsonl")
    ap.add_argument("--video_root", default="artifacts/augmented_clips/tail")
    ap.add_argument("--target", type=int, default=60, help="per-class sample budget")
    ap.add_argument("--max_per_clip", type=int, default=4)
    ap.add_argument("--skip_labels", default="null", help="comma-separated labels never augmented")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--probe_cache", default="",
                    help="JSON cache of (duration,width,height) per source clip")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.manifest)]
    skip = {s for s in args.skip_labels.split(",") if s}
    freq = collections.Counter(r["label"] for r in rows)
    by_label: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)

    # ---- probe every candidate source once, in parallel, before planning
    candidates: list[str] = []
    for label, count in freq.items():
        if label in skip or count >= args.target:
            continue
        candidates.extend(r["video"] for r in by_label[label] if os.path.exists(r["video"]))
    info_of = probe_all(candidates, args.workers, args.probe_cache or None)

    # ---- plan: round-robin over a class's clips so variants spread out evenly
    jobs, new_rows = [], []
    per_label_new: dict[str, int] = {}
    for label, count in sorted(freq.items()):
        if label in skip or count >= args.target:
            continue
        need = args.target - count
        clips = sorted(by_label[label], key=lambda r: r["clip_id"])
        made = 0
        for k in range(1, args.max_per_clip + 1):
            for row in clips:
                if made >= need:
                    break
                src = row["video"]
                info = info_of.get(src)
                if info is None:
                    continue
                op = plan_op(row["clip_id"], k, *info)
                new_id = f"{row['clip_id']}_aug{k}"
                dst = os.path.join(args.video_root, f"{new_id}.mp4")
                out = dict(row)
                out["clip_id"] = new_id
                out["video"] = dst
                out["full_video"] = dst
                out["aug_of"] = row["clip_id"]
                out["aug_op"] = {"start": op["start"], "length": op["length"],
                                 "crop": list(op["crop"])}
                jobs.append((src, dst, op))
                new_rows.append(out)
                made += 1
            if made >= need:
                break
        per_label_new[label] = made

    print(f"  source: {len(rows)} rows, {len(freq)} labels")
    print(f"  augmenting {len(per_label_new)} labels -> {len(new_rows)} new clips "
          f"(cap {args.max_per_clip}/clip, target {args.target})")
    short = {l: freq[l] + n for l, n in per_label_new.items() if freq[l] + n < args.target}
    print(f"  capped below target: {len(short)} labels, e.g. "
          f"{dict(sorted(short.items(), key=lambda x: x[1])[:5])}")
    by_hand = collections.Counter(r.get("hand_source", "?") for r in new_rows)
    print(f"  new clips by hand: {dict(by_hand)}")
    if args.dry_run:
        return

    ok = fail = 0
    with ProcessPoolExecutor(args.workers) as ex:
        futures = [ex.submit(render, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            _, good, msg = fut.result()
            ok += good
            if not good:
                fail += 1
                if fail <= 5:
                    print(f"  render failed: {msg}")
            if i % 200 == 0:
                print(f"  rendered {i}/{len(jobs)}", flush=True)

    kept = [r for r in new_rows if os.path.exists(r["video"])]
    Path(args.out_manifest).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_manifest, "w") as fh:
        for r in kept:
            fh.write(json.dumps(r) + "\n")
    with open(args.merged_manifest, "w") as fh:
        for r in rows + kept:
            fh.write(json.dumps(r) + "\n")

    final = collections.Counter(r["label"] for r in rows + kept)
    tail = [l for l in freq if l not in skip and freq[l] < 30]
    print(f"\n  rendered ok={ok} failed={fail}")
    print(f"  {args.out_manifest}: {len(kept)} rows")
    print(f"  {args.merged_manifest}: {len(rows) + len(kept)} rows")
    print(f"  tail(<30) classes: {len(tail)}, "
          f"{sum(freq[l] for l in tail)} -> {sum(final[l] for l in tail)} rows")


if __name__ == "__main__":
    main()
