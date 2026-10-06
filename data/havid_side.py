"""Right-hand (rh_v0) training/eval data, cropped identically to lh_v0.

Splits come from the dataset's own list files, not a directory walk:
    artifacts/havid_clips/rh_v0/train_list_video.txt   (7208 lines)
    artifacts/havid_clips/rh_v0/val_list_video.txt     ( 602 lines)
each line `"<label>/<clip>.mp4 <class_index>"`. `_extended_` clips are excluded
(4830 of the train lines; the val list has none).

CROP -- rh_v0 ships FULL FRAMES (1280x720) while lh_v0 is already a cropped
workspace view (650x600). Training the right hand on full frames while the left
hand used crops would make any left-vs-right difference a difference of framing.
The lh_v0 crop box was recovered by template-matching lh_v0 frames against
lh_v0_original: a constant (450, 100, 1100, 700) at 0.999 match on every clip
sampled. The same box is applied here -- the camera is fixed, and the box was
verified to cover the assembly box, the parts and both hands on right-hand
clips too.

    python -m data.havid_side --workers 24
"""
from __future__ import annotations
import argparse
import collections
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import common as C
from data.index import EXCLUDE_CLIPS
from data.segment import Segment, cut_segment

RH_ROOT = "artifacts/havid_clips/rh_v0"
# recovered from lh_v0 vs lh_v0_original by template matching (constant, 0.999)
LH_CROP = (450, 100, 1100, 700)
SPLITS = {"train": ("train_list_video.txt", "videos_train"),
          "test": ("val_list_video.txt", "videos_val")}
SLOTS = ("status", "verb", "manipulated_object", "target_object", "tool")

# "w" (wrong operation) is dropped, not mapped: the class does not hold up as an
# action, and it was already removed from the left-hand data. Keeping it would
# also inject an active clip whose verb and objects are all "not applicable",
# which composition_validity() correctly rejects.
DROP_LABELS = {"w"}


def read_list(path: str) -> list[tuple[str, str]]:
    """-> [(label, clip_filename)], `_extended_` dropped."""
    out = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        rel = line.split()[0]
        if "_extended_" in rel:
            continue
        label, name = rel.split("/", 1)
        out.append((label, name))
    return out


def _crop_one(job):
    src, dst_root, label, clip_id, crop = job
    try:
        cut_segment(Segment(src, 0.0, _duration(src), label, clip_id, crop), dst_root)
        return clip_id, None
    except Exception as e:
        return clip_id, str(e)


def _duration(path: str) -> float:
    import subprocess
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", path], capture_output=True, text=True).stdout.strip()
    return float(out) if out else 0.0


def build_grammar(rows: list[dict]) -> dict:
    vocab = {s: set() for s in SLOTS}
    counts = {s: collections.Counter() for s in SLOTS}
    tuples = set()
    for r in rows:
        g = r["gold_slots"]
        for s in SLOTS:
            vocab[s].add(g.get(s, ""))
            counts[s][g.get(s, "")] += 1
        tuples.add(tuple(g.get(s, "") for s in SLOTS[1:]))
    return {"slot_vocab": {s: sorted(vocab[s]) for s in SLOTS},
            "tuples": sorted(list(t) for t in tuples),
            "class_counts": {s: dict(sorted(counts[s].items())) for s in SLOTS}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rh_root", default=RH_ROOT)
    ap.add_argument("--out_videos", default="artifacts/havid_clips/rh_v0_cropped")
    ap.add_argument("--artifact_root", default="artifacts/havid_side_right")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no_crop", action="store_true", help="keep full frames (not recommended)")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    crop = None if args.no_crop else LH_CROP
    tax = C.load_tax()
    man_dir = os.path.join(args.artifact_root, "manifests")
    os.makedirs(man_dir, exist_ok=True)
    summary, train_rows = {}, []

    for split, (list_file, videos_dir) in SPLITS.items():
        entries = read_list(os.path.join(args.rh_root, list_file))
        jobs, rows, missing = [], [], 0
        dropped = 0
        for label, name in entries:
            clip_id = os.path.splitext(name)[0]
            if clip_id in EXCLUDE_CLIPS:
                continue
            if label in DROP_LABELS:
                dropped += 1
                continue
            src = os.path.join(args.rh_root, videos_dir, label, name)
            if not os.path.exists(src):
                missing += 1
                continue
            dst = os.path.join(args.out_videos, videos_dir, label, name)
            gs = C.gold_slots(tax, label)
            rows.append({"clip_id": clip_id, "label": label, "video": dst, "full_video": dst,
                         "description": C.holistic_desc(tax, label), "target_hand": "right",
                         "gold_slots": gs, "status_null": gs["status"] == "null",
                         "session_id": clip_id.split("_")[0]})
            if not (os.path.exists(dst) and os.path.getsize(dst) > 0):
                jobs.append((src, os.path.join(args.out_videos, videos_dir), label, clip_id, crop))

        print(f"\n[{split}] list={list_file}")
        print(f"  entries after dropping _extended_: {len(entries)}   source missing: {missing}")
        print(f"  dropped labels {sorted(DROP_LABELS)}: {dropped}")
        print(f"  clips: {len(rows)}   to cut now: {len(jobs)}   crop={crop}")
        if args.dry_run:
            continue

        errors = []
        if jobs:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(_crop_one, j) for j in jobs]
                for i, f in enumerate(as_completed(futs), 1):
                    cid, err = f.result()
                    if err:
                        errors.append((cid, err))
                    if i % 300 == 0 or i == len(futs):
                        print(f"    {i}/{len(futs)}  errors={len(errors)}", flush=True)
        if errors:
            bad = {c for c, _ in errors}
            rows = [r for r in rows if r["clip_id"] not in bad]
            print(f"  !! {len(errors)} failed, e.g. {errors[0]}")

        path = os.path.join(man_dir, f"{split}.jsonl")
        with open(path, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        n_null = sum(1 for r in rows if r["status_null"])
        print(f"  {path}: {len(rows)} clips ({len(rows)-n_null} active, {n_null} null)")
        summary[split] = {"clips": len(rows), "errors": len(errors)}
        if split == "train":
            train_rows = rows

    if train_rows and not args.dry_run:
        gram = build_grammar(train_rows)
        gp = os.path.join(man_dir, "train_grammar.json")
        json.dump(gram, open(gp, "w"), indent=1)
        print(f"\n  {gp}: {len(gram['tuples'])} tuples, "
              f"vocab { {s: len(v) for s, v in gram['slot_vocab'].items()} }")
        os.makedirs(os.path.join(args.artifact_root, "audits"), exist_ok=True)
        json.dump({"crop": crop, "source": args.rh_root, "splits": summary,
                   "lh_crop_box_note": "recovered from lh_v0 vs lh_v0_original, constant 0.999 match"},
                  open(os.path.join(args.artifact_root, "audits", "prep_rh_v0.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
