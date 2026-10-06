"""Side-view training material for BOTH hands of one view, cropped identically.

View 1 is a second camera on the same HA-ViD recordings: the same clips, the
same labels, the same splits -- a frontal view of the operator across the bench
instead of view 0's overhead workspace view. Session ids end in `S1` where view
0's end in `M0`.

CROP -- both directories ship FULL frames (1280x720). The view-0 box
(450,100,1100,700) is meaningless here because the camera moved, so the box was
re-derived by data.crop_havid from two pooled signals over 160
sampled clips: accumulated frame-difference motion, and 1849 MediaPipe hand
boxes. Motion alone is useless in this view (its 97th-percentile span is 82% of
the frame -- the operator's body and the background move too), so hand coverage
carried the decision, checked against rendered crops.

    V1_CROP = (150, 110, 1000, 670)   850x560, 52% of the frame
        99.41% of detected hand boxes fall entirely inside
        75.1% of accumulated motion energy retained
        (view 0 for scale: 650x600, 42% of the frame)

The two runner-up boxes were rejected on rendered frames, not on the summary
numbers: (200,120,1030,650) and (230,130,1000,660) both clip the left edge of
the assembly box -- and with it the slide rail -- in S21A26I31S1_sftg1ws_0 f4,
S13A11I21S1_sshc1dh_0 f20 and S13A23I23S1_sshc2dh_0 f18. Losing the target
object every frame is worse than losing the 0.2% of hand boxes that separated
the candidates, especially since target_object is the weakest slot in the model.

ONE BOX FOR BOTH HANDS, for the reason prep_rh_v0 gives: rh_v0 was cropped with
the lh_v0 box so a left-vs-right difference could never be a difference of
framing. lh_v1 and rh_v1 are literally the same footage, so the shared box is
both natural and required.

Everything else mirrors prep_rh_v0: splits come from the dataset's own list
files, `_extended_` clips are excluded, `w` (wrong operation) is dropped, and
macOS `._*` AppleDouble stubs -- which make up half the file count of this
upload and are not videos -- are skipped.

    python -m data.havid --workers 24
    python -m data.havid --hand lh --dry_run
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import common as C
from data.index import EXCLUDE_CLIPS
from data.segment import Segment, cut_segment

# One box per view, shared by both hands. Each is derived by
# data.crop_havid --view <v> and then checked on rendered frames,
# because the summary numbers alone pick boxes that clip the assembly box.
#   v1  frontal view across the bench   850x560, 52% of frame, 99.41% hand coverage
#   v2  overhead view of the bench      975x620, 66% of frame, 99.17% hand coverage
# v2 needs the looser box: its content genuinely spreads wider (hand boxes span
# x 126..1266), and the tighter candidates cut the row of parts along the bottom
# of the table -- target_object is already the weakest slot, so losing parts every
# frame costs more than the 0.2% of hand boxes that separated the candidates.
CROPS = {"v1": (150, 110, 1000, 670), "v2": (300, 0, 1275, 620)}
V1_CROP = CROPS["v1"]


def hands_for(view: str) -> dict:
    place = {"v1": "frontal", "v2": "overhead"}[view]
    return {
        "lh": {"root": f"artifacts/havid_clips/{place}_left",
               "out_videos": f"artifacts/havid_clips/{place}_left_cropped",
               "artifact_root": f"artifacts/havid_{place}_left",
               "target_hand": "left"},
        "rh": {"root": f"artifacts/havid_clips/{place}_right",
               "out_videos": f"artifacts/havid_clips/{place}_right_cropped",
               "artifact_root": f"artifacts/havid_{place}_right",
               "target_hand": "right"},
    }


HANDS = hands_for("v1")
SPLITS = {"train": ("train_list_video.txt", "videos_train"),
          "test": ("val_list_video.txt", "videos_val")}
SLOTS = ("status", "verb", "manipulated_object", "target_object", "tool")

# Same reasoning as prep_rh_v0: "w" is dropped, not mapped. It does not hold up
# as an action, it was already removed from the view-0 data on both hands, and it
# would inject an active clip whose verb and objects are all "not applicable".
DROP_LABELS = {"w"}

# View 0's exclusion list is keyed by `M0` clip ids; the same recording is `S1`
# here. Verified rather than assumed: S10A05I01S1_rgw_0 also decodes to a single
# frame (1280x720, nb_read_frames=1), the same defect that got the M0 clip
# excluded, and motion/C1 need >=2 frames. Deriving the set instead of hardcoding
# it keeps view 1 in step with any future addition upstream.
def exclude_for(view: str) -> set:
    """View 0's exclusion list is keyed by `M0` clip ids; the same recording is
    `S1`/`S2` here. Verified rather than assumed for v1: S10A05I01S1_rgw_0 also
    decodes to a single frame, the same defect that got the M0 clip excluded, and
    motion/C1 need >=2 frames. Deriving the set keeps every view in step with any
    future addition upstream."""
    tag = {"v1": "S1_", "v2": "S2_"}.get(view, "S1_")
    return {c.replace("M0_", tag) for c in EXCLUDE_CLIPS} | set(EXCLUDE_CLIPS)


EXCLUDE_V1 = exclude_for("v1")

# RESOLVED 2026-09-01. Five lh_v1 sntft clips arrived truncated (mdat declaring
# more bytes than the file held, no moov, 14-45% of frames absent) across three
# byte-identical uploads; a fourth upload delivered intact files whose frame
# counts and durations match the lh_v0 counterparts exactly (e.g. 210 frames /
# 14.000s for S13A19I23S1_sntft_1). No exclusion is needed, and none should be
# added back: the rh_v1 namesakes were always intact, so a hand-agnostic
# exclusion would have silently cost rh_v1 three good clips.
#     S07A05I01S1_sntft_0  S13A19I23S1_sntft_1  S28A22I23S1_sntft_0
#     S29A26I31S1_sntft_0  S30A04I01S1_sntft_0


def read_list(path: str) -> list[tuple[str, str]]:
    """-> [(label, clip_filename)]; `_extended_` and `._*` dropped."""
    out = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        rel = line.split()[0]
        if "_extended_" in rel:
            continue
        label, name = rel.split("/", 1)
        if name.startswith("._"):
            continue                     # AppleDouble stub, ffprobe rejects it
        out.append((label, name))
    return out


def _duration(path: str) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", path],
                         capture_output=True, text=True).stdout.strip()
    return float(out) if out else 0.0


def _crop_one(job):
    src, dst_root, label, clip_id, crop = job
    try:
        d = _duration(src)
        if d <= 0:
            return clip_id, "unreadable duration"
        cut_segment(Segment(src, 0.0, d, label, clip_id, crop), dst_root)
        return clip_id, None
    except Exception as exc:
        return clip_id, str(exc)


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


def prepare(hand: str, cfg: dict, crop, workers: int, dry_run: bool) -> dict:
    tax = C.load_tax()
    exclude = cfg.get("_exclude", EXCLUDE_V1)
    man_dir = os.path.join(cfg["artifact_root"], "manifests")
    os.makedirs(man_dir, exist_ok=True)
    summary, train_rows = {}, []
    print(f"\n================ {hand}  ({cfg['root']}) ================")

    for split, (list_file, videos_dir) in SPLITS.items():
        entries = read_list(os.path.join(cfg["root"], list_file))
        jobs, rows, missing, dropped = [], [], 0, 0
        for label, name in entries:
            clip_id = os.path.splitext(name)[0]
            if clip_id in exclude:
                continue
            if label in DROP_LABELS:
                dropped += 1
                continue
            src = os.path.join(cfg["root"], videos_dir, label, name)
            if not os.path.exists(src):
                missing += 1
                continue
            dst = os.path.join(cfg["out_videos"], videos_dir, label, name)
            gs = C.gold_slots(tax, label)
            rows.append({"clip_id": clip_id, "label": label, "video": dst, "full_video": dst,
                         "description": C.holistic_desc(tax, label),
                         "target_hand": cfg["target_hand"], "gold_slots": gs,
                         "status_null": gs["status"] == "null",
                         "session_id": clip_id.split("_")[0], "view": "v1"})
            if not (os.path.exists(dst) and os.path.getsize(dst) > 0):
                jobs.append((src, os.path.join(cfg["out_videos"], videos_dir), label, clip_id, crop))

        print(f"\n[{hand}/{split}] list={list_file}")
        print(f"  entries after dropping _extended_ and ._*: {len(entries)}   source missing: {missing}")
        print(f"  dropped labels {sorted(DROP_LABELS)}: {dropped}"
              f"   excluded ids: {len(exclude)}")
        print(f"  clips: {len(rows)}   to cut now: {len(jobs)}   crop={crop}")
        if dry_run:
            continue

        errors = []
        if jobs:
            with ProcessPoolExecutor(max_workers=workers) as ex:
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
        print(f"  {path}: {len(rows)} clips ({len(rows) - n_null} active, {n_null} null)")
        summary[split] = {"clips": len(rows), "errors": len(errors)}
        if split == "train":
            train_rows = rows

    if train_rows and not dry_run:
        gram = build_grammar(train_rows)
        gp = os.path.join(man_dir, "train_grammar.json")
        json.dump(gram, open(gp, "w"), indent=1)
        print(f"  {gp}: {len(gram['tuples'])} tuples, "
              f"vocab { {s: len(v) for s, v in gram['slot_vocab'].items()} }")
        os.makedirs(os.path.join(cfg["artifact_root"], "audits"), exist_ok=True)
        json.dump({"crop": list(crop) if crop else None, "source": cfg["root"],
                   "view": "v1", "target_hand": cfg["target_hand"], "splits": summary,
                   "crop_note": "derived by data.crop_havid; 99.41% hand "
                                "coverage, 75.1% motion energy; shared with the other hand"},
                  open(os.path.join(cfg["artifact_root"], "audits", "prep_v1.json"), "w"), indent=1)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--view", default="v1", choices=sorted(CROPS))
    ap.add_argument("--hand", choices=["lh", "rh", "both"], default="both")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--crop", default="", help="x1,y1,x2,y2 or 'none'; default = CROPS[view]")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    if args.crop == "none":
        crop = None
    elif args.crop:
        crop = tuple(int(v) for v in args.crop.split(","))
    else:
        crop = CROPS[args.view]
    table = hands_for(args.view)
    excl = exclude_for(args.view)
    hands = ["lh", "rh"] if args.hand == "both" else [args.hand]
    for hand in hands:
        cfg = dict(table[hand], _exclude=excl)
        prepare(hand, cfg, crop, args.workers, args.dry_run)


if __name__ == "__main__":
    main()
