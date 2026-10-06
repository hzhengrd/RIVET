"""Reorganize IKEA ASM atomic-action annotations into the training manifest contract.

IKEA ASM ships one continuous video per (scan, camera) as
<root>/<Furniture>/<scan>/<dev>/images/scan_video.avi, plus gt_segments.json
giving contiguous [start_frame, end_frame] atomic-action segments per scan and
the official cross-environment train/test split. This script cuts the segments
of one chosen camera into <out_root>/<label>/<clip_id>.mp4 and writes
gold_slots manifests.

Slot decomposition (validated to cover all 33 gt_segments labels exactly, with
IKEA ASM's own vocabulary -- never mapped onto HA-ViD's):
  "NA"                                    -> status null (the idle state)
  "align leg screw with table thread"     -> verb=align,  manip=leg screw, target=table thread
  "attach shelf to table"                 -> verb=attach, manip=shelf,     target=table
  "pick up leg" / "spin leg" / ...        -> verb=<longest matching verb>, manip=rest
  "position the drawer right side up"     -> verb=position, manip=drawer (special-cased;
                                             "right side up" is manner, not an object)
  tool is always "no tool" -- IKEA ASM annotates no tools.

NOTE " with " means the TARGET here ("align leg screw with table thread"),
unlike Assembly101 where " with " means the TOOL ("unscrew wheel with
screwdriver"). The two datasets therefore need separate parsers.

Frame numbers are native video indices (verified: a scan's max end_frame 3248
vs the annotation DB's nframes 3249) and segments are inclusive, so
seconds = frame / fps.

The fps is PROBED from each video, never assumed: the paper says ~24fps but the
released encodings are 25fps (verified on Lack_TV_Bench/0025_black_table_...:
25/1 fps, 3250 frames, 130.00s; annotated frames 0..3248 -> 129.96s at 25fps,
whereas 24fps would give 135.38s, i.e. 5.4s *longer than the video*). Assuming
24 would drift every clip progressively onto the wrong action.

    python -m data.ikea \
        --annotations APT_Datasets/ikea_asm/raw/action_annotations/gt_segments.json \
        --videos artifacts/ikea/videos --camera dev3 \
        --out_root artifacts/ikea/clips --workers 8
"""
from __future__ import annotations
import argparse
import collections
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import external as CE
from data.segment import Segment, cut_segment

# dev3 is the default camera throughout the official IKEA ASM action code
# (IKEAActionDataset.py, the C3D/P3D/I3D train scripts) and is released as the
# standalone "RGB top view" package.
CAMERAS = ["dev1", "dev2", "dev3"]
VIDEO_BASENAME = "scan_video.avi"
NA_LABEL = "NA"
FALLBACK_FPS = 25.0   # observed in the released encodings; only used if ffprobe fails

# verbs of the 32 atomic actions, longest-first matching
VERBS = ["pick up", "lay down", "align", "attach", "position", "insert", "slide",
         "spin", "tighten", "rotate", "flip", "push", "other"]
# "right side up" is manner, not an object -> the manipulated object is the drawer
SPECIAL = {"position the drawer right side up": ("position", "drawer", "not applicable")}
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slug(s: str) -> str:
    return _SLUG_RE.sub("_", s.strip().lower()).strip("_")


def decompose(label: str):
    """label -> (verb, manipulated_object, target_object) or None for the idle class."""
    if label == NA_LABEL:
        return None
    if label in SPECIAL:
        return SPECIAL[label]
    if label == "other":
        return ("other", "not applicable", "not applicable")
    rest, target = label, "not applicable"
    for sep in (" with ", " to "):
        if sep in rest:
            rest, target = rest.split(sep, 1)
            break
    for v in sorted(VERBS, key=len, reverse=True):
        if rest == v or rest.startswith(v + " "):
            return (v, rest[len(v):].strip() or "not applicable", target)
    raise ValueError(f"undecomposable IKEA ASM label: {label!r}")


def build_taxonomy(labels) -> CE.ExternalTaxonomy:
    rows = []
    for lab in sorted(labels):
        d = decompose(lab)
        if d is None:          # NA -> label "null"; CE.gold_slots yields the idle state
            continue
        v, m, t = d
        rows.append(CE.ActionRow(label=slug(lab), verb=v, manipulated_object=m,
                                 target_object=t, tool="no tool", description=lab))
    # NA is registered as the "null" idle label so idle clips score like HA-ViD's
    return CE.build_external_taxonomy(rows, include_null=True)


def label_code(label: str) -> str:
    return "null" if label == NA_LABEL else slug(label)


def scan_video_path(videos_root: str, scan: str, camera: str) -> str:
    return os.path.join(videos_root, scan, camera, "images", VIDEO_BASENAME)


def probe_fps(path: str) -> float:
    """Read the real frame rate rather than assuming it -- a wrong fps silently
    shifts every clip onto a different action."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=avg_frame_rate", "-of", "default=nw=1:nk=1", path],
            capture_output=True, text=True, check=True).stdout.strip()
        num, den = (out.split("/") + ["1"])[:2]
        fps = float(num) / float(den or 1)
        return fps if fps > 0 else FALLBACK_FPS
    except Exception:
        return FALLBACK_FPS


def _cut_one(job):
    seg, out_root = job
    try:
        cut_segment(seg, out_root)
        return seg.clip_id, None
    except Exception as e:
        return seg.clip_id, str(e)


def main():
    ap = argparse.ArgumentParser(description="IKEA ASM -> training manifests")
    ap.add_argument("--annotations", default="APT_Datasets/ikea_asm/raw/action_annotations/gt_segments.json")
    ap.add_argument("--videos", default="data/ikea_asm/videos")
    ap.add_argument("--camera", default="dev3", choices=CAMERAS)
    ap.add_argument("--out_root", default="data/ikea_asm/clips")
    ap.add_argument("--manifest_dir", default="artifacts/manifests")
    ap.add_argument("--min_duration", type=float, default=0.2)
    ap.add_argument("--max_duration", type=float, default=0.0, help="0 = no cap (NA runs reach ~105s)")
    ap.add_argument("--include_na", action="store_true", help="also cut the NA idle segments")
    ap.add_argument("--max_per_class", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--workspace_roi", default="",
                    help="JSON from derive_workspace_roi.py; crops each clip to its scan's "
                         "workspace so evidence is comparable to HA-ViD's cropped view")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    rois = {}
    if args.workspace_roi:
        rois = {k: tuple(v) for k, v in json.load(open(args.workspace_roi))["rois"].items()}
        print(f"workspace ROIs loaded: {len(rois)} scans")

    db = json.load(open(args.annotations))["database"]
    tax = build_taxonomy({a["label"] for v in db.values() for a in v["annotation"]})
    print(f"taxonomy: {len(tax.label_to_tuple)} action classes, "
          f"{len(tax.slot_vocab['verb'])} verbs, {len(tax.slot_vocab['manipulated_object'])} objects")

    by_split = collections.defaultdict(list)
    missing, fps_cache = set(), {}
    per_class, per_scan = collections.Counter(), collections.Counter()
    for scan, entry in sorted(db.items()):
        src = scan_video_path(args.videos, scan, args.camera)
        if not (os.path.exists(src) and os.path.getsize(src) > 0):
            missing.add(scan)
            continue
        if src not in fps_cache:
            fps_cache[src] = probe_fps(src)
        fps = fps_cache[src]
        split = "train" if entry["subset"] == "training" else "test"
        tag = slug(scan)
        for ann in entry["annotation"]:
            lab = ann["label"]
            if lab == NA_LABEL and not args.include_na:
                continue
            s, e = ann["segment"]
            dur = (e - s + 1) / fps                      # segments are inclusive
            if dur < args.min_duration or (args.max_duration and dur > args.max_duration):
                continue
            code = label_code(lab)
            if args.max_per_class and per_class[code] >= args.max_per_class:
                continue
            per_class[code] += 1
            idx = per_scan[(tag, code)]
            per_scan[(tag, code)] += 1
            by_split[split].append((scan, code, f"{tag}_{code}_{idx}", src, s / fps, (e + 1) / fps))

    print(f"\ncamera={args.camera}")
    print(f"  scans available / missing: {len(db) - len(missing)} / {len(missing)}")
    for split in ("train", "test"):
        print(f"  {split}: {len(by_split[split])} segments")
    if args.dry_run:
        return

    summary = {}
    for split in ("train", "test"):
        items = by_split[split]
        if not items:
            continue
        jobs, records, skipped = [], [], 0
        for scan, code, clip_id, src, t0, t1 in items:
            dest = os.path.join(args.out_root, code, f"{clip_id}.mp4")
            rec = CE.clip_record(tax, code, clip_id, dest, target_hand="both")
            rec.update(scan_id=scan, furniture=scan.split("/")[0], camera=args.camera,
                       source_split=split, start_sec=round(t0, 3), end_sec=round(t1, 3))
            records.append(rec)
            if os.path.exists(dest) and os.path.getsize(dest) > 0:
                skipped += 1
                continue
            jobs.append((Segment(src, t0, t1, code, clip_id, rois.get(scan)), args.out_root))

        print(f"\n[{split}] already cut: {skipped}   to cut now: {len(jobs)}")
        errors = []
        if jobs:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(_cut_one, j) for j in jobs]
                for i, fut in enumerate(as_completed(futs), 1):
                    cid, err = fut.result()
                    if err:
                        errors.append((cid, err))
                    if i % 500 == 0 or i == len(futs):
                        print(f"    cut {i}/{len(futs)}  (errors: {len(errors)})", flush=True)
        if errors:
            print(f"  !! {len(errors)} clips failed, e.g. {errors[0]}")
            bad = {c for c, _ in errors}
            records = [r for r in records if r["clip_id"] not in bad]
        CE.write_manifest(os.path.join(args.manifest_dir, f"ikea_asm_{args.camera}_{split}.jsonl"), records)
        summary[split] = {"clips": len(records), "errors": len(errors)}

    if summary:
        gram = {
            "dataset": "ikea_asm", "camera": args.camera,
            "fps": sorted({round(f, 3) for f in fps_cache.values()}),
            "workspace_roi": bool(rois),
            "slot_vocab": {k: sorted(v) for k, v in tax.slot_vocab.items()},
            "label_to_tuple": {k: list(v) for k, v in tax.label_to_tuple.items()},
            "label_to_desc": dict(tax.label_to_desc),
            "missing_scans": sorted(missing), "splits": summary,
        }
        p = os.path.join(args.manifest_dir, f"ikea_asm_{args.camera}_grammar.json")
        with open(p, "w") as f:
            json.dump(gram, f, indent=2)
        print(f"\ngrammar -> {p}")


if __name__ == "__main__":
    main()
