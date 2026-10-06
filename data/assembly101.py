"""Reorganize Assembly101 fine-grained action annotations into the training manifest
contract.

Assembly101 ships one continuous ~7min video per (session, camera view) plus a
CSV of (start_frame, end_frame, verb_cls, noun_cls, action_cls) segments. This
script cuts the segments of one chosen camera view into
<out_root>/<label>/<clip_id>.mp4 and writes gold_slots manifests.

Slot decomposition (validated to cover all 1380 fine-grained classes exactly,
with Assembly101's own vocabulary -- never mapped onto HA-ViD's):
  tool    <- " with X" suffix of action_cls ("with hand" -> "no tool")
  verb    <- verb_cls, with the two compound verbs split so the object that is
             actually manipulated lands in the right slot:
               "position screw on" + noun N -> verb=position, manip=screw, target=N
               "remove screw from" + noun N -> verb=remove,   manip=screw, target=N
  manip   <- noun_cls (or "screw" for the two compound verbs above)
  target  <- noun_cls for compound verbs, "other hand" for verb "pass",
             else "not applicable" (Assembly101 has no general target slot)
  status  <- always "active" (the dataset has no idle/background class)

Frame numbers in the annotations are at 30fps while the raw videos are 60fps
(verified: a session's max end_frame 14669 -> 488.97s fits its 493.43s video),
so seconds = frame / 30.0.

    python -m data.assembly101 \
        --annotations APT_Datasets/assembly101/raw/annotations/fine-grained-annotations \
        --recordings artifacts/assembly101_videos/recordings \
        --view v4 --out_root artifacts/assembly101_videos/clips --workers 8
"""
from __future__ import annotations
import argparse
import collections
import csv
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import external as CE
from data.segment import Segment, cut_segment

# camera-ID mapping from the official download-scripts README. v4/C10119 is the
# overhead view with the best single-view action-recognition accuracy in the
# Assembly101 paper (Table 6: v4 43.6 vs v1 43.1, side views v5 27.8 / v7 33.3).
VIEW_TO_CAMERA = {
    "v1": "C10095_rgb", "v2": "C10115_rgb", "v3": "C10118_rgb", "v4": "C10119_rgb",
    "v5": "C10379_rgb", "v6": "C10390_rgb", "v7": "C10395_rgb", "v8": "C10404_rgb",
}

ANNOT_FPS = 30.0          # annotations are indexed at 30fps; raw video is 60fps
SPLIT_FILES = {"train": "train.csv", "validation": "validation.csv", "test": "test.csv"}

_WITH_RE = re.compile(r"\s+with\s+(.+)$")
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slug(s: str) -> str:
    return _SLUG_RE.sub("_", s.strip().lower()).strip("_")


def decompose(action_cls: str, verb_cls: str, noun_cls: str) -> tuple[str, str, str, str]:
    """action/verb/noun -> (verb, manipulated_object, target_object, tool)."""
    tool, target, manip, verb = "no tool", "not applicable", noun_cls, verb_cls
    m = _WITH_RE.search(action_cls)
    if m:
        t = m.group(1).strip()
        tool = "no tool" if t == "hand" else t
    if verb_cls.endswith(" screw on"):
        verb, manip, target = verb_cls[: -len(" screw on")], "screw", noun_cls
    elif verb_cls.endswith(" screw from"):
        verb, manip, target = verb_cls[: -len(" screw from")], "screw", noun_cls
    elif verb_cls == "pass":
        target = "other hand"
    return verb, manip, target, tool


def build_taxonomy(actions_csv: str) -> CE.ExternalTaxonomy:
    rows = []
    for r in csv.DictReader(open(actions_csv)):
        v, m, t, tool = decompose(r["action_cls"], r["verb_cls"], r["noun_cls"])
        rows.append(CE.ActionRow(label=slug(r["action_cls"]), verb=v, manipulated_object=m,
                                 target_object=t, tool=tool, description=r["action_cls"]))
    return CE.build_external_taxonomy(rows)


def session_tag(session: str) -> str:
    """Long session dir -> compact, still-unique clip-id prefix.
    'nusar-2021_action_both_9011-a01_9011_user_id_2021-02-01_153724' -> '9011-a01_153724'"""
    m = re.match(r"nusar-\d+_action_both_([\w-]+?)_\d+_user_id_[\d-]+_(\d+)$", session)
    return f"{m.group(1)}_{m.group(2)}" if m else slug(session)


def load_head_actions(annot_dir: str) -> set[str]:
    """The dataset's official 142 head action classes (head_actions.txt).

    Assembly101's fine-grained label space is extremely long-tailed: of the 1244
    classes present in the v4 train split, 52% have <10 instances and 117 are
    singletons. Every class is kept, but the head/tail flag rides along in the
    manifest so the primary metric can be reported on head classes -- matching
    the split the dataset authors define, so numbers stay comparable to the
    published baselines."""
    path = os.path.join(annot_dir, "head_actions.txt")
    if not os.path.exists(path):
        return set()
    return {slug(l.strip()) for l in open(path) if l.strip()}


def read_segments(annot_dir: str, split: str, camera: str) -> list[dict]:
    path = os.path.join(annot_dir, SPLIT_FILES[split])
    out = []
    for r in csv.DictReader(open(path)):
        if not r["video"].endswith(f"/{camera}.mp4"):
            continue
        out.append({
            "session": r["video"].split("/")[0],
            "start_frame": int(r["start_frame"]), "end_frame": int(r["end_frame"]),
            "action_cls": r["action_cls"], "toy_id": r["toy_id"],
            "toy_name": r["toy_name"], "is_shared": r["is_shared"],
        })
    return out


def _cut_one(job):
    seg, out_root = job
    try:
        cut_segment(seg, out_root)
        return seg.clip_id, None
    except Exception as e:  # a corrupt/truncated source video must not kill the run
        return seg.clip_id, str(e)


def prepare_split(tax, annot_dir, recordings, camera, out_root, split, args, head=frozenset()):
    rows = read_segments(annot_dir, split, camera)
    have, missing_sessions = {}, set()
    for r in rows:
        src = os.path.join(recordings, r["session"], f"{camera}.mp4")
        if os.path.exists(src) and os.path.getsize(src) > 0:
            have.setdefault(r["session"], src)
        else:
            missing_sessions.add(r["session"])
    rows = [r for r in rows if r["session"] in have]

    kept, per_class, per_session = [], collections.Counter(), collections.Counter()
    for r in rows:
        dur = (r["end_frame"] - r["start_frame"]) / ANNOT_FPS
        if dur < args.min_duration:
            continue
        label = slug(r["action_cls"])
        if args.max_per_class and per_class[label] >= args.max_per_class:
            continue
        per_class[label] += 1
        tag = session_tag(r["session"])
        idx = per_session[(tag, label)]
        per_session[(tag, label)] += 1
        kept.append((r, label, f"{tag}_{label}_{idx}", dur))

    print(f"\n[{split}] view={camera}")
    print(f"  annotated segments: {len(read_segments(annot_dir, split, camera))}")
    print(f"  sessions available / missing: {len(have)} / {len(missing_sessions)}")
    print(f"  segments to cut after filters: {len(kept)}  ({len(per_class)} action classes)")
    if args.dry_run:
        return None

    jobs, records, skipped = [], [], 0
    for r, label, clip_id, dur in kept:
        dest = os.path.join(out_root, label, f"{clip_id}.mp4")
        rec = CE.clip_record(tax, label, clip_id, dest, target_hand="both")
        rec.update(session_id=r["session"], toy_id=r["toy_id"], toy_name=r["toy_name"],
                   is_shared=r["is_shared"], source_split=split, is_head=label in head,
                   start_sec=round(r["start_frame"] / ANNOT_FPS, 3),
                   end_sec=round(r["end_frame"] / ANNOT_FPS, 3))
        records.append(rec)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            skipped += 1
            continue
        jobs.append((Segment(have[r["session"]], r["start_frame"] / ANNOT_FPS,
                             r["end_frame"] / ANNOT_FPS, label, clip_id), out_root))

    print(f"  already cut (resume): {skipped}   to cut now: {len(jobs)}")
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
    return records, sorted(missing_sessions), errors


def main():
    ap = argparse.ArgumentParser(description="Assembly101 -> training manifests")
    ap.add_argument("--annotations", default="APT_Datasets/assembly101/raw/annotations/fine-grained-annotations")
    ap.add_argument("--recordings", default="artifacts/assembly101_videos/recordings")
    ap.add_argument("--view", default="v4", choices=sorted(VIEW_TO_CAMERA))
    ap.add_argument("--out_root", default="artifacts/assembly101_videos/clips")
    ap.add_argument("--manifest_dir", default="artifacts/manifests")
    ap.add_argument("--splits", nargs="+", default=["train", "validation", "test"], choices=sorted(SPLIT_FILES))
    ap.add_argument("--min_duration", type=float, default=0.2, help="drop degenerate sub-frame segments")
    ap.add_argument("--max_per_class", type=int, default=0, help="0 = keep all")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry_run", action="store_true", help="report counts, cut nothing")
    args = ap.parse_args()

    camera = VIEW_TO_CAMERA[args.view]
    tax = build_taxonomy(os.path.join(args.annotations, "actions.csv"))
    print(f"taxonomy: {len(tax.label_to_tuple)} action classes, "
          f"{len(tax.slot_vocab['verb'])} verbs, {len(tax.slot_vocab['manipulated_object'])} objects, "
          f"{len(tax.slot_vocab['tool'])} tools")

    head = load_head_actions(args.annotations)
    print(f"head action classes (official head_actions.txt): {len(head)}")

    summary = {}
    for split in args.splits:
        res = prepare_split(tax, args.annotations, args.recordings, camera,
                            args.out_root, split, args, head=head)
        if res is None:
            continue
        records, missing, errors = res
        name = {"train": "train", "validation": "val", "test": "test"}[split]
        CE.write_manifest(os.path.join(args.manifest_dir, f"assembly101_{args.view}_{name}.jsonl"), records)
        summary[split] = {"clips": len(records), "missing_sessions": missing, "errors": len(errors),
                          "head_clips": sum(1 for r in records if r["is_head"])}

    if summary and not args.dry_run:
        os.makedirs(args.manifest_dir, exist_ok=True)
        gram = {
            "dataset": "assembly101", "view": args.view, "camera": camera,
            "head_actions": sorted(head),
            "annot_fps": ANNOT_FPS,
            "slot_vocab": {k: sorted(v) for k, v in tax.slot_vocab.items()},
            "label_to_tuple": {k: list(v) for k, v in tax.label_to_tuple.items()},
            "label_to_desc": dict(tax.label_to_desc),
            "splits": summary,
        }
        p = os.path.join(args.manifest_dir, f"assembly101_{args.view}_grammar.json")
        with open(p, "w") as f:
            json.dump(gram, f, indent=2)
        print(f"\ngrammar -> {p}")


if __name__ == "__main__":
    main()
