"""Stage 0 — build per-split clip manifests.

Each manifest line: {clip_id, label, video, target_hand, status_null, gold_slots, holistic}.
Splits:
  official  : train = all videos_train clips; test = all videos_val clips (the eval set).
  comp      : P0.5 unseen-composition (train / seen_test / comp_test) from COMP_SPLIT.
  primitive : P0.5 novel-primitive (train / seen_test / prim_test) from PRIM_SPLIT.

    python -m data.index                 # all splits
    python -m data.index --split official
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import common as C

# clips to permanently exclude (unreadable / corrupt videos found during processing)
EXCLUDE_CLIPS = {
    "S10A05I01M0_rgw_0",   # only 1 readable frame (needs >=2 for motion/C1)
}


def _rec(tax, label, clip_id, video, target):
    gs = C.gold_slots(tax, label)
    return {"clip_id": clip_id, "label": label, "video": video, "target_hand": target,
            "status_null": gs["status"] == "null", "gold_slots": gs,
            "holistic": C.holistic_desc(tax, label)}


def _write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n_active = sum(1 for r in rows if not r["status_null"])
    print(f"  {path}: {len(rows)} clips ({n_active} active, {len(rows)-n_active} null)")


def official(tax):
    tr = [_rec(tax, lab, cid, v, C.clip_target_hand(C.LH_TRAIN_ROOT))
          for lab, cid, v in C.list_clips(C.LH_TRAIN_ROOT) if cid not in EXCLUDE_CLIPS]
    te = [_rec(tax, lab, cid, v, C.clip_target_hand(C.LH_VAL_ROOT))
          for lab, cid, v in C.list_clips(C.LH_VAL_ROOT) if cid not in EXCLUDE_CLIPS]
    _write(f"{C.MANIFEST_DIR}/official_train.jsonl", tr)
    _write(f"{C.MANIFEST_DIR}/official_test.jsonl", te)


def _from_split(tax, split_file, parts, name):
    data = json.load(open(split_file))["assignment"]
    target = "left"
    for part in parts:
        rows, seen = [], set()
        for item in data.get(part, []):
            lab = item["label"]
            cid = os.path.splitext(os.path.basename(item["path"]))[0]
            if cid in seen or cid in EXCLUDE_CLIPS:
                continue
            seen.add(cid)
            rows.append(_rec(tax, lab, cid, item["path"], target))
        _write(f"{C.MANIFEST_DIR}/{name}_{part}.jsonl", rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["official", "comp", "primitive", "all"], default="all")
    args = ap.parse_args()
    tax = C.load_tax()
    if args.split in ("official", "all"):
        print("official:"); official(tax)
    if args.split in ("comp", "all"):
        print("comp (P0.5 unseen-composition):")
        _from_split(tax, C.COMP_SPLIT, ["train", "seen_test", "comp_test"], "comp")
    if args.split in ("primitive", "all"):
        print("primitive (P0.5 novel-primitive):")
        _from_split(tax, C.PRIM_SPLIT, ["train", "seen_test", "prim_test"], "primitive")
    print(f"\nmanifests -> {C.MANIFEST_DIR}/")


if __name__ == "__main__":
    main()
