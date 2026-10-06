"""Build the pooled left+right ("dual-hand") split for a single unified model.

One model trained on both hands, evaluated separately on each, tests whether the
method needs a per-hand specialist at all.

THE BLOCKER THIS SOLVES
-----------------------
clip_ids collide across hands: 1762 of them are shared between lh_train and
rh_train, and 414 between the test sets. The same id names two genuinely
different clips -- different crop (lh_v0 vs rh_v0_cropped) and different
temporal bounds (e.g. S03A05I01M0_ibacb_0 is 3.27s on the left, 1.00s on the
right). Evidence and features are keyed by clip_id
(`evidence/{clip_id}/meta.json`, `features_vlm/{clip_id}.npy`), so pooling
without disambiguation would silently make one hand read the other's evidence.

Rows therefore get an `lh_`/`rh_` prefix on clip_id, and the merged evidence and
feature trees are built as SYMLINKS to the per-hand originals -- no copying, and
the originals stay untouched.

What makes the pooled setup sound, verified on the current manifests:
  - no cross-split leakage: lh_train ∩ rh_test = 0 and rh_train ∩ lh_test = 0
  - shared label space: verbs 6/6, tools 5/4, manip 24/25, target 27/28,
    union of valid tuples 73 (vs 67 and 68 alone)
  - colliding ids carry the SAME label in 1762/1762 cases, i.e. the two hands
    annotate the same action from their own perspective
  - the prompt already conditions on the hand via
    data.common.hand_focus -> "Focus only on the left/right hand."
    so one model can be told which hand to answer for.

    python -m model.pool --out_root artifacts/havid
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path

SLOTS = ("status", "verb", "manipulated_object", "target_object", "tool")
ELEMENT = SLOTS[1:]

HANDS = {
    "lh": {
        "train": "artifacts/havid_side_left/manifests/official_train_orig_now.jsonl",
        "test": "artifacts/havid_side_left/manifests/official_test_now.jsonl",
        "evidence": "artifacts/evidence",
        "features": "artifacts/havid_side_left/features_vlm",
    },
    "rh": {
        "train": "artifacts/havid_side_right/manifests/train.jsonl",
        "test": "artifacts/havid_side_right/manifests/test.jsonl",
        "evidence": "artifacts/havid_side_right/evidence",
        "features": "artifacts/havid_side_right/features_vlm",
    },
}


def build_grammar(rows: list[dict]) -> dict:
    vocab = {s: set() for s in SLOTS}
    counts = {s: collections.Counter() for s in SLOTS}
    tuples = set()
    for r in rows:
        g = r["gold_slots"]
        for s in SLOTS:
            vocab[s].add(g.get(s, ""))
            counts[s][g.get(s, "")] += 1
        tuples.add(tuple(g.get(s, "") for s in ELEMENT))
    return {"slot_vocab": {s: sorted(vocab[s]) for s in SLOTS},
            "tuples": sorted(list(t) for t in tuples),
            "class_counts": {s: dict(sorted(counts[s].items())) for s in SLOTS}}


def link(src: Path, dst: Path) -> bool:
    if dst.exists() or dst.is_symlink():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src.resolve(), dst)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out_root", default="artifacts/havid")
    ap.add_argument("--repo", default=".")
    args = ap.parse_args()

    root = Path(args.repo) / args.out_root
    man = root / "manifests"
    man.mkdir(parents=True, exist_ok=True)
    ev_root, ft_root = root / "evidence", root / "features_vlm"

    pooled = {"train": [], "test": []}
    n_ev = n_ft = miss_ev = miss_ft = 0
    for hand, paths in HANDS.items():
        for split in ("train", "test"):
            rows = [json.loads(l) for l in open(Path(args.repo) / paths[split])]
            for r in rows:
                old = r["clip_id"]
                new = f"{hand}_{old}"
                row = dict(r)
                row["clip_id"] = new
                row["hand_source"] = hand
                row["orig_clip_id"] = old
                pooled[split].append(row)

                # A plain symlink of the evidence DIRECTORY is not enough:
                # evidence_images() builds the image root from meta["clip_id"],
                # not from the manifest row, so a symlinked dir whose meta.json
                # still carries the unprefixed id sends every lookup to
                # evidence/<old_id>/kf01.jpg and the run dies with FileNotFound.
                # So: real directory, rewritten meta.json, symlinked images.
                src_ev = Path(args.repo) / paths["evidence"] / old
                if src_ev.is_dir():
                    dst_ev = ev_root / new
                    if not (dst_ev / "meta.json").exists():
                        dst_ev.mkdir(parents=True, exist_ok=True)
                        meta = json.loads((src_ev / "meta.json").read_text())
                        meta["clip_id"] = new
                        meta["orig_clip_id"] = old
                        (dst_ev / "meta.json").write_text(json.dumps(meta))
                        for f in src_ev.iterdir():
                            if f.name != "meta.json":
                                link(f, dst_ev / f.name)
                        n_ev += 1
                else:
                    miss_ev += 1
                for ext in (".npy", ".npz.npy", ".npz"):
                    src_ft = Path(args.repo) / paths["features"] / f"{old}{ext}"
                    if src_ft.exists():
                        n_ft += link(src_ft, ft_root / f"{new}{ext}")
                        break
                else:
                    miss_ft += 1
            print(f"  {hand} {split}: {len(rows)} rows")

    for split, rows in pooled.items():
        p = man / f"{split}.jsonl"
        with open(p, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        by = collections.Counter(r["hand_source"] for r in rows)
        print(f"  {p}: {len(rows)} rows  {dict(by)}")

    # per-hand eval manifests: one model, scored separately on each hand
    for hand in HANDS:
        rows = [r for r in pooled["test"] if r["hand_source"] == hand]
        p = man / f"test_{hand}.jsonl"
        with open(p, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"  {p}: {len(rows)} rows")

    gram = build_grammar(pooled["train"])
    (man / "train_grammar.json").write_text(json.dumps(gram, indent=1))
    print(f"  train_grammar.json: {len(gram['tuples'])} tuples, "
          f"vocab { {s: len(v) for s, v in gram['slot_vocab'].items()} }")

    report = {"train": len(pooled["train"]), "test": len(pooled["test"]),
              "evidence_links": n_ev, "feature_links": n_ft,
              "missing_evidence": miss_ev, "missing_features": miss_ft,
              "tuples": len(gram["tuples"])}
    (root / "audits").mkdir(parents=True, exist_ok=True)
    (root / "audits" / "build_dual.json").write_text(json.dumps(report, indent=1))
    print(f"\n  evidence links={n_ev} (missing {miss_ev})   feature links={n_ft} (missing {miss_ft})")


if __name__ == "__main__":
    main()
