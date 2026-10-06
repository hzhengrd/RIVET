"""Adapt Assembly101 and IKEA ASM manifests to the training layout.

data.assembly101 and data.ikea emit a slightly different row than the trainer
reads, so this converts rather than duplicating the cutting work:

  prepared row                      training row
  ----------------------------------------------------------------
  clip_id, label, video,            same
  target_hand, gold_slots           same
  holistic                       -> description
  (n/a)                          -> full_video    (= video; these datasets have
                                    no separate full-frame source, unlike
                                    HA-ViD's lh_v0 vs lh_v0_original)
  session_id / scan_id           -> session_id
  is_head, start_sec, ...           carried through for analysis

Grammar: model.grammar.apply_grammar() needs `slot_vocab` (INCLUDING
`status`, which the prepared grammar omits) plus `tuples` and `class_counts`,
which the prepared grammar does not emit at all. `tuples` is the valid
composition set and is derived HERE FROM THE TRAIN SPLIT ONLY -- taking it from
the full taxonomy would hand the model compositions its training split never
contained.

    python -m model.external_adapter --dataset assembly101_v4 \
        --artifact_root artifacts/assembly101
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

SLOTS = ("status", "verb", "manipulated_object", "target_object", "tool")
ELEMENT_SLOTS = SLOTS[1:]

# prepared manifest name -> training split name
SPLIT_ALIASES = {"train": "train", "val": "validation", "test": "test"}


def to_v5_row(r: dict) -> dict:
    out = {
        "clip_id": r["clip_id"],
        "label": r["label"],
        "video": r["video"],
        "full_video": r["video"],          # no separate full-frame source
        "description": r.get("holistic", ""),
        "target_hand": r.get("target_hand", "both"),
        "gold_slots": r["gold_slots"],
        "session_id": r.get("session_id") or r.get("scan_id", ""),
    }
    for k in ("is_head", "start_sec", "end_sec", "status_null", "toy_name", "furniture"):
        if k in r:
            out[k] = r[k]
    return out


def build_grammar(train_rows: list[dict]) -> dict:
    """slot_vocab / tuples / class_counts in the shape apply_grammar() expects.

    Derived from the TRAIN split's gold slots, matching how HA-ViD's
    *_grammar.json is built (its official_train grammar has 67 tuples, not the
    full taxonomy)."""
    vocab = {s: set() for s in SLOTS}
    counts = {s: collections.Counter() for s in SLOTS}
    tuples = set()
    for r in train_rows:
        g = r["gold_slots"]
        for s in SLOTS:
            v = g.get(s, "")
            vocab[s].add(v)
            counts[s][v] += 1
        tuples.add(tuple(g.get(s, "") for s in ELEMENT_SLOTS))
    return {
        "slot_vocab": {s: sorted(vocab[s]) for s in SLOTS},
        "tuples": sorted(list(t) for t in tuples),
        "class_counts": {s: dict(sorted(counts[s].items())) for s in SLOTS},
    }


def convert(dataset: str, src_dir: Path, artifact_root: Path, splits: list[str]) -> dict:
    man_out = artifact_root / "manifests"
    man_out.mkdir(parents=True, exist_ok=True)
    report: dict = {"dataset": dataset, "artifact_root": str(artifact_root), "splits": {}}
    train_rows: list[dict] = []

    for split in splits:
        src = src_dir / f"{dataset}_{split}.jsonl"
        if not src.exists():
            print(f"  skip missing {src}")
            continue
        rows = [json.loads(l) for l in open(src)]
        v5 = [to_v5_row(r) for r in rows]
        name = SPLIT_ALIASES.get(split, split)
        dst = man_out / f"{name}.jsonl"
        with open(dst, "w") as f:
            for r in v5:
                f.write(json.dumps(r) + "\n")
        print(f"  {dst}: {len(v5)} rows")
        report["splits"][name] = len(v5)
        if split == "train":
            train_rows = rows

    if not train_rows:
        raise SystemExit("no train split found; grammar cannot be built")

    gram = build_grammar(train_rows)
    # grammar_path() looks for "{train_split}_grammar.json"
    gpath = man_out / "train_grammar.json"
    gpath.write_text(json.dumps(gram, indent=1))
    print(f"  {gpath}: {len(gram['tuples'])} valid tuples, "
          f"vocab { {s: len(v) for s, v in gram['slot_vocab'].items()} }")
    report["tuples"] = len(gram["tuples"])
    report["slot_vocab_sizes"] = {s: len(v) for s, v in gram["slot_vocab"].items()}
    (artifact_root / "audits").mkdir(parents=True, exist_ok=True)
    (artifact_root / "audits" / "external_adapter.json").write_text(json.dumps(report, indent=1))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True,
                    help="manifest prefix, e.g. assembly101_v4 or ikea_asm_dev3")
    ap.add_argument("--src_dir", default="artifacts/manifests")
    ap.add_argument("--artifact_root", required=True,
                    help="artifact root, e.g. artifacts/assembly101")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = ap.parse_args()
    convert(args.dataset, Path(args.src_dir), Path(args.artifact_root), args.splits)


if __name__ == "__main__":
    main()
