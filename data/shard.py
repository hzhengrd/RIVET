"""Split manifests into shards so evidence can be built in parallel.

data.evidence has no sharding flag, but `records()` simply reads
`{artifact_root}/manifests/{split}.jsonl` for each name passed to `--splits`.
Writing `train_sh00.jsonl`, `train_sh01.jsonl`, ... therefore lets N processes
run concurrently against disjoint clip sets with no change to the builder.

At 5s/clip a single process needs ~118h for Assembly101's 84,284 clips; 12
shards bring that to ~10h.

Shards are contiguous slices, and `--resume` skips any clip whose meta.json
already exists, so re-running or overlapping shards is safe.

    python -m data.shard --root artifacts/assembly101/manifests \
        --splits train validation test --shards 12
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True, help="manifests directory")
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--shards", type=int, default=12)
    ap.add_argument("--prefix", default="sh")
    args = ap.parse_args()

    root = Path(args.root)
    names = []
    for split in args.splits:
        src = root / f"{split}.jsonl"
        if not src.exists():
            print(f"  skip missing {src}")
            continue
        rows = [l for l in open(src) if l.strip()]
        n = len(rows)
        per = -(-n // args.shards)          # ceil
        for i in range(args.shards):
            chunk = rows[i * per:(i + 1) * per]
            if not chunk:
                continue
            name = f"{split}_{args.prefix}{i:02d}"
            with open(root / f"{name}.jsonl", "w") as fh:
                fh.writelines(chunk)
            names.append((name, len(chunk)))
        print(f"  {split}: {n} rows -> {min(args.shards, -(-n//per))} shards of ~{per}")
    print("\nshard names:")
    print(",".join(n for n, _ in names))
    print(f"total rows across shards: {sum(c for _, c in names)}")


if __name__ == "__main__":
    main()
