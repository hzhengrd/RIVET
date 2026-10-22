"""Stratified head-capped subsample of a training manifest.

Drops instances only from classes exceeding `--cap`; every class survives, and
classes below the cap keep every example. Selection is deterministic given
`--seed` and spread over distinct source sessions.

    python -m model.subsample \
        --manifest artifacts/assembly101/manifests/train.jsonl \
        --out artifacts/assembly101/manifests/train_cap20.jsonl --cap 20
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path


def session_of(row: dict) -> str:
    """Best-effort recording identifier, used to spread a capped class over
    sessions rather than taking the first N rows of one recording."""
    for key in ("session_id", "video_id", "recording"):
        if key in row:
            return str(row[key])
    return str(row.get("clip_id", "")).split("_")[0]


def stratified_cap(rows: list[dict], cap: int, seed: int) -> list[dict]:
    by_label: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)

    kept: list[dict] = []
    for label in sorted(by_label):
        items = by_label[label]
        if len(items) <= cap:
            kept.extend(items)
            continue
        rng = random.Random(f"{seed}#{label}")
        # round-robin over sessions so the cap does not collapse onto one recording
        buckets: dict[str, list[dict]] = collections.defaultdict(list)
        for r in items:
            buckets[session_of(r)].append(r)
        for b in buckets.values():
            rng.shuffle(b)
        order = sorted(buckets)
        rng.shuffle(order)
        picked: list[dict] = []
        while len(picked) < cap:
            progressed = False
            for s in order:
                if buckets[s]:
                    picked.append(buckets[s].pop())
                    progressed = True
                    if len(picked) == cap:
                        break
            if not progressed:
                break
        kept.extend(picked)
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=int, default=20)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.manifest)]
    before = collections.Counter(r["label"] for r in rows)
    kept = stratified_cap(rows, args.cap, args.seed)
    after = collections.Counter(r["label"] for r in kept)

    capped = [l for l in before if before[l] > args.cap]
    intact = [l for l in before if before[l] <= args.cap]
    tail = [l for l in before if before[l] < 30]
    print(f"  source     : {len(rows)} rows, {len(before)} classes")
    print(f"  subsample  : {len(kept)} rows, {len(after)} classes  (cap={args.cap})")
    print(f"  capped     : {len(capped)} classes lost {len(rows) - len(kept)} rows")
    print(f"  untouched  : {len(intact)} classes with <= cap instances keep every example")
    print(f"  tail (<30) : {len(tail)} classes, "
          f"{sum(before[l] for l in tail)} -> {sum(after[l] for l in tail)} rows")
    # The invariant is per-class, not per-stratum: a class at or below the cap is
    # carried over whole. Classes between the cap and 30 are trimmed to the cap,
    # which is intended -- they are still far denser than the true tail.
    assert len(after) == len(before), "a class was dropped -- this must never happen"
    assert all(after[l] == before[l] for l in intact), "a class at or below the cap was altered"
    assert all(after[l] == min(before[l], args.cap) for l in before), "cap not respected"
    if args.dry_run:
        return

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        for r in kept:
            fh.write(json.dumps(r) + "\n")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
