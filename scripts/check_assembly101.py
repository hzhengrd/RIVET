"""Coverage check for the Assembly101 subsampled + augmented splits."""
import collections, glob, json, os
root = "artifacts/assembly101"
ev = set(os.listdir(f"{root}/evidence"))
ft = {f.split(".")[0] for f in os.listdir(f"{root}/features_vlm")}
for split in ("train_cap20", "train_aug_only", "train_aug", "validation_mon", "test"):
    p = f"{root}/manifests/{split}.jsonl"
    if not os.path.exists(p):
        print(f"  {split}: MISSING"); continue
    rows = [json.loads(l) for l in open(p)]
    vid = sum(1 for r in rows if os.path.exists(r["video"]))
    e = sum(1 for r in rows if r["clip_id"] in ev)
    f = sum(1 for r in rows if r["clip_id"] in ft)
    n = len(collections.Counter(r["label"] for r in rows))
    flag = "OK " if vid == e == f == len(rows) else "GAP"
    print(f"  {flag} {split:<16} {len(rows):>6} rows  {n:>4} classes  "
          f"videos={vid:<6} evidence={e:<6} features={f}")
g = f"{root}/manifests/train_grammar.json"
if os.path.exists(g):
    print(f"      grammar: {len(json.load(open(g))['tuples'])} tuples")
