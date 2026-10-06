"""Coverage check for the view-1 pooled dual-hand splits."""
import collections, glob, json, os, sys
VIEW = sys.argv[1] if len(sys.argv) > 1 else "v1"
root = f"artifacts/havid_{VIEW}"
for split in ("train", "train_aug_only", "train_aug", "test_lh", "test_rh"):
    p = f"{root}/manifests/{split}.jsonl"
    if not os.path.exists(p):
        print(f"  {split}: MISSING"); continue
    rows = [json.loads(l) for l in open(p)]
    ev = sum(1 for r in rows if os.path.exists(f"{root}/evidence/{r['clip_id']}/meta.json"))
    ft = sum(1 for r in rows if glob.glob(f"{root}/features_vlm/{r['clip_id']}.np*"))
    bx = sum(1 for r in rows if os.path.exists(f"{root}/hand_boxes/{r['clip_id']}.json"))
    by = collections.Counter(r.get("hand_source", "?") for r in rows)
    flag = "OK " if ev == ft == len(rows) else "GAP"
    print(f"  {flag} {split:<15} {len(rows):>5} rows {dict(by)}  evidence={ev:<5} features={ft:<5} boxes={bx}")
g = f"{root}/manifests/train_grammar.json"
if os.path.exists(g):
    d = json.load(open(g))
    print(f"      grammar: {len(d['tuples'])} tuples")
