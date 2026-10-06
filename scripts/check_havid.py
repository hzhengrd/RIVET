"""Coverage check for the view-1 splits: every manifest row must have a cropped
video, evidence and (once extracted) a vision feature file."""
import glob, json, os, sys

PLACE = {"v1": "frontal", "v2": "overhead"}.get(sys.argv[1] if len(sys.argv) > 1 else "frontal",
                                   sys.argv[1] if len(sys.argv) > 1 else "frontal")

for h in ("left", "right"):
    root = f"artifacts/havid_{PLACE}_{h}"
    for split in ("train", "test"):
        p = f"{root}/manifests/{split}.jsonl"
        if not os.path.exists(p):
            print(f"  {h}_{PLACE}/{split}: MISSING")
            continue
        rows = [json.loads(l) for l in open(p)]
        vid = sum(1 for r in rows if os.path.exists(r["video"]))
        ev = sum(1 for r in rows if os.path.exists(f"{root}/evidence/{r['clip_id']}/meta.json"))
        ft = sum(1 for r in rows if glob.glob(f"{root}/features_vlm/{r['clip_id']}.np*"))
        nul = sum(1 for r in rows if r["status_null"])
        flag = "OK " if vid == ev == len(rows) else "GAP"
        print(f"  {flag} {h}_{PLACE}/{split:<5} {len(rows):>5} rows ({len(rows)-nul} active, {nul} null)"
              f"  videos={vid:<5} evidence={ev:<5} features={ft}")
    g = f"{root}/manifests/train_grammar.json"
    if os.path.exists(g):
        d = json.load(open(g))
        print(f"      grammar: {len(d['tuples'])} tuples, "
              f"vocab {{{', '.join(f'{k}:{len(v)}' for k, v in d['slot_vocab'].items())}}}")
