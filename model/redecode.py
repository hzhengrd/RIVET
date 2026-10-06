"""Re-apply grammar decoding to existing prediction jsonl files.

Generation is greedy and independent of decoding, so this is equivalent to
re-running infer with a new --decoding flag, without loading the VLM.

    python -m model.redecode \
        --config configs/base.yaml \
        --decoding hard_id
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .config import load_config, output_path
from .grammar import apply_grammar, load_train_grammar, parse
from .utils import read_jsonl, write_jsonl


def _is_default_eval_file(path: Path) -> bool:
    stem = path.stem
    return stem == "official_test" or stem.startswith("official_test__bank_")


def redecode_rows(rows: list[dict], grammar: dict, mode: str) -> tuple[list[dict], int]:
    changed = 0
    out = []
    for row in rows:
        parsed, _strict = parse(row.get("raw") or "")
        slots, diag = apply_grammar(parsed, grammar, mode)
        item = dict(row)
        if item.get("prediction") != slots or item.get("decoding") != mode:
            changed += 1
        item["prediction"] = slots
        item["grammar"] = diag
        item["decoding"] = mode
        out.append(item)
    return out, changed


def process_file(path: Path, cfg: dict, mode: str, grammars: dict) -> str:
    rows = read_jsonl(path)
    if not rows:
        return "empty"
    current = rows[0].get("decoding")
    if current == mode and all(row.get("decoding") == mode for row in rows):
        return "already"
    train_split = rows[0].get("train_split") or "official_train"
    if train_split not in grammars:
        grammars[train_split] = load_train_grammar(cfg, train_split)
    new_rows, changed = redecode_rows(rows, grammars[train_split], mode)
    if not changed:
        return "already"
    if path.stem == "official_test" and current and current != mode:
        backup = path.with_name(f"{path.stem}__dec_{current}.jsonl")
        if not backup.exists():
            shutil.copy2(path, backup)
    write_jsonl(path, new_rows)
    return f"rewrote:{changed}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--decoding", default="hard_id")
    parser.add_argument("--drop_redundant_dec", action="store_true", default=True,
                        help="Remove official_test__dec_<mode>.jsonl once that mode is the default file.")
    args = parser.parse_args()

    for config_path in args.config:
        cfg = load_config(config_path)
        root = output_path(cfg, "predictions")
        if not root.exists():
            print(f"skip missing predictions: {root}")
            continue
        grammars: dict = {}
        files = sorted(p for p in root.glob("**/*.jsonl") if _is_default_eval_file(p))
        print(f"=== {config_path} ({len(files)} files under {root}) ===")
        for path in files:
            status = process_file(path, cfg, args.decoding, grammars)
            print(f"  {status:12s} {path.relative_to(root)}")
            if args.drop_redundant_dec and path.stem == "official_test":
                redundant = path.with_name(f"official_test__dec_{args.decoding}.jsonl")
                if redundant.exists():
                    redundant.unlink()
                    print(f"  dropped      {redundant.relative_to(root)}")


if __name__ == "__main__":
    main()
