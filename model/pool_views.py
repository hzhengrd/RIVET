"""Build the pooled left+right split for the frontal or overhead view.

model.pool hardcodes the side-view paths. This rebinds that table and calls
its main(), so clip-id prefixes, symlinked evidence and the pooled grammar
stay the same.

    python -m model.pool_views --view frontal
"""
from __future__ import annotations

import sys

_PLACE = {"v1": "frontal", "frontal": "frontal", "v2": "overhead", "overhead": "overhead"}
_SIDE = {"lh": "left", "rh": "right"}


def hands_for(view: str) -> dict:
    place = _PLACE[view]
    out = {}
    for hand, side in _SIDE.items():
        root = f"artifacts/havid_{place}_{side}"
        out[hand] = {
            "train": f"{root}/manifests/train.jsonl",
            "test": f"{root}/manifests/test.jsonl",
            "evidence": f"{root}/evidence",
            "features": f"{root}/features_vlm",
        }
    return out


HANDS_V1 = hands_for("frontal")


def main() -> None:
    from . import pool as target
    view = "frontal"
    if "--view" in sys.argv:
        i = sys.argv.index("--view")
        view = sys.argv[i + 1]
        del sys.argv[i:i + 2]
    place = _PLACE[view]
    target.HANDS = hands_for(view)
    if not any(a == "--out_root" for a in sys.argv):
        sys.argv += ["--out_root", f"artifacts/havid_{place}"]
    print(f"[pool] pooling {place} left+right -> artifacts/havid_{place}")
    target.main()


if __name__ == "__main__":
    main()
