"""Run training and inference with SpatialEvidenceAdapter (model/spatial.py).

The stock trainer and inferencer are left untouched: this rebinds the
`EvidenceActionAdapter` name inside their module namespaces before calling
`main()`. Python resolves module globals at call time, so the swap takes effect
without editing a single line of the shipped code, and any run launched through
the original entry points is unaffected.

    python -m model.run_spatial train  --config ... --variant ... --seed 17 ...
    python -m model.run_spatial infer  --config ... --variant ... --seed 17 ...

Every other flag is passed straight through to the underlying module.
"""
from __future__ import annotations

import sys

from .spatial import SpatialEvidenceAdapter


def _patch(module) -> None:
    module.EvidenceActionAdapter = SpatialEvidenceAdapter


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "infer"):
        raise SystemExit("usage: python -m model.run_spatial {train|infer} [args...]")
    mode = sys.argv.pop(1)
    if mode == "train":
        from . import train as target
    else:
        from . import infer as target
    _patch(target)
    print(f"[run_spatial] {mode}: adapter = SpatialEvidenceAdapter")
    target.main()


if __name__ == "__main__":
    main()
