"""Resolve (and enforce) the runner a Stage-2 variant declares in the matrix.

The trainer has no notion of a runner: the adapter class is decided by which
module is invoked. Declaring `runner:` in the matrix and reading it back here
keeps the launch command and the experiment definition from drifting apart --
a mismatch would silently train a different model than the matrix documents.

    python scripts/stage2.py <config> <matrix> <variant>
        -> prints the module path to invoke, exits non-zero if unresolvable
"""
import sys

from model.config import load_config
from model.experiment import variant_spec

DEFAULT = "model.run_spatial"
ALLOWED = {"run_spatial", "run_informative", "run_hand_anchored"}

cfg_path, matrix, variant = sys.argv[1:4]
spec = variant_spec(load_config(cfg_path), variant, matrix)
runner = spec.get("runner")
if runner is None:
    print(DEFAULT)
elif runner in ALLOWED:
    print(f"model.{runner}")
else:
    sys.exit(f"FATAL: variant {variant!r} declares unknown runner {runner!r}")
