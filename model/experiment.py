from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .config import output_path, repo_path

# Main matrix:
#   Baselines: video_only, rivet_sft, rivet_r_sft, rivet_r_grpo
#   Combine (parent × residual): sft_e, sft_e_res, r_sft_e, r_sft_e_res, r_grpo_e, r_grpo_e_res
#
# Adapter fields:
#   rapt: enable EvidenceActionAdapter
#   residualize: residual on/off
#   init_variant: parent LoRA to load
#   freeze_vlm: adapter-only (secondary ablation)
#   bank_mode: evidence | learned | shuffled

def _combine(parent: str, residual: bool, name: str, **extra: Any) -> dict[str, Any]:
    reliability = parent.startswith("rivet_r")
    return {
        "stage": "sft",
        "evidence": True,
        "use_badges": True,
        "use_som": True,
        "necessity": True,
        "degrade": True,
        "reliability_mix": reliability,
        "arbitration": reliability,
        "rapt": True,
        "residualize": residual,
        "slot_aux": True,
        "utility_gate": True,
        "bank_mode": "evidence",
        "init_variant": parent,
        "freeze_vlm": False,
        "baseline": parent,
        "name": name,
        **extra,
    }


BUILTINS: dict[str, dict[str, Any]] = {
    "video_only": {
        "stage": "sft", "evidence": False, "necessity": False, "degrade": False,
        "reliability_mix": False, "arbitration": False, "rapt": False, "baseline": None,
    },
    "rivet_sft": {
        "stage": "sft", "evidence": True, "use_badges": True, "use_som": True,
        "necessity": True, "degrade": True, "reliability_mix": False,
        "arbitration": False, "rapt": False, "baseline": "video_only",
    },
    "rivet_r_sft": {
        "stage": "sft", "evidence": True, "use_badges": True, "use_som": True,
        "necessity": True, "degrade": True, "reliability_mix": True,
        "arbitration": True, "rapt": False, "baseline": "video_only",
    },
    # 2 extra epochs on the orig rivet_r_sft LoRA, with tail oversampling.
    # Does not overwrite the 2-epoch parent.
    "rivet_r_sft_e4os": {
        "stage": "sft", "evidence": True, "use_badges": True, "use_som": True,
        "necessity": True, "degrade": True, "reliability_mix": True,
        "arbitration": True, "rapt": False, "init_variant": "rivet_r_sft",
        "baseline": "rivet_r_sft",
    },
    "rivet_r_grpo": {
        "stage": "grpo", "init_variant": "rivet_r_sft", "evidence": True,
        "use_badges": True, "use_som": True, "do_no_harm": True,
        "rapt": False, "baseline": "rivet_r_sft",
    },
    # Main combine matrix
    "sft_e": _combine("rivet_sft", False, "sft_e"),
    "sft_e_res": _combine("rivet_sft", True, "sft_e_res"),
    "r_sft_e": _combine("rivet_r_sft", False, "r_sft_e"),
    "r_sft_e_res": _combine("rivet_r_sft", True, "r_sft_e_res"),
    # Stage-2 L0 from the 4-epoch oversampled parent (not the 2-epoch symlink).
    "r_sft_e_L0_e4os": _combine(
        "rivet_r_sft_e4os", False, "r_sft_e_L0_e4os",
        injection_layer=0, injection_site="decoder_early",
    ),
    # From-scratch joint LoRA + L0 adapter (no parent). Control for two-stage.
    "r_sft_e_L0_joint": {
        "stage": "sft", "evidence": True, "use_badges": True, "use_som": True,
        "necessity": True, "degrade": True, "reliability_mix": True,
        "arbitration": True, "rapt": True, "residualize": False, "slot_aux": True,
        "utility_gate": True, "bank_mode": "evidence", "freeze_vlm": False,
        "injection_layer": 0, "injection_site": "decoder_early",
        "baseline": "rivet_r_sft_e4os", "name": "r_sft_e_L0_joint",
    },
    "r_grpo_e": _combine("rivet_r_grpo", False, "r_grpo_e"),
    "r_grpo_e_res": _combine("rivet_r_grpo", True, "r_grpo_e_res"),
}


def variant_spec(cfg: dict[str, Any], name: str, matrix: str | None = None) -> dict[str, Any]:
    if name in BUILTINS:
        return {"name": name, **BUILTINS[name]}
    matrix_path = repo_path(cfg, matrix or "configs/method.yaml")
    payload = yaml.safe_load(matrix_path.read_text())
    for item in payload["experiments"]:
        if item["name"] == name:
            return dict(item)
    raise KeyError(f"Unknown variant {name!r}")


def run_dir(cfg: dict[str, Any], train_split: str, variant: str, seed: int) -> Path:
    return output_path(cfg, "models", train_split, variant, f"seed_{seed}")


def prediction_path(cfg: dict[str, Any], train_split: str, variant: str, seed: int, split: str) -> Path:
    return output_path(cfg, "predictions", train_split, variant, f"seed_{seed}", f"{split}.jsonl")


def complete_marker(path: Path) -> Path:
    for name in ("complete.json", "parent_complete.json", "evidence_complete.json"):
        candidate = path / name
        if candidate.exists():
            return candidate
    return path / "complete.json"


def main_matrix_variants() -> list[str]:
    return [
        "video_only", "rivet_sft", "rivet_r_sft", "rivet_r_grpo",
        "sft_e", "sft_e_res", "r_sft_e", "r_sft_e_res", "r_grpo_e", "r_grpo_e_res",
    ]
