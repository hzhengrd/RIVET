"""Offline tests for the model package (no GPU)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from model.corruption import draw_corruption, gate_target
from model.experiment import BUILTINS, main_matrix_variants
from model.grammar import composition_validity, parse
from model.metrics import correctness
from model.adapter import EvidenceActionAdapter


def test_main_matrix_has_ten_variants():
    names = main_matrix_variants()
    assert len(names) == 10
    assert "rivet_sft" in names and "sft_e_res" in names and "r_grpo_e" in names


def test_combine_variants_toggle_residual():
    assert BUILTINS["sft_e"]["residualize"] is False
    assert BUILTINS["sft_e_res"]["residualize"] is True
    assert BUILTINS["r_sft_e"]["init_variant"] == "rivet_r_sft"
    assert BUILTINS["r_grpo_e"]["init_variant"] == "rivet_r_grpo"


def test_parse_and_exact_scoring():
    slots, strict = parse(
        '{"status":"null","action_verb":"x","manipulated_object":"y","target_object":"z","tool":"t"}'
    )
    assert slots["status"] == "null" and slots["verb"] == "null" and strict
    row = {"gold": slots, "prediction": slots, "strict_json": True}
    result = correctness(row)
    assert result["exact"] == 1


def test_gate_targets():
    assert gate_target("correct") == 1.0
    assert gate_target("wrong") == 0.0
    assert gate_target("absent") == 0.0


def test_adapter_identity_at_init_and_slot_aux():
    cfg = {
        "token_count": 4, "token_dim": 16, "phase_heads": 4, "residualize": False,
        "utility_gate": True, "slot_aux": True, "quantizer": "continuous",
        "token_dropout": 0.0, "bank_mode": "evidence",
    }
    vocab = {
        "status": ["active", "null"], "verb": ["pick up", "null"],
        "manipulated_object": ["bolt", "not applicable"],
        "target_object": ["table", "not applicable"], "tool": ["no tool"],
    }
    adapter = EvidenceActionAdapter(32, 24, cfg, slot_vocab=vocab)
    assert float(adapter.alpha.detach()) == 0.0
    adapter.set_grid(torch.randn(2, 3, 2, 2, 24))
    hidden = torch.randn(2, 5, 32, requires_grad=True)
    out = adapter(hidden)
    assert out.shape == hidden.shape
    # alpha=0 => exact identity
    assert torch.allclose(out, hidden)
    # Force a non-identity scale so gradients reach `hidden` through the residual path.
    with torch.no_grad():
        adapter.alpha.fill_(0.5)
    out2 = adapter(hidden)
    loss = out2.sum() + adapter.slot_aux_loss({
        "status": "active", "verb": "pick up", "manipulated_object": "bolt",
        "target_object": "table", "tool": "no tool",
    })
    loss.backward()
    assert hidden.grad is not None
    assert float(adapter.alpha.detach()) != 0.0


def test_residual_changes_tokens():
    cfg = {
        "token_count": 4, "token_dim": 16, "phase_heads": 4, "residualize": True,
        "utility_gate": False, "slot_aux": False, "quantizer": "continuous",
        "token_dropout": 0.0, "bank_mode": "evidence",
    }
    adapter = EvidenceActionAdapter(32, 24, cfg)
    with torch.no_grad():
        adapter.alpha.fill_(1.0)
    adapter.set_grid(torch.randn(1, 2, 2, 2, 24))
    hidden = torch.zeros(1, 4, 32)
    out = adapter(hidden)
    assert not torch.allclose(out, hidden)


def test_corruption_wrong_donor():
    import numpy as np
    rows = [{"clip_id": "a", "label": "l1"}, {"clip_id": "b", "label": "l2"}]
    draw = draw_corruption(rows[0], rows, 0, np.random.default_rng(0), mode="wrong")
    assert draw.evidence_clip_id == "b"


def test_is_extended_clip_and_orig_filter():
    from model.utils import is_extended_clip
    assert is_extended_clip("clip_extended_1")
    assert is_extended_clip("foo_extended")
    assert not is_extended_clip("clip_001")
    rows = [{"clip_id": "a"}, {"clip_id": "a_extended_0"}, {"clip_id": "b"}]
    kept = [r for r in rows if not is_extended_clip(r["clip_id"])]
    assert [r["clip_id"] for r in kept] == ["a", "b"]


def test_subset_indices_aligns_prompt_and_bank():
    from model.utils import subset_indices
    assert subset_indices(10, None) == list(range(10))
    assert subset_indices(10, 10) == list(range(10))
    assert subset_indices(10, 4) == [0, 3, 6, 9]


def test_resolve_feature_file_legacy_npz_npy(tmp_path):
    from model.features import resolve_feature_file
    clip = "clipA"
    legacy = tmp_path / f"{clip}.npz.npy"
    legacy.write_bytes(b"x")
    assert resolve_feature_file(tmp_path, clip) == legacy
    modern = tmp_path / f"{clip}.npy"
    modern.write_bytes(b"y")
    assert resolve_feature_file(tmp_path, clip) == modern


def test_apply_grammar_hard_id_projects_unseen_tuple():
    from model.grammar import apply_grammar
    grammar = {
        "slot_vocab": {
            "status": ["active", "null"], "verb": ["pick up", "insert"],
            "manipulated_object": ["bolt", "ball"],
            "target_object": ["table", "base"],
            "tool": ["no tool", "wrench"],
        },
        "tuples": [["pick up", "bolt", "table", "no tool"]],
    }
    parsed = {"status": "active", "verb": "insert", "manipulated_object": "bolt",
              "target_object": "table", "tool": "no tool"}
    free, _ = apply_grammar(parsed, grammar, "free")
    assert free["verb"] == "insert"
    hard, diag = apply_grammar(parsed, grammar, "hard_id")
    assert hard["verb"] == "pick up"
    assert diag.get("projected") is True


def test_meta_from_path_does_not_treat_decoding_as_evidence():
    from pathlib import Path
    from model.metrics import _meta_from_path
    meta = _meta_from_path(Path("outputs/havid_side_left/predictions/official_train_orig/"
                                "r_sft_e/seed_17/official_test__dec_hard_id.jsonl"))
    assert meta["eval_split"] == "official_test"
    assert meta["evidence_mode"] == "correct"
    assert meta["decoding"] == "hard_id"
    bank = _meta_from_path(Path("outputs/x/predictions/official_train_orig/"
                                "r_sft_e_L0/seed_17/official_test__bank_shuffled.jsonl"))
    assert bank["evidence_mode"] == "correct"
    assert bank["bank_mode"] == "shuffled"
    ep = _meta_from_path(Path("outputs/x/predictions/official_train_orig_os/"
                              "r_sft_e_L0_e4os/seed_17/official_test__epoch_2.jsonl"))
    assert ep["evidence_mode"] == "correct"
    assert ep["epoch"] == 2


def test_redecode_rows_from_raw():
    from model.redecode import redecode_rows
    grammar = {
        "slot_vocab": {
            "status": ["active", "null"], "verb": ["pick up", "insert"],
            "manipulated_object": ["bolt"], "target_object": ["table"], "tool": ["no tool"],
        },
        "tuples": [["pick up", "bolt", "table", "no tool"]],
    }
    rows = [{
        "raw": '{"status": "active", "action_verb": "insert", "manipulated_object": "bolt", '
               '"target_object": "table", "tool": "no tool"}',
        "prediction": {"status": "active", "verb": "insert", "manipulated_object": "bolt",
                       "target_object": "table", "tool": "no tool"},
        "decoding": "free",
    }]
    out, changed = redecode_rows(rows, grammar, "hard_id")
    assert changed == 1
    assert out[0]["decoding"] == "hard_id"
    assert out[0]["prediction"]["verb"] == "pick up"


def test_composition_validity_null():
    grammar = {
        "slot_vocab": {
            "status": ["active", "null"], "verb": ["null", "pick up"],
            "manipulated_object": ["not applicable", "bolt"],
            "target_object": ["not applicable", "table"],
            "tool": ["no tool"],
        },
        "tuples": [["pick up", "bolt", "table", "no tool"]],
    }
    slots = {"status": "null", "verb": "null", "manipulated_object": "not applicable",
             "target_object": "not applicable", "tool": "no tool"}
    assert composition_validity(slots, grammar)["valid"]
