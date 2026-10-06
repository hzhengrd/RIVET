from __future__ import annotations

import difflib
import json
import re
from pathlib import Path
from typing import Any

from .utils import SLOTS


def norm(value: Any, slot: str) -> str:
    value = re.sub(r"\s+", " ", str(value or "").strip().lower()).rstrip(".")
    if slot == "tool" and value in {"", "none", "n/a", "not applicable", "without tool"}:
        return "no tool"
    if slot in {"manipulated_object", "target_object"} and value in {"", "none", "n/a", "null"}:
        return "not applicable"
    return value


def parse(text: str) -> tuple[dict[str, str], bool]:
    raw: dict[str, Any] = {}
    strict = False
    match = re.search(r"\{.*?\}", text or "", re.S)
    if match:
        try:
            raw = json.loads(match.group())
            strict = isinstance(raw, dict)
        except json.JSONDecodeError:
            pass
    if not raw:
        pattern = re.compile(
            r'"?(status|action_verb|verb|manipulated_object|target_object|tool)"?\s*[:=]\s*"?([^",}\n]+)',
            re.I,
        )
        raw = {key.lower(): value for key, value in pattern.findall(text or "")}
    result = {
        "status": norm(raw.get("status") or "active", "status"),
        "verb": norm(raw.get("action_verb", raw.get("verb")), "verb"),
        "manipulated_object": norm(raw.get("manipulated_object"), "manipulated_object"),
        "target_object": norm(raw.get("target_object"), "target_object"),
        "tool": norm(raw.get("tool"), "tool"),
    }
    if result["status"] == "null":
        result.update(verb="null", manipulated_object="not applicable",
                      target_object="not applicable", tool="no tool")
    return result, strict


def _nearest(value: str, candidates: list[str]) -> str:
    if value in candidates:
        return value
    match = difflib.get_close_matches(value, candidates, n=1, cutoff=0.88)
    return match[0] if match else value


def load_grammar(grammar_path: Path | str) -> dict[str, Any]:
    return json.loads(Path(grammar_path).read_text())


def grammar_path(cfg: dict[str, Any], train_split: str):
    """Resolve `{train_split}_grammar.json` under manifests (v3 naming)."""
    from .config import artifact_path
    primary = artifact_path(cfg, "manifests", f"{train_split}_grammar.json")
    if primary.exists():
        return primary
    if train_split.endswith("_orig"):
        base = artifact_path(cfg, "manifests", f"{train_split[:-len('_orig')]}_grammar.json")
        if base.exists():
            return base
    if train_split.endswith("_os"):
        base = artifact_path(cfg, "manifests", f"{train_split[:-len('_os')]}_grammar.json")
        if base.exists():
            return base
    # Legacy alias used in early drafts.
    legacy = artifact_path(cfg, "manifests", "train_grammar.json")
    if legacy.exists():
        return legacy
    return primary


def load_train_grammar(cfg: dict[str, Any], train_split: str):
    return load_grammar(grammar_path(cfg, train_split))


def apply_grammar(parsed: dict[str, str], grammar: dict[str, Any] | Path, mode: str
                  ) -> tuple[dict[str, str], dict[str, Any]]:
    if mode == "free":
        return parsed, {"mode": mode}
    if isinstance(grammar, Path):
        grammar = load_grammar(grammar)
    result = dict(parsed)
    for slot in SLOTS:
        result[slot] = _nearest(result[slot], grammar["slot_vocab"].get(slot, []))
    tuple_value = tuple(result[slot] for slot in SLOTS[1:])
    valid = [tuple(item) for item in grammar["tuples"]]
    diagnostic: dict[str, Any] = {"mode": mode, "in_train_tuple": tuple_value in valid}
    if mode == "hard_id" and result["status"] != "null" and tuple_value not in valid:
        costs = [sum(a != b for a, b in zip(tuple_value, candidate)) for candidate in valid]
        chosen = valid[min(range(len(valid)), key=costs.__getitem__)]
        result.update(dict(zip(SLOTS[1:], chosen)))
        diagnostic["projected"] = True
    return result, diagnostic


def composition_validity(slots: dict[str, str], grammar: dict[str, Any]) -> dict[str, bool]:
    in_vocab = all(slots.get(slot) in grammar["slot_vocab"].get(slot, []) for slot in SLOTS)
    if slots.get("status") == "null":
        null_consistent = (slots.get("verb") == "null"
                           and slots.get("manipulated_object") == "not applicable"
                           and slots.get("target_object") == "not applicable"
                           and slots.get("tool") == "no tool")
    else:
        null_consistent = (slots.get("verb") not in {"", "null"}
                           and slots.get("manipulated_object") not in {"", "not applicable"})
    return {"in_vocab": in_vocab, "null_consistent": null_consistent,
            "valid": in_vocab and null_consistent}
