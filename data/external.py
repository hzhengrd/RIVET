"""Generic taxonomy + manifest builder for external assembly datasets
(Assembly101, IKEA ASM, ...), reusing the label/gold_slots/manifest contract
that data.common and data.taxonomy define for HA-ViD, but
built directly from each dataset's own (verb, noun) annotations instead of
HA-ViD's regex-from-description parsing. Slot vocabularies are dataset-native
and are never mapped onto HA-ViD's.
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from typing import Iterable, NamedTuple

from data.taxonomy import ELEMENT_SLOTS, norm


class ActionRow(NamedTuple):
    """One distinct action class as it appears in the source dataset."""
    label: str                 # short unique code, e.g. slugified "attach_wheel"
    verb: str
    manipulated_object: str
    target_object: str
    tool: str                  # "no tool" if the dataset has no tool slot
    description: str           # human-readable sentence, for holistic_desc


@dataclass
class ExternalTaxonomy:
    """Duck-type compatible with the HA-ViD taxonomy so scoring works
    unchanged on these datasets."""
    label_to_tuple: dict = field(default_factory=dict)
    label_to_desc: dict = field(default_factory=dict)
    tuple_to_label: dict = field(default_factory=dict)
    slot_vocab: dict = field(default_factory=dict)

    @property
    def valid_tuples(self) -> list:
        return list(self.tuple_to_label.keys())

    def is_valid(self, t: tuple) -> bool:
        return t in self.tuple_to_label


# HA-ViD registers its idle clips in the taxonomy (labels "null"/"w" -> the
# all-"not applicable" tuple), which is why its null clips score correctly.
# Datasets with an idle class must do the same or every idle clip counts as a
# taxonomy miss on holistic_direct / raw_valid / holistic_proj.
NULL_ROW = ActionRow(label="null", verb="null", manipulated_object="not applicable",
                     target_object="not applicable", tool="no tool", description="null")


def build_external_taxonomy(rows: Iterable[ActionRow], *, include_null: bool = False) -> ExternalTaxonomy:
    """Build an ExternalTaxonomy directly from labeled action rows (no
    description-regex parsing -- the caller already has verb/noun labels).
    Set include_null for datasets with an idle/background class."""
    tax = ExternalTaxonomy()
    rows = list(rows) + ([NULL_ROW] if include_null else [])
    for r in rows:
        t = (
            norm(r.verb, slot="verb"),
            norm(r.manipulated_object, slot="manipulated_object"),
            norm(r.target_object, slot="target_object"),
            norm(r.tool, slot="tool") or "no tool",
        )
        tax.label_to_tuple[r.label] = t
        tax.tuple_to_label.setdefault(t, r.label)
        tax.label_to_desc[r.label] = norm(r.description) or r.description
        for slot, val in zip(ELEMENT_SLOTS, t):
            tax.slot_vocab.setdefault(slot, set()).add(val)
    return tax


def load_grammar(path: str) -> ExternalTaxonomy:
    """Rebuild the taxonomy from a `*_grammar.json` written by the prep scripts.
    Needed by scoring: data.common.load_tax() is hardcoded to HA-ViD's
    ground truth, and scoring an external run against HA-ViD's label space makes
    every taxonomy-dependent metric (holistic_direct, holistic_proj, raw_valid)
    silently zero."""
    g = json.load(open(path))
    tax = ExternalTaxonomy()
    tax.label_to_tuple = {k: tuple(v) for k, v in g["label_to_tuple"].items()}
    tax.slot_vocab = {k: set(v) for k, v in g["slot_vocab"].items()}
    for label, t in tax.label_to_tuple.items():
        tax.tuple_to_label.setdefault(t, label)
    tax.label_to_desc = g.get("label_to_desc") or {k: k.replace("_", " ") for k in tax.label_to_tuple}
    return tax


def gold_slots(tax: ExternalTaxonomy, label: str) -> dict:
    """Mirrors data.common.gold_slots but against an
    ExternalTaxonomy. null/unknown labels map to the idle state."""
    if label == "null" or label not in tax.label_to_tuple:
        return {"status": "null", "verb": "null", "manipulated_object": "not applicable",
                "target_object": "not applicable", "tool": "no tool"}
    v, m, t, tool = tax.label_to_tuple[label]
    return {"status": "active", "verb": v, "manipulated_object": m,
            "target_object": t, "tool": tool}


def holistic_desc(tax: ExternalTaxonomy, label: str) -> str:
    return tax.label_to_desc.get(label, "null")


def clip_record(tax: ExternalTaxonomy, label: str, clip_id: str, video: str,
                 target_hand: str = "both") -> dict:
    """Builds one manifest row in the exact shape data.index
    writes for HA-ViD, so build_sft_data.py and downstream training code need
    no changes to consume it. target_hand defaults to "both": neither
    Assembly101 nor IKEA ASM labels a single active hand the way HA-ViD's
    lh_v0/rh_v0 crops do."""
    gs = gold_slots(tax, label)
    return {"clip_id": clip_id, "label": label, "video": video, "target_hand": target_hand,
            "status_null": gs["status"] == "null", "gold_slots": gs,
            "holistic": holistic_desc(tax, label)}


def write_manifest(path: str, rows: list[dict]) -> None:
    """Same JSONL contract as data.index._write."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n_active = sum(1 for r in rows if not r["status_null"])
    print(f"  {path}: {len(rows)} clips ({n_active} active, {len(rows) - n_active} null)")
