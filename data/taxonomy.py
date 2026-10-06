"""HA-ViD action taxonomy: valid holistic actions, slot normalization, and the
valid-composition set used for constrained decoding / projection.

The "valid composition set" is the dataset's *action taxonomy* (the label space),
i.e. the set of holistic actions that exist in HA-ViD. Projecting a predicted
element tuple onto this set is legitimate constrained structured prediction. It is
NOT leakage as long as the set is the taxonomy / training label space and never
the individual test sample's ground truth.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from typing import Iterable

# canonical slot order used for the composite tuple (status handled separately)
ELEMENT_SLOTS = ["verb", "manipulated_object", "target_object", "tool"]
ALL_SLOTS = ["status"] + ELEMENT_SLOTS

# map prediction-side slot names -> canonical GT slot names
PRED_TO_CANON = {
    "status": "status",
    "action_verb": "verb",
    "verb": "verb",
    "manipulated_object": "manipulated_object",
    "target_object": "target_object",
    "tool": "tool",
    "holistic_action": "holistic",
}

_NA = {"not applicable", "n/a", "na", "none", "null", "", "no object", "no target"}
_NO_TOOL = {"no tool", "none", "no", "without tool", "n/a", "not applicable", ""}
_WS = re.compile(r"\s+")

# GT data-quality canonicalization (audit fix #2): one malformed target label that
# embeds a typo + leaks the tool into target_object.
GT_CANON = {
    "screw hole c1 uing a phillips screwdriver": "screw hole c1",
    "screw hole c1 using a phillips screwdriver": "screw hole c1",
}
# common description typos when parsing the full taxonomy
_DESC_TYPO = {"cyliner": "cylinder", "cyinder": "cylinder", "on to": "onto"}


def norm(s, *, slot: str | None = None) -> str:
    """Lowercase, collapse whitespace, canonicalize null/no-tool variants."""
    if s is None:
        s = ""
    s = _WS.sub(" ", str(s).strip().lower())
    s = s.rstrip(".")
    if s in GT_CANON:
        s = GT_CANON[s]
    if slot == "tool":
        if s in _NO_TOOL:
            return "no tool"
    if slot in ("manipulated_object", "target_object", "verb", "status"):
        if s in _NA:
            return "not applicable" if slot != "status" else "null"
    return s


def element_tuple(struct: dict) -> tuple:
    """(verb, manip, target, tool) canonicalized from a structured dict."""
    return tuple(norm(struct.get(s), slot=s) for s in ELEMENT_SLOTS)


@dataclass
class Taxonomy:
    """Valid holistic actions and their canonical element decomposition."""
    label_to_tuple: dict[str, tuple] = field(default_factory=dict)   # code -> elem tuple
    label_to_desc: dict[str, str] = field(default_factory=dict)      # code -> sentence
    tuple_to_label: dict[tuple, str] = field(default_factory=dict)
    slot_vocab: dict[str, set] = field(default_factory=dict)         # slot -> valid values

    @property
    def valid_tuples(self) -> list[tuple]:
        return list(self.tuple_to_label.keys())

    def is_valid(self, t: tuple) -> bool:
        return t in self.tuple_to_label


def build_taxonomy(ground_truth_paths: Iterable[str],
                   mapping_semantics_path: str | None = None) -> Taxonomy:
    """Build the taxonomy from one or more ground_truth.jsonl files (label ->
    canonical element tuple) and, optionally, a mapping_semantics.txt for the full
    label/description list. Using train GT is the leakage-free choice; eval GT is
    acceptable because we only use it to enumerate the *label space*, never to map
    an individual prediction to its own answer."""
    tax = Taxonomy()
    if mapping_semantics_path:
        try:
            with open(mapping_semantics_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    m = re.match(r'^(\S+)\s+"?(.*?)"?$', line)
                    if m:
                        tax.label_to_desc[m.group(1)] = m.group(2)
        except FileNotFoundError:
            pass

    for gt_path in ground_truth_paths:
        for line in open(gt_path):
            rec = json.loads(line)
            struct = rec.get("answers", {}).get("structured", {})
            if not struct:
                continue
            label = rec.get("label")
            t = element_tuple(struct)
            if label is not None and label not in tax.label_to_tuple:
                tax.label_to_tuple[label] = t
            tax.tuple_to_label.setdefault(t, label)
            tax.label_to_desc.setdefault(label, norm(rec.get("description")))
            for s in ALL_SLOTS:
                tax.slot_vocab.setdefault(s, set()).add(norm(struct.get(s), slot=s))

    # audit fix #3: extend the valid-composition set to the FULL taxonomy, not just
    # eval-present actions. Parse mapping_semantics descriptions for any label not
    # already decomposed from GT. Descriptions carry no tool -> infer "no tool"
    # (the missing actions are all insert/place, which use no tool).
    _verb_re = re.compile(r"^(insert|place|screw|slide|rotate|push|pull) the (.+?) "
                          r"(?:into|onto|to|on|in) (?:the )?(.+)$")
    for label, desc in tax.label_to_desc.items():
        if label in tax.label_to_tuple:
            continue
        d = desc
        for a, b in _DESC_TYPO.items():
            d = d.replace(a, b)
        m = _verb_re.match(d)
        if not m:
            continue
        t = (norm(m.group(1), slot="verb"),
             norm(m.group(2), slot="manipulated_object"),
             norm(m.group(3), slot="target_object"),
             "no tool")
        tax.label_to_tuple[label] = t
        tax.tuple_to_label.setdefault(t, label)
    return tax
