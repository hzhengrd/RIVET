"""Run SpatialEvidenceAdapter with the auxiliary slot loss restricted to
informative slots (|V(s)| > 1), so datasets with one degenerate slot
(status on Assembly101, tool on IKEA ASM) do not dilute lambda_slot. The output
schema, prompt, grammar and reported metrics are unchanged.

    python -m model.run_informative train --config ... --variant ...
    python -m model.run_informative infer --config ... --variant ...
"""
from __future__ import annotations

import sys
from typing import Any

import torch
import torch.nn.functional as F

from .spatial import SpatialEvidenceAdapter
from .utils import SLOTS


class InformativeSlotAdapter(SpatialEvidenceAdapter):
    """SpatialEvidenceAdapter whose auxiliary loss ignores single-valued slots."""

    def __init__(self, d_model: int, feature_dim: int, cfg: dict[str, Any],
                 slot_vocab: dict[str, list[str]] | None = None):
        super().__init__(d_model, feature_dim, cfg, slot_vocab)
        self.informative = {s for s in SLOTS if len(self.slot_vocab.get(s, [])) > 1}
        dropped = sorted(set(self.slot_heads) - self.informative)
        if dropped:
            print(f"[run_v2s] auxiliary slot loss restricted to {sorted(self.informative)}; "
                  f"degenerate slots excluded: {dropped}", flush=True)

    def slot_aux_loss(self, gold_slots: dict[str, str]) -> torch.Tensor:
        if not self.slot_heads or "tokens" not in self.last:
            return torch.zeros((), device=next(self.parameters()).device)
        pooled = self.last["tokens"].mean(1)
        total = pooled.new_zeros(())
        count = 0
        for slot, head in self.slot_heads.items():
            if slot not in self.informative:
                continue
            vocab = self.slot_vocab.get(slot, [])
            value = gold_slots.get(slot, "")
            if not vocab or value not in vocab:
                continue
            target = torch.full((pooled.shape[0],), vocab.index(value),
                                device=pooled.device, dtype=torch.long)
            total = total + F.cross_entropy(head(pooled), target)
            count += 1
        return total / max(1, count)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "infer"):
        raise SystemExit("usage: python -m model.run_informative {train|infer} [args...]")
    mode = sys.argv.pop(1)
    if mode == "train":
        from . import train as target
    else:
        from . import infer as target
    target.AdapterClass = InformativeSlotAdapter
    print(f"[run_informative] {mode}: adapter = InformativeSlotAdapter")
    target.main()


if __name__ == "__main__":
    main()
