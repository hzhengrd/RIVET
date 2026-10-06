"""Inference, including harm-curve evidence modes and the evidence bank."""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from .config import artifact_path, load_config, resolve_vlm
from .corruption import CorruptionDraw, draw_corruption, harm_levels
from .experiment import prediction_path, run_dir, variant_spec
from .features import load_feature
from .grammar import apply_grammar, load_train_grammar, parse
from .adapter import AdapterHook, EvidenceActionAdapter, hidden_size
from .prompts import make_messages
from .utils import read_jsonl, sha256, write_jsonl


def training_split(eval_split: str) -> str:
    if eval_split.startswith("composition"):
        return "composition_train"
    if eval_split.startswith("primitive"):
        return "primitive_train"
    return "official_train"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--matrix")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", default="official_test")
    parser.add_argument("--train_split")
    parser.add_argument("--model_variant")
    parser.add_argument("--decoding")
    parser.add_argument("--evidence_mode", default="correct",
                        choices=["correct", "shuffled", "partial", "wrong", "absent"])
    parser.add_argument("--bank_mode", default="",
                        help="Override the evidence bank: evidence|learned|shuffled|absent")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model_dir", default="",
                        help="Load weights from this directory instead of the variant run_dir "
                             "(used for per-epoch snapshots).")
    parser.add_argument("--output_suffix", default="",
                        help="Appended to the prediction filename as __<suffix>, e.g. epoch_1")
    args = parser.parse_args()

    cfg = load_config(args.config)
    spec = variant_spec(cfg, args.variant, args.matrix)
    train_split = args.train_split or training_split(args.split)
    model_variant = args.model_variant or args.variant
    from pathlib import Path
    model_dir = Path(args.model_dir) if args.model_dir else run_dir(cfg, train_split, model_variant, args.seed)

    import torch
    from peft import PeftModel
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoVLM
    from qwen_vl_utils import process_vision_info

    model_source = resolve_vlm(cfg)
    dtype = torch.bfloat16 if cfg["vlm"]["dtype"] == "bfloat16" else torch.float16
    processor = AutoProcessor.from_pretrained(model_dir if (model_dir / "tokenizer_config.json").exists()
                                              else model_source, trust_remote_code=True)
    base = AutoVLM.from_pretrained(model_source, trust_remote_code=True, torch_dtype=dtype,
                                   low_cpu_mem_usage=True).to(args.device)
    model = PeftModel.from_pretrained(base, model_dir).to(args.device)
    model.eval()

    hook = adapter = None
    adapter_cfg_path = model_dir / "v5_rapt_adapter_config.json"
    adapter_weight = model_dir / "v5_rapt_adapter.pt"
    if spec.get("rapt") and adapter_cfg_path.exists() and adapter_weight.exists():
        acfg = json.loads(adapter_cfg_path.read_text())
        rapt = dict(acfg["rapt"])
        if args.bank_mode:
            rapt["bank_mode"] = args.bank_mode
        grammar_tmp = load_train_grammar(cfg, train_split)
        adapter = EvidenceActionAdapter(
            acfg["d_model"], acfg["feature_dim"], rapt,
            slot_vocab=grammar_tmp.get("slot_vocab"),
        ).to(args.device, dtype=dtype)
        adapter.load_state_dict(torch.load(adapter_weight, map_location=args.device, weights_only=True))
        hook = AdapterHook(model, adapter, int(acfg["layer"]))

    grammar = load_train_grammar(cfg, train_split)
    rows = read_jsonl(artifact_path(cfg, "manifests", f"{args.split}.jsonl"))
    if args.limit:
        rows = rows[: args.limit]
    decoding = args.decoding or cfg["evaluation"]["decoding"]
    rng = np.random.default_rng(args.seed)

    predictions = []
    for index, row in enumerate(rows):
        corruption = None
        if args.evidence_mode != "correct" and spec.get("evidence", False):
            draw = draw_corruption(row, rows, index, rng, mode=args.evidence_mode,
                                   n_keyframes=int(cfg.get("evidence", {}).get("num_keyframes", 10)))
            corruption = {
                "mode": draw.mode,
                "evidence_clip_id": draw.evidence_clip_id,
                "shuffle_order": draw.shuffle_order,
                **draw.notes,
            }
        messages = make_messages(cfg, row, spec, answer=None, corruption=corruption, rng=rng)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        batch = processor(text=[text], images=images, videos=videos, return_tensors="pt",
                          padding=True).to(args.device)

        if adapter is not None:
            bank = args.bank_mode or (corruption or {}).get("mode") or "evidence"
            if bank == "absent":
                adapter.set_grid(None)
                adapter.bank_mode = "learned"
                adapter.gate_override = 0.0
            elif bank == "learned":
                adapter.set_grid(None)
                adapter.bank_mode = "learned"
                adapter.gate_override = None
            else:
                clip = (corruption or {}).get("evidence_clip_id") or row["clip_id"]
                try:
                    grid = load_feature(cfg, clip, args.device, spec).to(dtype=dtype)
                    adapter.set_grid(grid)
                    adapter.bank_mode = "shuffled" if bank == "shuffled" else "evidence"
                    adapter.gate_override = None
                except FileNotFoundError:
                    adapter.set_grid(None)
                    adapter.bank_mode = "learned"

        t0 = time.time()
        with torch.no_grad():
            generated = model.generate(
                **batch, max_new_tokens=int(cfg["vlm"]["max_new_tokens"]), do_sample=False,
            )
        latency = time.time() - t0
        new_tokens = generated[0, batch["input_ids"].shape[1]:]
        raw = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
        parsed, strict = parse(raw)
        slots, grammar_diag = apply_grammar(parsed, grammar, decoding)
        gate_val = None
        if adapter is not None and adapter.last.get("gate") is not None:
            gate_val = float(adapter.last["gate"].mean().detach().cpu())
        predictions.append({
            "clip_id": row["clip_id"],
            "gold": row["gold_slots"],
            "prediction": slots,
            "raw": raw,
            "strict_json": strict,
            "grammar": grammar_diag,
            "latency_s": latency,
            "evidence_mode": args.evidence_mode,
            "bank_mode": args.bank_mode or (adapter.bank_mode if adapter else None),
            "gate": gate_val,
            "variant": args.variant,
            "seed": args.seed,
            "train_split": train_split,
            "eval_split": args.split,
            "decoding": decoding,
            "split": args.split,
            "epoch": int(args.output_suffix.split("_", 1)[1]) if args.output_suffix.startswith("epoch_") else None,
        })

    out = prediction_path(cfg, train_split, args.variant, args.seed, args.split)
    if args.evidence_mode != "correct":
        out = out.with_name(out.stem + f"__{args.evidence_mode}.jsonl")
    if args.decoding and args.decoding != cfg["evaluation"]["decoding"]:
        out = out.with_name(out.stem + f"__dec_{args.decoding}.jsonl")
    if args.output_suffix:
        out = out.with_name(out.stem + f"__{args.output_suffix}.jsonl")
    write_jsonl(out, predictions)
    if hook is not None:
        hook.remove()
    print(f"wrote {len(predictions)} predictions -> {out}")


if __name__ == "__main__":
    main()
