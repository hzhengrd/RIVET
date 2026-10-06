"""Stage-1 LoRA SFT and Stage-2 evidence-adapter training.

Baselines (video_only, rivet_sft, rivet_r_sft): standard LoRA SFT.
Adapter cells (*_e / *_e_res): load a parent LoRA and joint-train
EvidenceActionAdapter with a slot auxiliary loss and a reliability gate.
Residualization is optional.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import numpy as np
import torch.nn.functional as F

from .config import artifact_path, load_config, resolve_vlm
from .corruption import gate_target
from .experiment import complete_marker, run_dir, variant_spec
from .features import load_feature
from .grammar import load_train_grammar
from .adapter import AdapterHook, EvidenceActionAdapter, adapter_config, hidden_size
from .prompts import build_training_items, make_messages
from .utils import environment_snapshot, is_extended_clip, read_jsonl, set_seed, write_json


def encode(processor, cfg: dict[str, Any], item: dict[str, Any], spec: dict[str, Any], device: str):
    from qwen_vl_utils import process_vision_info
    answer = item["answer"]
    messages = make_messages(
        cfg, item["row"], spec, answer, task=item["task"], degraded=item["degraded"],
        corruption=item.get("corruption"),
    )
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    images, videos = process_vision_info(messages)
    batch = processor(text=[text], images=images, videos=videos, return_tensors="pt", padding=True).to(device)
    answer_ids = processor.tokenizer(answer, add_special_tokens=False)["input_ids"]
    labels = batch["input_ids"].clone()
    labels[:, :-len(answer_ids)] = -100
    batch["labels"] = labels
    return batch, len(answer_ids)


def _parent_splits(train_split: str) -> list[str]:
    """Orig-data runs must load a parent trained on the same orig split.

    `official_train_orig` must not silently fall back to the extended
    `official_train` parent. Oversampled `*_os` splits fall back to the
    non-oversampled orig parent of the same name.
    """
    if str(train_split).endswith("_orig"):
        return [train_split]
    splits = [train_split]
    if str(train_split).endswith("_os"):
        splits.append(str(train_split)[:-3])
    if "official_train" not in splits:
        splits.append("official_train")
    return splits


def _resolve_parent(cfg, spec, train_split, seed):
    parent = spec.get("init_variant")
    if not parent:
        return None
    # Prefer an in-tree parent; fall back to configured parent roots.
    last = None
    for split in _parent_splits(train_split):
        local = run_dir(cfg, split, parent, seed)
        last = local
        if complete_marker(local).exists():
            return local
    for root_key, marker in (
        ("parent_root", "parent_complete.json"),
        ("evidence_parent_root", "evidence_complete.json"),
    ):
        root = cfg.get("project", {}).get(root_key)
        if not root:
            continue
        from .config import repo_path
        for split in _parent_splits(train_split):
            candidate = repo_path(cfg, root) / "models" / split / parent / f"seed_{seed}"
            if (candidate / marker).exists():
                return candidate
    return last


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--matrix")
    parser.add_argument("--train_split", default="official_train")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--exclude_extended", action="store_true",
        help="Drop clip_ids containing '_extended' (also implied by *_orig train_split).",
    )
    parser.add_argument(
        "--stop_after_epoch", type=int, default=0,
        help="1-based: save a snapshot and exit after this epoch so eval can run "
             "before later epochs. 0 = run all epochs. complete.json is written "
             "only after the last planned epoch.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    spec = variant_spec(cfg, args.variant, args.matrix)
    if spec.get("stage", "sft") != "sft":
        raise ValueError(f"{args.variant} is not an SFT variant (use grpo module)")
    out = run_dir(cfg, args.train_split, args.variant, args.seed)
    if args.resume and not args.force and (out / "complete.json").exists():
        print(f"completed run exists; skipping {out}")
        return

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoProcessor, get_cosine_schedule_with_warmup
    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoVLM

    set_seed(args.seed)
    model_source = resolve_vlm(cfg)
    dtype = torch.bfloat16 if cfg["vlm"]["dtype"] == "bfloat16" else torch.float16
    processor = AutoProcessor.from_pretrained(model_source, trust_remote_code=True)
    model = AutoVLM.from_pretrained(model_source, trust_remote_code=True, torch_dtype=dtype,
                                    low_cpu_mem_usage=True).to(args.device)

    parent_dir = _resolve_parent(cfg, spec, args.train_split, args.seed) if spec.get("init_variant") else None
    freeze_vlm = bool(spec.get("freeze_vlm", False))
    if spec.get("init_variant"):
        if parent_dir is None or not complete_marker(parent_dir).exists():
            raise FileNotFoundError(
                f"{args.variant} requires parent {spec.get('init_variant')} at {parent_dir}"
            )
        model = PeftModel.from_pretrained(model, parent_dir, is_trainable=not freeze_vlm)
    else:
        lora = LoraConfig(
            r=int(cfg["vlm"]["lora_rank"]), lora_alpha=int(cfg["vlm"]["lora_alpha"]),
            lora_dropout=float(cfg["vlm"]["lora_dropout"]), target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    adapter = hook = None
    rapt_cfg = None
    if spec.get("rapt"):
        grammar = load_train_grammar(cfg, args.train_split)
        rapt_cfg = adapter_config(
            cfg,
            residualize=spec.get("residualize", False),
            utility_gate=spec.get("utility_gate", True),
            slot_aux=spec.get("slot_aux", True),
            bank_mode=spec.get("bank_mode", "evidence"),
            token_count=spec.get("token_count"),
        )
        # Infer feature dim from a sample cache or config default.
        feat_dim = int(cfg.get("features", {}).get("feature_dim", 0))
        if not feat_dim:
            feat_dir = artifact_path(cfg, "features")
            sample = next(feat_dir.glob("*.npy"), None) or next(feat_dir.glob("*.npz"), None)
            if sample is not None:
                feat_dim = int(np.load(sample).shape[-1])
            else:
                feat_dim = int(cfg.get("features", {}).get("fallback_dim", 96))
        adapter = EvidenceActionAdapter(
            hidden_size(model), feat_dim, rapt_cfg, slot_vocab=grammar.get("slot_vocab"),
        ).to(args.device, dtype=dtype)
        layer = int(spec.get("injection_layer", cfg["rapt"].get("injection_layer", 16)))
        hook = AdapterHook(model, adapter, layer)

    lora_params = [p for p in model.parameters() if p.requires_grad]
    adapter_params = [] if adapter is None else list(adapter.parameters())
    # Dual LR: lower for LoRA when jointly training with adapter.
    param_groups = []
    if lora_params:
        lr = float(cfg["sft"]["learning_rate"])
        # Reduced LoRA LR only when continuing a trained parent; from-scratch
        # joint training keeps the full SFT LR on LoRA.
        if spec.get("rapt") and not freeze_vlm and spec.get("init_variant"):
            lr = float(cfg["sft"].get("lora_lr", lr * 0.25))
        param_groups.append({"params": lora_params, "lr": lr})
    if adapter_params:
        param_groups.append({
            "params": adapter_params,
            "lr": float(cfg["sft"].get("adapter_lr", cfg["sft"]["learning_rate"])),
        })
    parameters = lora_params + adapter_params
    parameter_counts = {
        "vlm_total": sum(p.numel() for p in model.parameters()),
        "vlm_trainable": sum(p.numel() for p in lora_params),
        "adapter_trainable": sum(p.numel() for p in adapter_params),
    }
    optimizer = torch.optim.AdamW(param_groups, weight_decay=float(cfg["sft"]["weight_decay"]))

    rows = read_jsonl(artifact_path(cfg, "manifests", f"{args.train_split}.jsonl"))
    n_manifest = len(rows)
    drop_extended = args.exclude_extended or str(args.train_split).endswith("_orig")
    if drop_extended:
        rows = [row for row in rows if not is_extended_clip(row.get("clip_id", ""))]
    if args.limit:
        rows = rows[: args.limit]
    items = build_training_items(cfg, rows, spec, args.seed)
    out.mkdir(parents=True, exist_ok=True)

    grad_accum = int(cfg["sft"]["grad_accum"])
    epochs = int(cfg["sft"]["epochs"])
    total_updates = max(1, int(np.ceil(len(items) * epochs / grad_accum)))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_updates * float(cfg["sft"]["warmup_ratio"])), total_updates)
    state_path = out / "training_state.pt"
    start_step = 0
    if args.resume and state_path.exists():
        state = torch.load(state_path, map_location=args.device, weights_only=False)
        model.load_state_dict(state["model"], strict=False)
        if adapter is not None and state.get("adapter"):
            adapter.load_state_dict(state["adapter"], strict=False)
        optimizer.load_state_dict(state["optimizer"])
        if state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        start_step = int(state["step"])

    write_json(out / "run_config.json", {
        "config": cfg, "variant": spec, "train_split": args.train_split, "seed": args.seed,
        "stage": "sft", "n_items": len(items), "n_manifest": n_manifest, "n_rows": len(rows),
        "exclude_extended": drop_extended,
        "parent_dir": str(parent_dir) if parent_dir else None,
        "parameter_counts": parameter_counts,
        "environment": environment_snapshot(cfg["_repo_root"]),
    })

    step = 0
    optimizer.zero_grad()
    model.train()
    logs = []
    started = time.time()
    skipped = 0
    feature_hits = 0
    feature_misses = 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    def save_training_state(current_step: int) -> None:
        trainable = {name: value.detach().cpu() for name, value in model.state_dict().items()
                     if "lora_" in name}
        payload = {"model": trainable, "optimizer": optimizer.state_dict(),
                   "scheduler": scheduler.state_dict(), "step": current_step}
        if adapter is not None:
            payload["adapter"] = {k: v.detach().cpu() for k, v in adapter.state_dict().items()}
        torch.save(payload, state_path)

    def save_weights(dest) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(dest)
        processor.save_pretrained(dest)
        if adapter is not None:
            torch.save(adapter.state_dict(), dest / "v5_rapt_adapter.pt")
            write_json(dest / "v5_rapt_adapter_config.json", {
                "rapt": rapt_cfg, "feature_dim": int(adapter.learned_bank.shape[-1]),
                "layer": int(spec.get("injection_layer", cfg["rapt"].get("injection_layer", 16))),
                "d_model": hidden_size(model),
            })

    train_cfg = cfg.get("training", cfg["sft"])
    last_completed_epoch = 0
    for epoch in range(epochs):
        epoch_trained = False
        for item in items:
            step += 1
            if step <= start_step:
                continue
            epoch_trained = True
            try:
                batch, _n_answer = encode(processor, cfg, item, spec, args.device)
            except FileNotFoundError as exc:
                skipped += 1
                if skipped <= 10:
                    print(f"skip missing asset: {exc}", flush=True)
                continue

            if adapter is None:
                loss = model(**batch).loss
                diagnostics = {"task_loss": float(loss.detach())}
            else:
                corr = item.get("corruption") or {}
                mode = corr.get("mode")
                bank_clip = corr.get("evidence_clip_id") or item["row"]["clip_id"]
                try:
                    if mode == "absent":
                        adapter.set_grid(None)
                        adapter.bank_mode = "learned"
                    else:
                        adapter.bank_mode = spec.get("bank_mode", "evidence")
                        grid = load_feature(cfg, bank_clip, args.device, spec).to(dtype=dtype)
                        adapter.set_grid(grid)
                        feature_hits += 1
                except FileNotFoundError:
                    adapter.set_grid(None)
                    adapter.bank_mode = "learned"
                    feature_misses += 1
                    if feature_misses <= 10:
                        print(f"feature miss, using learned bank: {bank_clip}", flush=True)

                out_fwd = model(**batch)
                nll = out_fwd.loss
                current = dict(adapter.last)
                slot_loss = nll.new_zeros(())
                if item["task"] == "main" and rapt_cfg.get("slot_aux", True):
                    slot_loss = adapter.slot_aux_loss(item["row"]["gold_slots"])
                gate_loss = nll.new_zeros(())
                if rapt_cfg.get("utility_gate", True) and "gate_logit" in current:
                    target = torch.tensor(
                        [gate_target(mode if item["task"] == "main" else "correct")],
                        device=nll.device, dtype=current["gate_logit"].dtype,
                    )
                    gate_loss = F.binary_cross_entropy_with_logits(
                        current["gate_logit"], target.expand_as(current["gate_logit"]),
                    )
                residual_loss = nll.new_zeros(())
                if rapt_cfg.get("residualize", False) and "predicted" in current:
                    residual_loss = F.mse_loss(current["predicted"], current["action"].detach())
                    residual_loss = residual_loss + current.get("quantizer_loss", nll.new_zeros(()))
                loss = (
                    nll
                    + float(train_cfg.get("lambda_slot", 0.5)) * slot_loss
                    + float(train_cfg.get("lambda_gate", 0.25)) * gate_loss
                    + float(train_cfg.get("lambda_residual", 0.1)) * residual_loss
                )
                diagnostics = {
                    "task_loss": float(nll.detach()),
                    "slot_loss": float(slot_loss.detach()),
                    "gate_loss": float(gate_loss.detach()),
                    "residual_loss": float(residual_loss.detach()),
                    "gate": float(current["gate"].mean().detach()) if "gate" in current else None,
                }

            (loss / grad_accum).backward()
            if step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(parameters, float(cfg["sft"]["max_grad_norm"]))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            entry = {
                "step": step, "epoch": epoch, "task": item["task"], "degraded": item["degraded"],
                "corruption": (item.get("corruption") or {}).get("mode"),
                "clip_id": item["row"]["clip_id"], "loss": float(loss.detach()),
                "elapsed": time.time() - started, **diagnostics,
            }
            logs.append(entry)
            if step % int(cfg["sft"]["log_every"]) == 0:
                print(json.dumps(entry), flush=True)
                with (out / "metrics.jsonl").open("a") as handle:
                    for row in logs:
                        handle.write(json.dumps(row) + "\n")
                logs.clear()
            if step % int(cfg["sft"]["save_every"]) == 0:
                save_training_state(step)
            if args.max_steps and step - start_step >= args.max_steps:
                break
        else:
            last_completed_epoch = epoch + 1
            if epoch_trained:
                save_training_state(step)
                save_weights(out / f"epoch_{last_completed_epoch}")
                save_weights(out)
                print(f"saved epoch {last_completed_epoch}/{epochs} -> {out / f'epoch_{last_completed_epoch}'}",
                      flush=True)
            if args.stop_after_epoch and last_completed_epoch >= args.stop_after_epoch:
                break
            continue
        if args.max_steps and step - start_step >= args.max_steps:
            break

    if logs:
        with (out / "metrics.jsonl").open("a") as handle:
            for row in logs:
                handle.write(json.dumps(row) + "\n")
    save_weights(out)
    if hook is not None:
        hook.remove()
    write_complete = last_completed_epoch >= epochs
    if args.max_steps and not args.stop_after_epoch:
        write_complete = True
    if write_complete:
        write_json(out / "complete.json", {
            "stage": "sft", "steps": step, "skipped": skipped, "elapsed": time.time() - started,
            "feature_hits": feature_hits, "feature_misses": feature_misses,
            "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
            "epochs_completed": last_completed_epoch,
        })
        print(f"saved {out}")
    else:
        print(f"epoch {last_completed_epoch}/{epochs} checkpointed at {out}; "
              f"complete.json deferred until the last epoch", flush=True)


if __name__ == "__main__":
    main()
