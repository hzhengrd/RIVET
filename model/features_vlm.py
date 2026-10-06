"""Evidence feature extraction that actually reaches the Qwen3-VL vision tower.

Standalone replacement for model.features.extract_all. The original module
is left untouched: this writes to its own output dir (default
`features_vlm`), so existing artifacts and results are never overwritten.

WHY THIS EXISTS
---------------
model.features._pixels_from_image() returns only `pixel_values` and
deliberately skips `image_grid_thw`:

    for key in ("pixel_values", "pixel_values_images", "image_grid_thw"):
        ...
        if key == "image_grid_thw":
            continue

But Qwen3-VL's tower requires it -- `Qwen3VLVisionModel.forward(hidden_states,
grid_thw, ...)` has no default for grid_thw. Verified end-to-end against the
real 8B checkpoint on a real evidence keyframe:

    get_image_features(pixel_values=px)                 -> AttributeError:
        'NoneType' object has no attribute 'tolist'
    get_image_features(pixel_values=px, image_grid_thw=g) -> tokens (728, 1152)

The exception was swallowed by a bare `except Exception:` that falls back to
`_histogram_tile()`, and extract_all counted the fallback as `done`, so
features_audit.json reported `failed: 0, ok: true`. Every cached feature in
`artifacts/havid_side_left/features/` is therefore a 96-d colour histogram tiled across
the 4x4 grid (all 16 cells identical) rather than 1152-d vision features.

FIXES
-----
1. Carry `image_grid_thw` through and pass it to the tower.
2. Pool over the TRUE (h, w) from grid_thw instead of assuming a square patch
   layout -- the real grid is e.g. 26x28, which ceil(sqrt(728))=27 scrambles.
3. Fallbacks are counted, reported per-clip, and gate the audit's `ok` flag
   instead of passing silently.

    python -m model.features_vlm --config configs/base.yaml \
        --splits official_train_orig,official_test --device cuda
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import apply_overrides, artifact_path, feature_root, load_config, resolve_vlm
from .features import keyframe_paths, resolve_feature_file
from .utils import read_jsonl, write_json

DEFAULT_OUTPUT_DIRNAME = "features_vlm"


def pool_tokens(tokens, grid_h: int, grid_w: int, out_h: int, out_w: int):
    """(N, D) vision tokens laid out on grid_h x grid_w -> (out_h, out_w, D).

    Uses the real grid shape from image_grid_thw. Qwen3-VL merges 2x2 patches,
    so N is (grid_h*grid_w)/merge**2; the merge factor is derived from N rather
    than assumed."""
    import torch
    import torch.nn.functional as F

    n, dim = tokens.shape
    cells = grid_h * grid_w
    merge = 1
    if cells != n:
        ratio = cells / max(n, 1)
        m = int(round(ratio ** 0.5))
        if m >= 1 and (grid_h % m == 0) and (grid_w % m == 0) and (grid_h // m) * (grid_w // m) == n:
            merge = m
        else:
            raise ValueError(f"token count {n} incompatible with grid {grid_h}x{grid_w}")
    gh, gw = grid_h // merge, grid_w // merge
    grid = tokens.reshape(1, gh, gw, dim).permute(0, 3, 1, 2)
    pooled = F.adaptive_avg_pool2d(grid, (out_h, out_w))
    return pooled[0].permute(1, 2, 0).contiguous()


def encode_image(processor, model, image, device, dtype, out_h: int, out_w: int):
    """One PIL image -> (out_h, out_w, D) real vision features. Raises on failure;
    the caller decides what to do rather than silently degrading."""
    import torch

    image_processor = getattr(processor, "image_processor", None)
    inputs = (image_processor if image_processor is not None else processor)(
        images=image, return_tensors="pt")
    if "pixel_values" not in inputs or inputs["pixel_values"] is None:
        raise RuntimeError("processor returned no pixel_values")
    if "image_grid_thw" not in inputs or inputs["image_grid_thw"] is None:
        raise RuntimeError("processor returned no image_grid_thw (required by the Qwen3-VL tower)")

    pixel = inputs["pixel_values"].to(device=device, dtype=dtype)
    grid_thw = inputs["image_grid_thw"].to(device=device)
    with torch.no_grad():
        out = model.get_image_features(pixel_values=pixel, image_grid_thw=grid_thw)
    tokens = out[0] if isinstance(out, (tuple, list)) else getattr(out, "last_hidden_state", out)
    if tokens is None:
        raise RuntimeError("vision tower returned None")
    if hasattr(tokens, "ndim") and tokens.ndim == 3:
        tokens = tokens[0]
    _, gh, gw = (int(v) for v in grid_thw[0].tolist())
    pooled = pool_tokens(tokens.float(), gh, gw, out_h, out_w)
    return pooled.cpu().numpy().astype(np.float32)


def extract_clip(processor, model, cfg, clip_id, device, dtype):
    """[T, out_h, out_w, D] for one clip's evidence keyframes."""
    from PIL import Image

    out_h = int(cfg.get("features", {}).get("spatial_h", 4))
    out_w = int(cfg.get("features", {}).get("spatial_w", 4))
    paths = keyframe_paths(cfg, clip_id, use_badges=True, spec=None)
    if not paths:
        raise FileNotFoundError(f"no keyframes for {clip_id}")
    frames = [encode_image(processor, model, Image.open(p).convert("RGB"),
                           device, dtype, out_h, out_w) for p in paths]
    return np.stack(frames, axis=0)


def extract_all(cfg: dict[str, Any], splits: list[str], device: str = "cuda",
                limit: int = 0, force: bool = False, skip_missing: bool = False,
                poll_seconds: int = 0, poll_idle_rounds: int = 15) -> dict[str, Any]:
    import torch
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoVLM

    model_source = resolve_vlm(cfg)
    dtype = torch.bfloat16 if cfg["vlm"]["dtype"] == "bfloat16" else torch.float16
    processor = AutoProcessor.from_pretrained(model_source, trust_remote_code=True)
    model = AutoVLM.from_pretrained(model_source, trust_remote_code=True, dtype=dtype,
                                    low_cpu_mem_usage=True).to(device)
    model.eval()

    out_dir = feature_root(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"vision features -> {out_dir}  (model {model_source})")

    done = failed = 0
    failures: list[str] = []
    shapes: dict[str, int] = {}
    dims: set[int] = set()
    idle = 0
    round_idx = 0
    skipped = 0
    while True:
        round_idx += 1
        round_done = 0
        deferred = 0
        skipped = 0
        for split in splits:
            path = artifact_path(cfg, "manifests", f"{split}.jsonl")
            if not path.exists():
                print(f"  skip missing manifest {path}")
                continue
            rows = read_jsonl(path)
            if limit:
                rows = rows[:limit]
            print(f"  {split}: {len(rows)} clips  (round {round_idx})")
            for i, row in enumerate(rows, 1):
                clip_id = row["clip_id"]
                dest = resolve_feature_file(out_dir, clip_id)
                if dest.exists() and not force:
                    skipped += 1
                    continue
                try:
                    grid = extract_clip(processor, model, cfg, clip_id, device, dtype)
                    np.save(dest, grid)
                    shapes[str(grid.shape)] = shapes.get(str(grid.shape), 0) + 1
                    dims.add(int(grid.shape[-1]))
                    done += 1
                    round_done += 1
                except FileNotFoundError as exc:
                    if skip_missing:
                        deferred += 1
                    else:
                        failed += 1
                        if len(failures) < 50:
                            failures.append(f"{clip_id}: {exc!r}")
                except Exception as exc:  # noqa: BLE001 — audited, never silently degraded
                    failed += 1
                    if len(failures) < 50:
                        failures.append(f"{clip_id}: {exc!r}")
                if i % 200 == 0:
                    print(f"    {i}/{len(rows)}  done={done} deferred={deferred} "
                          f"failed={failed}", flush=True)

        report = {"ok": failed == 0 and deferred == 0, "done": done, "skipped": skipped,
                  "failed": failed, "deferred": deferred, "failures": failures,
                  "shapes": shapes, "feature_dims": sorted(dims),
                  "feature_root": str(out_dir), "extractor": "features_vlm",
                  "round": round_idx}
        write_json(artifact_path(cfg, "audits", "features_vlm_audit.json"), report)
        print(f"  round {round_idx}: +{round_done} written, deferred={deferred} "
              f"failed={failed}", flush=True)
        if deferred == 0 or poll_seconds <= 0:
            break
        if round_done == 0:
            idle += 1
            if idle >= poll_idle_rounds:
                print(f"  stopping poll after {idle} idle rounds "
                      f"({deferred} still missing evidence)", flush=True)
                break
        else:
            idle = 0
        print(f"  waiting {poll_seconds}s for more evidence...", flush=True)
        time.sleep(poll_seconds)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--splits", default="official_train_orig,official_test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output_dirname", default=DEFAULT_OUTPUT_DIRNAME,
                        help="feature dir under the artifact root; kept separate from "
                             "the existing 'features' so nothing is overwritten")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--skip_missing", action="store_true",
                        help="do not count missing evidence as a failure; used while "
                             "evidence is still being built")
    parser.add_argument("--poll_seconds", type=int, default=0,
                        help="keep the model loaded and retry missing clips every N seconds")
    parser.add_argument("--poll_idle_rounds", type=int, default=15,
                        help="stop polling after this many consecutive rounds with no new files")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    cfg.setdefault("features", {})["output_dirname"] = args.output_dirname
    report = extract_all(cfg, [s for s in args.splits.split(",") if s],
                         device=args.device, limit=args.limit, force=args.force,
                         skip_missing=args.skip_missing, poll_seconds=args.poll_seconds,
                         poll_idle_rounds=args.poll_idle_rounds)
    print(f"done={report['done']} skipped={report['skipped']} failed={report['failed']} "
          f"dims={report['feature_dims']} ok={report['ok']}")
    if report["failures"]:
        print("first failures:")
        for f in report["failures"][:5]:
            print("  ", f)


if __name__ == "__main__":
    main()
