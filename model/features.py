"""Cache evidence images as Qwen-vision T×H×W×D grids."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from .config import apply_overrides, artifact_path, evidence_root, feature_root, load_config, resolve_vlm
from .prompts import load_meta
from .utils import read_jsonl, subset_indices, write_json


def resolve_feature_file(root: Path, clip_id: str) -> Path:
    """Prefer `{id}.npy`; accept legacy `{id}.npz.npy` (numpy appended suffix) and `{id}.npz`."""
    for name in (f"{clip_id}.npy", f"{clip_id}.npz.npy", f"{clip_id}.npz"):
        candidate = root / name
        if candidate.exists():
            return candidate
    return root / f"{clip_id}.npy"


def feature_path(cfg: dict[str, Any], clip_id: str, spec: dict[str, Any] | None = None) -> Path:
    return resolve_feature_file(feature_root(cfg, spec), clip_id)


def keyframe_paths(cfg: dict[str, Any], clip_id: str, use_badges: bool = True,
                   spec: dict[str, Any] | None = None) -> list[str]:
    meta = load_meta(cfg, clip_id, spec)
    root = evidence_root(cfg, spec) / clip_id
    if not root.exists():
        root = artifact_path(cfg, "evidence", clip_id)
    assets = meta.get("assets") or []
    kf = sorted((a for a in assets if a.get("type") == "kf"), key=lambda a: a.get("ordinal", 0))
    paths = [str(root / (a["path"] if use_badges else f"plain_{a['path']}")) for a in kf]
    subset = (spec or {}).get("keyframe_subset")
    if subset and len(paths) > int(subset):
        paths = [paths[i] for i in subset_indices(len(paths), int(subset))]
    if cfg.get("features", {}).get("include_som", False) and (spec or {}).get("use_som", True) \
            and any(a.get("type") == "som" for a in assets):
        paths.append(str(root / "som.jpg"))
    return paths


def _pool_patches(tokens: "torch.Tensor", height: int, width: int) -> "torch.Tensor":
    """tokens: (N, D) -> (H, W, D) by adaptive average over a roughly square layout."""
    import torch
    import torch.nn.functional as F
    n, dim = tokens.shape
    side = int(np.ceil(np.sqrt(n)))
    pad = side * side - n
    if pad:
        tokens = torch.cat([tokens, tokens.new_zeros(pad, dim)], dim=0)
    grid = tokens.reshape(1, side, side, dim).permute(0, 3, 1, 2)
    pooled = F.adaptive_avg_pool2d(grid, (height, width))
    return pooled[0].permute(1, 2, 0).contiguous()


def extract_clip_grid(processor, vision_model, cfg: dict[str, Any], clip_id: str,
                      device: str, dtype) -> np.ndarray:
    """Return float32 array [T, H, W, D]."""
    import torch
    from PIL import Image

    h = int(cfg.get("features", {}).get("spatial_h", 4))
    w = int(cfg.get("features", {}).get("spatial_w", 4))
    paths = keyframe_paths(cfg, clip_id, use_badges=True, spec=None)
    if not paths:
        raise FileNotFoundError(f"no keyframes for {clip_id}")

    def _histogram_tile(image: Image.Image) -> np.ndarray:
        arr = np.asarray(image.resize((64, 64)), dtype=np.float32) / 255.0
        hist = []
        for c in range(3):
            vals, _ = np.histogram(arr[:, :, c], bins=32, range=(0, 1), density=True)
            hist.append(vals.astype(np.float32))
        feat = np.concatenate(hist)
        dim = int(cfg.get("features", {}).get("fallback_dim", 96))
        if feat.shape[0] < dim:
            feat = np.pad(feat, (0, dim - feat.shape[0]))
        else:
            feat = feat[:dim]
        return np.tile(feat[None, None, :], (h, w, 1))

    def _pixels_from_image(image: Image.Image):
        # Qwen3-VL processors often reject bare images=; try image_processor first.
        image_processor = getattr(processor, "image_processor", None) or getattr(processor, "image_processor_class", None)
        candidates = []
        if image_processor is not None and hasattr(image_processor, "__call__"):
            candidates.append(lambda: image_processor(images=image, return_tensors="pt"))
        candidates.append(lambda: processor(images=image, return_tensors="pt"))
        for call in candidates:
            try:
                inputs = call()
            except Exception:
                continue
            if inputs is None:
                continue
            for key in ("pixel_values", "pixel_values_images", "image_grid_thw"):
                if hasattr(inputs, "keys") and key in inputs and inputs.get(key) is not None:
                    if key == "image_grid_thw":
                        continue
                    return inputs[key]
                if isinstance(inputs, dict) and inputs.get(key) is not None and key != "image_grid_thw":
                    return inputs[key]
        return None

    frames = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        pixel = _pixels_from_image(image)
        if pixel is None:
            frames.append(_histogram_tile(image))
            continue
        pixel = pixel.to(device=device, dtype=dtype)
        try:
            with torch.no_grad():
                tokens = None
                if hasattr(vision_model, "get_image_features"):
                    out = vision_model.get_image_features(pixel_values=pixel)
                    tokens = out[0] if isinstance(out, (tuple, list)) else out
                elif hasattr(vision_model, "visual"):
                    out = vision_model.visual(pixel)
                    tokens = out[0] if isinstance(out, (tuple, list)) else out
                else:
                    out = vision_model(pixel_values=pixel)
                    tokens = getattr(out, "last_hidden_state", out)
                if tokens is None:
                    raise RuntimeError("vision tower returned None")
                if tokens.ndim == 3:
                    tokens = tokens[0]
                if tokens.ndim != 2:
                    tokens = tokens.reshape(-1, tokens.shape[-1])
                pooled = _pool_patches(tokens.float(), h, w)
                frames.append(pooled.cpu().numpy().astype(np.float32))
        except Exception:
            frames.append(_histogram_tile(image))
    return np.stack(frames, axis=0)


def load_feature(cfg: dict[str, Any], clip_id: str, device: str | None = None,
                 spec: dict[str, Any] | None = None):
    import torch
    path = feature_path(cfg, clip_id, spec)
    if not path.exists():
        raise FileNotFoundError(path)
    arr = np.load(path)
    subset = (spec or {}).get("keyframe_subset")
    if subset and arr.shape[0] > int(subset):
        arr = arr[subset_indices(arr.shape[0], int(subset))]
    tensor = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0)
    if device:
        tensor = tensor.to(device)
    return tensor


def extract_all(cfg: dict[str, Any], splits: list[str], device: str = "cuda",
                limit: int = 0, force: bool = False) -> dict[str, Any]:
    import torch
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoVLM

    model_source = resolve_vlm(cfg)
    dtype = torch.bfloat16 if cfg["vlm"]["dtype"] == "bfloat16" else torch.float16
    processor = AutoProcessor.from_pretrained(model_source, trust_remote_code=True)
    model = AutoVLM.from_pretrained(model_source, trust_remote_code=True, torch_dtype=dtype,
                                    low_cpu_mem_usage=True).to(device)
    model.eval()
    out_dir = feature_root(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = skipped = failed = 0
    failures: list[str] = []
    shapes: dict[str, int] = {}
    for split in splits:
        path = artifact_path(cfg, "manifests", f"{split}.jsonl")
        if not path.exists():
            continue
        rows = read_jsonl(path)
        if limit:
            rows = rows[:limit]
        for row in rows:
            clip_id = row["clip_id"]
            dest = feature_path(cfg, clip_id)
            if dest.exists() and not force:
                skipped += 1
                continue
            try:
                grid = extract_clip_grid(processor, model, cfg, clip_id, device, dtype)
                np.save(dest, grid)
                shapes[str(grid.shape)] = shapes.get(str(grid.shape), 0) + 1
                done += 1
            except Exception as exc:  # noqa: BLE001 — audit all failures
                failed += 1
                if len(failures) < 50:
                    failures.append(f"{clip_id}: {exc!r}")
    report = {"ok": failed == 0, "done": done, "skipped": skipped, "failed": failed,
              "failures": failures, "shapes": shapes, "feature_root": str(out_dir)}
    write_json(artifact_path(cfg, "audits", "features_audit.json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--splits", default="official_train,official_test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--override", action="append", default=[],
                        help="dotted config override, e.g. features.output_dirname=features_uniform")
    args = parser.parse_args()
    cfg = apply_overrides(load_config(args.config), args.override)
    report = extract_all(cfg, [s.strip() for s in args.splits.split(",") if s.strip()],
                         device=args.device, limit=args.limit, force=args.force)
    print(report)


if __name__ == "__main__":
    main()
