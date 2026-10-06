from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open() as handle:
        cfg = yaml.safe_load(handle) or {}
    parent = cfg.pop("extends", None)
    if parent:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        cfg = _merge(load_config(parent_path), cfg)
    cfg["_config_path"] = str(path)
    cfg["_repo_root"] = str(find_repo_root(path.parent))
    return cfg


def find_repo_root(start: str | Path) -> Path:
    cur = Path(start).resolve()
    for candidate in (cur, *cur.parents):
        if (candidate / "model").is_dir() and (candidate / "configs").is_dir():
            return candidate
    return Path.cwd().resolve()


def repo_path(cfg: dict[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(cfg["_repo_root"]) / path


def artifact_path(cfg: dict[str, Any], *parts: str) -> Path:
    return repo_path(cfg, cfg["project"]["artifact_root"]).joinpath(*parts)


def output_path(cfg: dict[str, Any], *parts: str) -> Path:
    return repo_path(cfg, cfg["project"]["output_root"]).joinpath(*parts)


def resolve_vlm(cfg: dict[str, Any]) -> str:
    local = repo_path(cfg, cfg["vlm"].get("local_dir", ""))
    if local.is_dir() and any((local / name).exists() for name in ("config.json", "adapter_config.json")):
        return str(local)
    return str(cfg["vlm"]["model"])
