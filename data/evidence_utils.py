from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SLOTS = ("status", "verb", "manipulated_object", "target_object", "tool")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        tmp = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp, path)


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        tmp = Path(handle.name)
        json.dump(obj, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while data := handle.read(chunk):
            digest.update(data)
    return digest.hexdigest()


def stable_int(text: str, modulo: int = 2**31 - 1) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16) % modulo


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def session_id(clip_id: str) -> str:
    return clip_id.split("_")[0]


def git_revision(root: str | Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def environment_snapshot(root: str | Path) -> dict[str, Any]:
    snap: dict[str, Any] = {
        "created_at": now_iso(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_revision": git_revision(root),
    }
    try:
        import torch
        snap.update(
            torch=torch.__version__, cuda=torch.version.cuda,
            cuda_available=torch.cuda.is_available(),
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        )
    except ImportError:
        snap["torch"] = None
    return snap


def structured_answer(slots: dict[str, str]) -> str:
    ordered = {
        "status": slots["status"],
        "action_verb": slots["verb"],
        "manipulated_object": slots["manipulated_object"],
        "target_object": slots["target_object"],
        "tool": slots["tool"],
    }
    return json.dumps(ordered, ensure_ascii=False)
