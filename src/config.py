from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(ROOT)
    cfg["_config_path"] = str(cfg_path.resolve())
    return cfg


def results_dir(cfg: dict[str, Any]) -> Path:
    p = ROOT / cfg["project"]["results_dir"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def data_dir(cfg: dict[str, Any]) -> Path:
    p = ROOT / cfg["project"]["data_dir"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def hf_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("huggingface") or {}


def hf_cache_dir(cfg: dict[str, Any]) -> str | None:
    raw = hf_settings(cfg).get("cache_dir")
    if not raw:
        return None
    return str(Path(raw).expanduser().resolve())


def models_local_dir(cfg: dict[str, Any]) -> Path | None:
    raw = hf_settings(cfg).get("models_local_dir")
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def _is_local_model_dir(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").exists()


def resolve_model_path(name_or_path: str, cfg: dict[str, Any]) -> str:
    """解析模型路径：优先本地目录，否则返回 HuggingFace 仓库 ID。

    查找顺序：
    1. name_or_path 本身是含 config.json 的本地目录
    2. models_local_dir 下的常见子目录名（兼容后训练项目的 models/Qwen2.5-7B/ 布局）
    3. 原样返回（走 HF 远程或 ~/.cache/huggingface）
    """
    direct = Path(name_or_path).expanduser()
    if _is_local_model_dir(direct):
        return str(direct.resolve())

    local_root = models_local_dir(cfg)
    if local_root is None:
        return name_or_path

    short = name_or_path.split("/")[-1]
    candidates = [
        local_root / name_or_path,
        local_root / short,
        local_root / name_or_path.replace("/", "--"),
    ]
    # 兼容后训练项目 models/Qwen2.5-7B/ 命名（无 -Instruct 后缀）
    if short.endswith("-Instruct"):
        candidates.append(local_root / short.removesuffix("-Instruct"))
    for candidate in candidates:
        if _is_local_model_dir(candidate):
            return str(candidate.resolve())

    return name_or_path


def describe_model_source(name_or_path: str, cfg: dict[str, Any]) -> str:
    resolved = resolve_model_path(name_or_path, cfg)
    if Path(resolved).is_dir():
        return f"本地: {resolved}"
    cache = hf_cache_dir(cfg)
    if cache:
        return f"HF ID: {resolved} (cache_dir={cache})"
    return f"HF ID: {resolved} (默认 ~/.cache/huggingface)"
