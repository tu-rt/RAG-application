"""
服务器启动前环境检查（对应后训练项目防坑清单）

用法:
  python scripts/check_env.py
  python scripts/check_env.py --config config.server.yaml
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    data_dir,
    describe_model_source,
    hf_cache_dir,
    load_config,
    models_local_dir,
    resolve_model_path,
)

RAG_IMPORTS_REQUIRED = ("transformers", "sentence_transformers", "bitsandbytes")
RAG_IMPORTS_OPTIONAL = ("gradio",)


def _check(label: str, ok: bool, detail: str = "", *, required: bool = True) -> bool:
    if ok:
        mark = "OK"
    elif required:
        mark = "FAIL"
    else:
        mark = "WARN"
    msg = f"  [{mark}] {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return ok if required else True


def _local_model_ok(name_or_path: str, cfg: dict) -> tuple[bool, str]:
    resolved = resolve_model_path(name_or_path, cfg)
    p = Path(resolved)
    if p.is_dir():
        if (p / "config.json").exists():
            return True, str(p)
        return False, f"目录存在但缺少 config.json: {p}"
    return True, f"将在线/HF 缓存加载: {resolved}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = parser.parse_args()

    print("=" * 60)
    print("RAG 项目 — 服务器环境检查")
    print("=" * 60)

    all_ok = True

    # 1. nvidia-smi
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True,
            )
            all_ok &= _check("nvidia-smi", True, out.stdout.strip().split("\n")[0])
        except subprocess.CalledProcessError as exc:
            all_ok &= _check("nvidia-smi", False, str(exc))
    else:
        all_ok &= _check("nvidia-smi", False, "未找到命令")

    # 2. PyTorch CUDA
    try:
        import torch

        cuda_ok = torch.cuda.is_available()
        detail = f"torch {torch.__version__}, cuda {torch.version.cuda}" if cuda_ok else "CUDA 不可用"
        all_ok &= _check("PyTorch CUDA", cuda_ok, detail)
    except ImportError:
        all_ok &= _check("PyTorch", False, "未安装 torch")

    # 3. pip check（TRT 里其它项目如 ever-beta 的缺依赖不影响 RAG）
    if shutil.which("pip"):
        r = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
        pip_ok = r.returncode == 0
        detail = "无冲突" if pip_ok else (r.stdout + r.stderr).strip()[:200]
        _check(
            "pip check",
            pip_ok,
            detail if pip_ok else "与 RAG 无关时可忽略，确保 sentence-transformers 已装即可",
            required=False,
        )

    # 4. RAG 关键依赖
    for pkg in RAG_IMPORTS_REQUIRED:
        try:
            __import__(pkg)
            all_ok &= _check(f"import {pkg}", True)
        except ImportError:
            all_ok &= _check(
                f"import {pkg}",
                False,
                "pip install -r requirements-rag-extra.txt -i https://pypi.tuna.tsinghua.edu.cn/simple",
            )
    for pkg in RAG_IMPORTS_OPTIONAL:
        try:
            __import__(pkg)
            _check(f"import {pkg}", True)
        except ImportError:
            _check(
                f"import {pkg}",
                False,
                "仅 scripts/app.py 需要；评测可跳过",
                required=False,
            )

    # 5. 配置与模型路径
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        all_ok &= _check("配置文件", False, str(cfg_path))
        print("\n检查未通过。")
        sys.exit(1)

    cfg = load_config(cfg_path)
    all_ok &= _check("配置文件", True, str(cfg_path.resolve()))

    mcfg = cfg["models"]
    for key in ("llm", "embedding", "reranker"):
        if key == "reranker" and not mcfg.get("enable_reranker", True):
            continue
        ok, detail = _local_model_ok(mcfg[key], cfg)
        resolved = resolve_model_path(mcfg[key], cfg)
        is_local = Path(resolved).is_dir() and (Path(resolved) / "config.json").exists()

        if key == "llm" and not is_local:
            print(f"       ⚠ {key}: {describe_model_source(mcfg[key], cfg)}")
            root = models_local_dir(cfg)
            if root and root.is_dir():
                subs = [p.name for p in sorted(root.iterdir()) if p.is_dir()][:8]
                print(f"         models_local_dir 下现有目录: {subs or '（空）'}")
            print("         请将 models.llm 改为含 config.json 的绝对路径")
            all_ok &= _check(f"模型 {key}", False, detail)
        elif not is_local and "/" in mcfg[key] and not Path(mcfg[key]).expanduser().exists():
            print(f"       ⚠ {key}: {describe_model_source(mcfg[key], cfg)}（将首次联网下载）")
            all_ok &= _check(f"模型 {key}", ok, detail)
        else:
            all_ok &= _check(f"模型 {key}", ok, detail)

    # 6. 评测数据缓存
    cache = data_dir(cfg) / "multifieldqa_zh_subset.json"
    max_samples = cfg.get("dataset", {}).get("max_samples", 50)
    if cache.exists():
        import json

        with open(cache, encoding="utf-8") as f:
            n = len(json.load(f))
        if n < max_samples:
            all_ok &= _check(
                "评测数据缓存",
                True,
                f"{cache.name} 仅 {n} 条 < max_samples={max_samples}，将静默截断",
            )
            print("       ⚠ 增大 max_samples 前请重新生成或上传更大缓存")
        else:
            all_ok &= _check("评测数据缓存", True, f"{n} 条 >= max_samples={max_samples}")
    else:
        all_ok &= _check(
            "评测数据缓存",
            True,
            f"未找到 {cache.name}，首次 evaluate 需联网下载 LongBench",
        )

    if hf_cache_dir(cfg):
        _check("HF cache_dir", True, hf_cache_dir(cfg))

    print("=" * 60)
    if all_ok:
        print("检查通过。建议: python scripts/smoke_test.py")
    else:
        print("存在 FAIL 项，请先修复再运行评测。")
        sys.exit(1)


if __name__ == "__main__":
    main()
