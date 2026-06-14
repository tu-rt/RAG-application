from __future__ import annotations

import json
import random
import zipfile
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download


def load_longbench_multifieldqa(
    max_samples: int,
    seed: int = 42,
    cache_path: Path | None = None,
    hf_cache_dir: str | None = None,
) -> list[dict[str, Any]]:
    """加载 LongBench multifieldqa_zh，返回统一字段。

    通过 huggingface_hub 下载 data.zip 并手动解压，
    绕过 datasets 库的脚本加载和网络限制。
    """
    if cache_path and cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            rows = json.load(f)
        n_cached = len(rows)
        if n_cached < max_samples:
            print(
                f"[dataset] 警告: 缓存 {cache_path.name} 仅 {n_cached} 条，"
                f"少于 max_samples={max_samples}，将只评测 {n_cached} 条"
            )
        else:
            print(f"[dataset] 使用本地缓存 {cache_path}（{n_cached} 条）")
        return rows[:max_samples]

    # 用 hf_hub_download 下载 data.zip（自动走 HF_ENDPOINT 镜像）
    print("[dataset] 本地缓存不存在，下载 LongBench data.zip …")
    download_kwargs: dict = {
        "repo_id": "THUDM/LongBench",
        "filename": "data.zip",
        "repo_type": "dataset",
    }
    if hf_cache_dir:
        download_kwargs["cache_dir"] = hf_cache_dir
    zip_path = hf_hub_download(**download_kwargs)
    zip_path = Path(zip_path)

    # 解压到 data/longbench_raw/
    extract_dir = zip_path.parent / "longbench_extracted"
    if not extract_dir.exists():
        print(f"[dataset] 解压到 {extract_dir}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
    else:
        print(f"[dataset] 使用已解压目录 {extract_dir}")

    # 读取 multifieldqa_zh.jsonl
    jsonl_path = extract_dir / "multifieldqa_zh.jsonl"
    if not jsonl_path.exists():
        # 尝试 data 子目录
        jsonl_path = extract_dir / "data" / "multifieldqa_zh.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"找不到 multifieldqa_zh.jsonl，请检查 {extract_dir} 目录结构")

    all_items: list[dict[str, Any]] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            question = item.get("input") or item.get("question") or ""
            context = item.get("context") or ""
            answers = item.get("answers") or item.get("answer") or []
            if isinstance(answers, str):
                answers = [answers]
            all_items.append(
                {
                    "question": question.strip(),
                    "context": context.strip(),
                    "answers": [str(a).strip() for a in answers if str(a).strip()],
                }
            )

    # 随机采样
    rng = random.Random(seed)
    rng.shuffle(all_items)
    selected = all_items[:max_samples]

    rows: list[dict[str, Any]] = []
    for i, item in enumerate(selected):
        rows.append({"id": i, **item})

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"[dataset] 加载 {len(rows)} 条样本")
    return rows
