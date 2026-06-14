"""
批量评测 R0–R4，输出 results/metrics_summary.csv 与 details.jsonl

用法（在项目根目录）:
  python scripts/evaluate.py
  python scripts/evaluate.py --modes R0,R1,R2
  python scripts/evaluate.py --max_samples 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import data_dir, hf_cache_dir, load_config, results_dir
from src.dataset_loader import load_longbench_multifieldqa
from src.metrics import best_f1_over_refs, recall_at_k
from src.rag_pipeline import RAGPipeline
from src.runtime import (
    build_embedder,
    build_generator,
    build_reranker,
    print_runtime_banner,
)


def parse_modes(cfg: dict, override: str | None) -> list[str]:
    if override:
        return [m.strip() for m in override.split(",") if m.strip()]
    return list(cfg["experiments"]["run_ids"])


def run_experiment(
    mode: str,
    samples: list[dict],
    pipeline: RAGPipeline,
    chunk_size: int,
) -> tuple[pd.DataFrame, list[dict]]:
    records = []
    details = []

    for sample in tqdm(samples, desc=mode):
        sid = sample["id"]
        q = sample["question"]
        doc = sample["context"]
        refs = sample["answers"]

        if mode in ("R1", "R2", "R3", "R4"):
            pipeline.chunk_size = chunk_size
        result = pipeline.run(doc, q, mode=mode)

        retrieved_texts = [c.text for c in result.retrieved]
        hit = False
        if mode == "R0":
            hit = False
        else:
            hit = recall_at_k(refs, retrieved_texts, pipeline.top_k)

        f1 = best_f1_over_refs(result.answer, refs)
        row = {
            "id": sid,
            "mode": mode,
            "chunk_size": chunk_size if mode != "R0" else 0,
            "recall_at_k": int(hit),
            "f1": round(f1, 4),
            "question": q,
            "prediction": result.answer,
            "references": refs,
        }
        records.append(row)
        details.append(
            {
                **row,
                "retrieved": [{"id": c.chunk_id, "score": c.score, "text": c.text[:200]} for c in result.retrieved],
                "rewritten_query": result.rewritten_query,
            }
        )

    return pd.DataFrame(records), details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--modes", default=None, help="如 R0,R1,R2,R3")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    print_runtime_banner(cfg)
    out_dir = results_dir(cfg)
    cache = data_dir(cfg) / "multifieldqa_zh_subset.json"

    max_samples = args.max_samples or cfg["dataset"]["max_samples"]
    samples = load_longbench_multifieldqa(
        max_samples=max_samples,
        seed=cfg["dataset"]["seed"],
        cache_path=cache,
        hf_cache_dir=hf_cache_dir(cfg),
    )
    print(f"已加载 {len(samples)} 条样本（缓存: {cache}）")

    embedder = build_embedder(cfg)
    generator = build_generator(cfg)
    reranker = build_reranker(cfg)

    rag_cfg = cfg["rag"]
    pipeline = RAGPipeline(
        embedder=embedder,
        generator=generator,
        reranker=reranker,
        overlap=rag_cfg["chunk_overlap"],
        top_k=rag_cfg["top_k"],
        rerank_top_k=rag_cfg["rerank_top_k"],
        max_new_tokens=rag_cfg["max_new_tokens"],
        temperature=rag_cfg["temperature"],
    )

    modes = parse_modes(cfg, args.modes)
    chunk_map = {
        "R1": rag_cfg["chunk_sizes"][0] if rag_cfg["chunk_sizes"] else 256,
        "R2": rag_cfg["chunk_sizes"][-1] if rag_cfg["chunk_sizes"] else 512,
        "R3": rag_cfg["chunk_sizes"][-1] if rag_cfg["chunk_sizes"] else 512,
        "R4": rag_cfg["chunk_sizes"][-1] if rag_cfg["chunk_sizes"] else 512,
    }

    all_frames = []
    all_details = []

    for mode in modes:
        cs = chunk_map.get(mode, 512)
        df, details = run_experiment(mode, samples, pipeline, chunk_size=cs)
        all_frames.append(df)
        all_details.extend(details)

    summary = pd.concat(all_frames, ignore_index=True)
    agg = (
        summary.groupby("mode", as_index=False)
        .agg(
            recall_at_k_mean=("recall_at_k", "mean"),
            f1_mean=("f1", "mean"),
            n=("id", "count"),
        )
        .round(4)
    )

    summary_path = out_dir / "metrics_per_sample.csv"
    agg_path = out_dir / "metrics_summary.csv"
    details_path = out_dir / "details.jsonl"

    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    agg.to_csv(agg_path, index=False, encoding="utf-8-sig")
    with open(details_path, "w", encoding="utf-8") as f:
        for item in all_details:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\n=== 汇总 ===")
    print(agg.to_string(index=False))
    print(f"\n已保存:\n  {summary_path}\n  {agg_path}\n  {details_path}")


if __name__ == "__main__":
    main()
