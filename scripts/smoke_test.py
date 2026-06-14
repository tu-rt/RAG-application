"""最小冒烟：不下载完整 LongBench，用内置短文本验证 pipeline。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.rag_pipeline import RAGPipeline
from src.runtime import build_embedder, build_generator, print_runtime_banner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    print_runtime_banner(cfg)
    doc = (
        "中南大学位于湖南省长沙市，是教育部直属的全国重点大学。"
        "学校设有计算机学院、数学与统计学院等院系。"
        "长沙是湖南省省会，有岳麓山、橘子洲等著名景点。"
    ) * 3
    q = "中南大学在哪个城市？"

    embedder = build_embedder(cfg)
    generator = build_generator(cfg)
    rag = cfg["rag"]
    pipeline = RAGPipeline(
        embedder=embedder,
        generator=generator,
        chunk_size=256,
        overlap=rag["chunk_overlap"],
        top_k=3,
        max_new_tokens=128,
    )

    for mode in ("R0", "R2"):
        print(f"\n===== {mode} =====")
        r = pipeline.run(doc, q, mode=mode)
        print("回答:", r.answer)
        print("检索条数:", len(r.retrieved))

    print("\n冒烟完成。若上述有正常中文回答，可运行 scripts/evaluate.py")


if __name__ == "__main__":
    main()
