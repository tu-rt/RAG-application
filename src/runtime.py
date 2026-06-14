from __future__ import annotations

from typing import Any

from src.config import describe_model_source, hf_cache_dir, load_config, resolve_model_path
from src.embedding import EmbeddingModel
from src.generator import QwenGenerator
from src.retriever import Reranker


def print_runtime_banner(cfg: dict[str, Any]) -> None:
    """启动时打印关键配置，避免用错模型（对应后训练 #9）。"""
    mcfg = cfg["models"]
    ds = cfg.get("dataset") or {}
    hf = cfg.get("huggingface") or {}

    print("=" * 60)
    print(f"配置文件: {cfg.get('_config_path', 'config.yaml')}")
    print(f"项目: {cfg.get('project', {}).get('name', '')}")
    print("-" * 60)
    print(f"  LLM:       {describe_model_source(mcfg['llm'], cfg)}")
    print(f"             device={mcfg.get('llm_device', 'cuda')}  use_4bit={mcfg.get('use_4bit')}")
    if mcfg.get("llm_fallback"):
        print(f"  LLM 回退:  {describe_model_source(mcfg['llm_fallback'], cfg)}")
    print(f"  嵌入:      {describe_model_source(mcfg['embedding'], cfg)}")
    print(f"             device={mcfg.get('embedding_device', 'cpu')}")
    if mcfg.get("enable_reranker", True):
        print(f"  重排:      {describe_model_source(mcfg['reranker'], cfg)}")
        print(f"             device={mcfg.get('reranker_device', 'cpu')}")
    print(f"  评测样本:  max_samples={ds.get('max_samples')}  seed={ds.get('seed')}")
    if hf.get("models_local_dir"):
        print(f"  本地模型根: {hf['models_local_dir']}")
    if hf_cache_dir(cfg):
        print(f"  HF cache:  {hf_cache_dir(cfg)}")
    print("=" * 60)

    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU: {name} ({mem:.1f} GB)  CUDA {torch.version.cuda}")
        else:
            print("GPU: 不可用（将使用 CPU，LLM 会很慢）")
    except Exception as exc:
        print(f"GPU 检测跳过: {exc}")
    print()


def build_embedder(cfg: dict[str, Any]) -> EmbeddingModel:
    mcfg = cfg["models"]
    path = resolve_model_path(mcfg["embedding"], cfg)
    return EmbeddingModel(
        path,
        device=mcfg.get("embedding_device", "cpu"),
        cache_folder=hf_cache_dir(cfg),
    )


def build_generator(cfg: dict[str, Any]) -> QwenGenerator:
    mcfg = cfg["models"]
    return QwenGenerator(
        model_name=resolve_model_path(mcfg["llm"], cfg),
        device=mcfg.get("llm_device", "cuda"),
        use_4bit=mcfg.get("use_4bit", True),
        fallback_model=(
            resolve_model_path(mcfg["llm_fallback"], cfg)
            if mcfg.get("llm_fallback")
            else None
        ),
        fallback_use_4bit=mcfg.get("use_4bit_fallback", False),
        cache_dir=hf_cache_dir(cfg),
    )


def build_reranker(cfg: dict[str, Any]) -> Reranker | None:
    mcfg = cfg["models"]
    if not mcfg.get("enable_reranker", True):
        return None
    return Reranker(
        resolve_model_path(mcfg["reranker"], cfg),
        device=mcfg.get("reranker_device", "cpu"),
        cache_folder=hf_cache_dir(cfg),
    )
