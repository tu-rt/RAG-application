"""
Gradio 交互演示：粘贴长文档 + 提问

用法:
  python scripts/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.rag_pipeline import RAGPipeline
from src.runtime import build_embedder, build_generator, build_reranker, print_runtime_banner


def build_pipeline(cfg: dict) -> RAGPipeline:
    embedder = build_embedder(cfg)
    generator = build_generator(cfg)
    reranker = build_reranker(cfg)
    rag = cfg["rag"]
    return RAGPipeline(
        embedder=embedder,
        generator=generator,
        reranker=reranker,
        chunk_size=rag["chunk_sizes"][-1],
        overlap=rag["chunk_overlap"],
        top_k=rag["top_k"],
        rerank_top_k=rag["rerank_top_k"],
        max_new_tokens=rag["max_new_tokens"],
        temperature=rag["temperature"],
    )


def main() -> None:
    cfg = load_config()
    print_runtime_banner(cfg)
    pipeline = build_pipeline(cfg)

    def answer_fn(document: str, question: str, mode: str, chunk_size: int):
        if not document.strip() or not question.strip():
            return "请填写文档与问题。", ""
        pipeline.chunk_size = int(chunk_size)
        result = pipeline.run(document, question, mode=mode)
        refs = "\n\n".join(
            f"[{i+1}] (score={c.score:.3f})\n{c.text[:400]}"
            for i, c in enumerate(result.retrieved)
        )
        meta = ""
        if result.rewritten_query:
            meta += f"改写查询: {result.rewritten_query}\n\n"
        meta += f"检索片段:\n{refs or '（无）'}"
        return result.answer, meta

    with gr.Blocks(title="中文文档 RAG Demo") as demo:
        gr.Markdown(
            "## 中文文档 RAG 问答（低显存版）\n"
            "嵌入: CPU · 生成: Qwen2.5-1.5B (4bit)\n\n"
            "模式: R0=无检索 | R2=检索 | R3=检索+重排 | R4=查询改写+检索"
        )
        doc = gr.Textbox(label="文档", lines=12, placeholder="粘贴长文本…")
        q = gr.Textbox(label="问题", lines=2)
        mode = gr.Dropdown(
            choices=["R0", "R2", "R3", "R4"],
            value="R2",
            label="实验模式",
        )
        chunk_size = gr.Slider(
            128, 1024, value=512, step=64, label="chunk_size（字符）"
        )
        btn = gr.Button("生成回答", variant="primary")
        ans = gr.Textbox(label="回答")
        debug = gr.Textbox(label="检索详情", lines=10)

        btn.click(answer_fn, inputs=[doc, q, mode, chunk_size], outputs=[ans, debug])

    demo.launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
