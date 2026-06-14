from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.generator import QwenGenerator
from src.retriever import InMemoryRetriever, Reranker, RetrievedChunk


@dataclass
class RAGResult:
    answer: str
    retrieved: list[RetrievedChunk]
    prompt: str
    rewritten_query: Optional[str] = None


def build_rag_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        context = "（无检索到相关段落）"
    else:
        blocks = []
        for i, c in enumerate(chunks, 1):
            blocks.append(f"[{i}] {c.text}")
        context = "\n\n".join(blocks)
    return (
        "你是一个严谨的中文问答助手。请仅根据下列参考段落回答问题；"
        "若段落中没有答案，请明确回答「根据给定资料无法确定」。\n\n"
        f"参考段落：\n{context}\n\n"
        f"问题：{question}\n\n"
        "请给出简洁准确的答案："
    )


def build_no_rag_prompt(question: str) -> str:
    return (
        "请直接回答下列问题。若你不确定，请回答「不确定」。\n\n"
        f"问题：{question}\n\n"
        "答案："
    )


def rewrite_query(question: str, generator: QwenGenerator, max_new_tokens: int = 64) -> str:
    prompt = (
        "将下面的问题改写为适合文档检索的简洁查询，只输出改写结果，不要解释。\n\n"
        f"原问题：{question}\n\n"
        "检索查询："
    )
    return generator.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.1)


class RAGPipeline:
    def __init__(
        self,
        embedder,
        generator: QwenGenerator,
        reranker: Optional[Reranker] = None,
        chunk_size: int = 512,
        overlap: int = 64,
        top_k: int = 5,
        rerank_top_k: int = 3,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
    ) -> None:
        self.embedder = embedder
        self.generator = generator
        self.reranker = reranker
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def run(
        self,
        document: str,
        question: str,
        mode: str = "R2",
    ) -> RAGResult:
        if mode == "R0":
            prompt = build_no_rag_prompt(question)
            ans = self.generator.generate(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
            return RAGResult(answer=ans, retrieved=[], prompt=prompt)

        retriever = InMemoryRetriever(self.embedder)
        retriever.build_index(document, self.chunk_size, self.overlap)

        query = question
        rewritten = None
        if mode == "R4":
            rewritten = rewrite_query(question, self.generator)
            query = rewritten or question

        retrieved = retriever.search(query, top_k=self.top_k)

        if mode == "R3" and self.reranker is not None and retrieved:
            retrieved = self.reranker.rerank(query, retrieved, top_k=self.rerank_top_k)

        prompt = build_rag_prompt(question, retrieved)
        ans = self.generator.generate(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        return RAGResult(
            answer=ans,
            retrieved=retrieved,
            prompt=prompt,
            rewritten_query=rewritten,
        )
