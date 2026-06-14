# RAG Application: 中文文档检索增强生成与评测

基于 Qwen2.5 + BGE 向量检索 + Cross-Encoder 重排的中文 RAG 问答系统，在 LongBench MultiFieldQA-Zh 数据集上完成 5 组消融实验（R0-R4），量化检索、分块策略和重排对生成质量的影响。

## 技术栈

- **LLM**: Qwen2.5-0.5B / 1.5B / 7B-Instruct（4bit 量化）
- **向量嵌入**: BAAI/bge-small-zh-v1.5（CPU 推理）
- **重排模型**: BAAI/bge-reranker-base（Cross-Encoder，CPU 推理）
- **评测数据集**: [LongBench / multifieldqa_zh](https://huggingface.co/datasets/THUDM/LongBench)
- **评测指标**: Recall@K、F1
- **演示界面**: Gradio

## 实验设计

| 模式 | 说明 | 消融维度 |
|------|------|---------|
| R0 | 无检索，LLM 直接作答（幻觉基线） | - |
| R1 | chunk=256，向量检索 Top-K | 分块粒度 |
| R2 | chunk=512，向量检索 Top-K | 分块粒度 |
| R3 | R2 + Cross-Encoder 重排 | 重排序 |
| R4 | 查询改写 + R2 检索 | 查询优化 |

核心消融维度：**分块大小、重排序、查询改写对 RAG 效果的影响**。

## 关键实现

- **低显存运行方案**: 嵌入和重排均运行在 CPU，仅 LLM 占用 GPU，4GB 显存即可完整运行 1.5B 模型
- **单文档闭卷检索**: 适配 MultiFieldQA-Zh 设定，每条样本独立构建向量索引
- **查询改写**: 利用 LLM 将自然问题改写为检索友好的短查询，提升召回率
- **多模式对比评测**: 自动批量运行 R0-R4，输出汇总 CSV 和逐条详情 JSONL
- **Gradio 交互 Demo**: 可视化检索片段与生成答案，直观展示 RAG 效果

## 项目结构

```
RAG-application/
├── config.yaml              # 模型与实验配置
├── config.server.yaml       # 实验室服务器配置
├── requirements.txt
├── requirements-rag-extra.txt  # RAG 额外依赖
├── scripts/
│   ├── evaluate.py          # 批量评测（R0-R4）
│   ├── app.py               # Gradio 交互演示
│   ├── smoke_test.py        # 快速冒烟测试
│   └── check_env.py         # 环境检查
├── src/
│   ├── rag_pipeline.py      # RAG 核心管线（5 种模式）
│   ├── retriever.py         # 向量检索 + Cross-Encoder 重排
│   ├── embedding.py         # BGE 嵌入封装
│   ├── generator.py         # Qwen 生成封装
│   ├── chunking.py          # 文本分块（可配置 overlap）
│   ├── metrics.py           # Recall@K / F1 计算逻辑
│   ├── dataset_loader.py    # LongBench 数据加载
│   ├── config.py            # 配置解析
│   └── runtime.py           # 运行时设备管理
├── data/                    # 数据集缓存（自动生成，已 gitignore）
└── results/                 # 评测结果（自动生成，已 gitignore）
```

## 快速开始

### 环境准备

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 国内用户建议设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com
```

需要 CUDA 版 PyTorch。

### 冒烟测试

```bash
python scripts/evaluate.py --max_samples 1 --modes R0,R2
```

1 条样本，约 5-15 分钟（含首次模型下载）。

### 完整评测

```bash
python scripts/evaluate.py
```

默认 50 条样本，输出到 `results/`：
- `metrics_summary.csv` — 各模式 Recall@K、F1 均值
- `metrics_per_sample.csv` — 逐条结果
- `details.jsonl` — 检索片段与生成答案详情

### 交互 Demo

```bash
python scripts/app.py
# 浏览器打开 http://127.0.0.1:7860
```

## 硬件适配

| 配置项 | Local (RTX 3060 4GB) | Lab (RTX 4090 24GB) |
|--------|----------------------|---------------------|
| LLM | Qwen2.5-1.5B 4bit | Qwen2.5-7B fp16 |
| 嵌入模型 | bge-small-zh-v1.5 (CPU) | bge-m3 (可选) |
| 重排模型 | bge-reranker-base (CPU) | bge-reranker-base (CPU) |
| 评测样本数 | 1-20 | 50+ |

4GB 显存关键策略：
- 嵌入和重排始终在 CPU 运行
- LLM 使用 4bit 量化（约 2GB 显存）
- 0.5B 模型作为兼容性备选（约 1GB 显存）
