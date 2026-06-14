# RAG Application: 中文文档检索增强生成与评测

本项目基于 Qwen2.5 + BGE 向量检索 + Cross-Encoder 重排，搭建中文长文档 RAG 问答系统，并在 LongBench MultiFieldQA-Zh 子集上完成 R0-R4 五组消融实验，分析检索、分块、重排和查询改写对回答质量的影响。

## 实验结果速览

评测数据：LongBench MultiFieldQA-Zh 子集 50 条。

| 模式 | 配置 | Recall@5 | F1 | 结论 |
|------|------|----------|----|------|
| R0 | 无检索，LLM 直接回答 | 0% | 0.051 | 闭卷回答效果较弱 |
| R1 | chunk=256，向量检索 Top-5 | 30% | - | 小 chunk 召回不足 |
| R2 | chunk=512，向量检索 Top-5 | 38% | 0.145 | 当前最优配置 |
| R3 | R2 + Cross-Encoder 重排 | 未超过 R2 | 未超过 R2 | Top-5 内重排未带来额外收益 |
| R4 | 查询改写 + R2 检索 | 未超过 R2 | 未超过 R2 | 当前改写策略未提升效果 |

> 注：当前 R3 是在向量检索 Top-5 内进行重排，候选集合不变，因此主要观察上下文排序对生成质量的影响，不保证提升 Recall@5。若要测试 reranker 对 Recall@5 的提升，应先召回 Top-20，再重排取 Top-5。

## 技术栈

- **LLM**: Qwen2.5-0.5B / 1.5B / 7B-Instruct
- **向量嵌入**: BAAI/bge-small-zh-v1.5
- **重排模型**: BAAI/bge-reranker-base（Cross-Encoder）
- **评测数据集**: [LongBench / multifieldqa_zh](https://huggingface.co/datasets/THUDM/LongBench)
- **评测指标**: Recall@K、字符级 F1
- **演示界面**: Gradio

## 实验设计

| 模式 | 说明 | 消融维度 |
|------|------|----------|
| R0 | 无检索，LLM 直接作答 | 闭卷基线 |
| R1 | chunk=256，向量检索 Top-5 | 分块粒度 |
| R2 | chunk=512，向量检索 Top-5 | 分块粒度 |
| R3 | R2 + Cross-Encoder 重排 | 候选排序 |
| R4 | 查询改写 + R2 检索 | 查询优化 |

核心消融维度：**分块大小、重排序、查询改写对 RAG 效果的影响**。

## 关键实现

- **低显存运行方案**: local 配置下嵌入和重排运行在 CPU，仅 LLM 占用 GPU，4GB 显存可运行 1.5B 模型
- **服务器配置**: `config.server.yaml` 支持 4090 环境下使用 Qwen2.5-7B fp16 进行完整评测
- **单文档闭卷检索**: 适配 MultiFieldQA-Zh 设定，每条样本独立构建向量索引
- **查询改写消融**: 使用 LLM 将原问题改写为短查询，并分析其在当前配置下未带来额外收益的原因
- **多模式批量评测**: 自动运行 R0-R4，输出汇总 CSV、逐条结果 CSV 和检索详情 JSONL
- **Gradio Demo**: 可视化检索片段与生成答案，便于展示 RAG 流程

## 项目结构

```text
RAG-application/
├── config.yaml              # 本机 4GB 显存配置
├── config.server.yaml       # 实验室 4090 服务器配置
├── requirements.txt
├── requirements-rag-extra.txt
├── scripts/
│   ├── evaluate.py          # 批量评测 R0-R4
│   ├── app.py               # Gradio 交互演示
│   ├── smoke_test.py        # 快速冒烟测试
│   └── check_env.py         # 环境检查
├── src/
│   ├── rag_pipeline.py      # RAG 核心管线
│   ├── retriever.py         # 向量检索 + Cross-Encoder 重排
│   ├── embedding.py         # BGE 嵌入封装
│   ├── generator.py         # Qwen 生成封装
│   ├── chunking.py          # 文本分块
│   ├── metrics.py           # Recall@K / F1
│   ├── dataset_loader.py    # LongBench 数据加载
│   ├── config.py            # 配置解析
│   └── runtime.py           # 运行时设备管理
├── data/                    # 数据缓存，已 gitignore
└── results/                 # 评测结果，已 gitignore
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

首次运行会下载模型与数据集，耗时取决于网络环境。

### 完整评测

```bash
python scripts/evaluate.py
```

默认评测 50 条样本，输出到 `results/`：

- `metrics_summary.csv` — 各模式 Recall@K、F1 均值
- `metrics_per_sample.csv` — 逐条评测结果
- `details.jsonl` — 检索片段、分数、生成答案详情

### 使用服务器配置

```bash
python scripts/evaluate.py --config config.server.yaml
```

`config.server.yaml` 中的本地模型路径需要按实际服务器路径修改。

### 交互 Demo

```bash
python scripts/app.py
# 浏览器打开 http://127.0.0.1:7860
```

## 硬件适配

| 配置项 | Local (RTX 3060 4GB) | Lab (RTX 4090 24GB) |
|--------|----------------------|---------------------|
| 配置文件 | config.yaml | config.server.yaml |
| LLM | Qwen2.5-1.5B 4bit | Qwen2.5-7B fp16 |
| 嵌入模型 | bge-small-zh-v1.5 (CPU) | bge-small-zh-v1.5 (CUDA 可选) |
| 重排模型 | bge-reranker-base (CPU) | bge-reranker-base (CUDA 可选) |
| 样本数 | 1-20 | 50+ |

4GB 显存关键策略：
- 嵌入和重排默认在 CPU 运行
- LLM 使用 4bit 量化
- 0.5B 模型作为兼容性备选

## 指标说明

- **Recall@K**：若任一参考答案字符串出现在 Top-K 检索片段中，则记为命中。
- **F1**：生成答案与参考答案之间的字符级 F1，取多个参考答案中的最大值。

该评测是轻量自动评测，适合比较不同 RAG 配置的相对效果，不等同于完整人工评测。

## 主要结论

- 无检索基线在中文长文档问答中表现较弱，RAG 能显著提升可回答性。
- 在当前数据与模型配置下，chunk=512 优于 chunk=256，Recall@5 从 30% 提升至 38%。
- Cross-Encoder 重排和查询改写在当前 Top-5 设定下未超过纯向量检索配置。
- 后续可尝试 Top-20 召回 + rerank Top-5、bge-m3 嵌入、多查询改写和更细粒度 chunk 策略。
