# 中文文档 RAG 问答与检索评测（低显存版）

面向 **RTX 3060 Laptop 4GB** 的本机可跑项目，对应实习简历 **项目二：RAG 应用算法**。

## 功能

| 模式 | 说明 |
|------|------|
| **R0** | 无检索，仅 LLM 作答（幻觉对照） |
| **R1** | chunk=256，向量检索 Top-K |
| **R2** | chunk=512，向量检索 Top-K |
| **R3** | R2 + Cross-Encoder 重排（CPU） |
| **R4** | 查询改写 + R2 检索 |

- 数据集：[LongBench / multifieldqa_zh](https://huggingface.co/datasets/THUDM/LongBench)（自动下载，首次需联网）
- 嵌入：`BAAI/bge-small-zh-v1.5`（**CPU**，不占显存）
- 生成：`Qwen/Qwen2.5-1.5B-Instruct`（**4bit**，约 2GB 显存）
- 回退：`Qwen/Qwen2.5-0.5B-Instruct`（4bit 失败时）

## 环境准备（Windows）

```powershell
cd "C:\Users\hasee\Desktop\Cursor\RAG 应用算法"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**国内 HuggingFace 镜像（推荐）：**

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

需要已安装 **CUDA 版 PyTorch**（与显卡驱动匹配）。可访问 [pytorch.org](https://pytorch.org) 按 3060 + CUDA 版本安装。

### bitsandbytes 在 Windows 上失败？

1. 打开 `config.yaml`，设 `use_4bit: false`，并把 `llm` 改为 `Qwen/Qwen2.5-0.5B-Instruct`  
2. 或仅把 `llm` 改为 `0.5B`，保持 `use_4bit: false`（约 1GB 显存）

## 快速运行

### 1）冒烟（1 条样本，约 5–15 分钟，含首次下模型）

```powershell
python scripts\evaluate.py --max_samples 1 --modes R0,R2
```

### 2）正式评测（默认 20 条，可在 config.yaml 改 `max_samples`）

```powershell
python scripts\evaluate.py
# 或双击 run_evaluate.bat
```

结果输出到 `results/`：

- `metrics_summary.csv` — 各模式 Recall@K、F1 均值  
- `metrics_per_sample.csv` — 逐条结果  
- `details.jsonl` — 检索片段与回答详情  

### 3）交互 Demo

```powershell
python scripts\app.py
# 浏览器打开 http://127.0.0.1:7860
```

## 目录结构

```text
RAG 应用算法/
├── config.yaml          # 模型与实验配置
├── requirements.txt
├── scripts/
│   ├── evaluate.py      # 批量评测
│   └── app.py           # Gradio 演示
├── src/
│   ├── chunking.py
│   ├── embedding.py
│   ├── generator.py
│   ├── retriever.py
│   ├── rag_pipeline.py
│   ├── metrics.py
│   └── dataset_loader.py
├── data/                # 数据集缓存（自动生成）
└── results/             # 评测结果（自动生成）
```

## 显存策略（4GB）

1. **嵌入永远在 CPU**，避免与 LLM 同时占 GPU  
2. **生成用 1.5B + 4bit**，不要在本机跑 7B  
3. **重排在 CPU**（`bge-reranker-base`）  
4. 评测时 `--max_samples` 先小后大；正式写简历建议 **50 条**（本机约数小时）

## 实验室服务器（4090 + TRT 环境）

可与后训练项目 **共用 TRT 虚拟环境**（`torch 2.5.1+cu121` + `transformers 4.46.x`），只需补装 RAG 额外依赖：

```bash
conda activate TRT
pip install -r requirements-rag-extra.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip check
```

**7B 模型无需重复下载**：在后训练项目的 `models/` 目录已有完整权重时，复制 `config.server.yaml` 并修改：

```yaml
huggingface:
  models_local_dir: "/path/to/post-training-project/models"  # 后训练模型根目录
models:
  llm: "Qwen/Qwen2.5-7B-Instruct"   # 自动解析为 models/Qwen2.5-7B-Instruct
  use_4bit: false
```

**还需准备（7B 之外）：**

| 资源 | 大小 | 说明 |
|------|------|------|
| `BAAI/bge-small-zh-v1.5` | ~100 MB | 向量嵌入（必须） |
| `BAAI/bge-reranker-base` | ~400 MB | R3 重排（可设 `enable_reranker: false` 跳过） |
| `data/multifieldqa_zh_subset.json` | ~2 MB | 评测 50 条缓存，**建议 scp 上传** |

**标准启动流程：**

```bash
export HF_ENDPOINT=https://hf-mirror.com   # 仅下载嵌入/重排时需要
export CUDA_VISIBLE_DEVICES=0

python scripts/check_env.py --config config.server.yaml
python scripts/smoke_test.py --config config.server.yaml   # 若加了 --config 支持
python scripts/evaluate.py --config config.server.yaml --max_samples 1 --modes R0,R2
nohup python scripts/evaluate.py --config config.server.yaml > eval.log 2>&1 &
```

> 启动时会打印当前 LLM/嵌入/重排的实际加载路径，避免用错模型。

## 后续在实验室（4090）可升级

在 `config.yaml` 中可改为：

- `llm: Qwen/Qwen2.5-7B-Instruct`
- `embedding: BAAI/bge-m3`
- `max_samples: 50` 或更多  

代码无需改结构，仅改配置即可。

## 简历可写指标（跑完后填入）

- 在 LongBench MultiFieldQA-Zh **N 条**上对比 R0–R4  
- Recall@5 从 R0 → R2 提升 **x**  
- 平均 F1 提升 **y**  
- 完成 chunk 256 vs 512、有无 rerank 消融  

## 常见问题

**Q: CUDA out of memory**  
减小 `max_samples`；改用 0.5B；关闭其他占 GPU 程序。

**Q: 下载模型很慢**  
设置 `HF_ENDPOINT=https://hf-mirror.com`，或提前用 `huggingface-cli download` 下载到本地后在 config 填本地路径。

**Q: LongBench 加载报错**  
确认 `datasets` 版本 ≥ 2.20；或检查网络与镜像。
