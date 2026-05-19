# τ³-bench

## 简介

τ³-bench（Tau Cubed Bench）是 tau-bench 系列的 v1.0.0 版本。在 τ²-bench 基础上新增了知识检索领域 `banking_knowledge`、跨领域 75+ 任务质量修复，以及音频原生（voice）评测模式。

- 项目地址：https://github.com/sierra-research/tau2-bench （v1.0.0）
- PyPI 包名仍为 `tau2`。**τ³-bench 与 τ²-bench（v0.2.0）无法在同一环境共存**，请二选一。

核心特点：
- 动态交互：模拟真实用户与 AI 代理的多轮对话
- 工具集成：代理需要正确使用提供的 API 工具
- 策略遵循：代理需要遵循业务策略
- 知识检索（新增）：可插拔 RAG 流水线，支持 BM25、稠密向量（OpenAI / Qwen）、grep、沙箱 shell、reranker 等
- 任务质量（新增）：airline / retail / banking 等领域共 75+ 处 expected actions 修复

支持的评测领域：
- airline：航空公司客服
- retail：零售客服
- telecom：电信客服
- banking_knowledge：银行客服，配套 698 份政策/流程文档（RAG）

## 安装依赖

```bash
pip install evalscope
# 需要 Python 3.12-3.13。安装时请带上 [knowledge] extra 以支持 banking_knowledge 领域。
pip install "tau2[knowledge] @ git+https://github.com/sierra-research/tau2-bench@v1.0.0"

# 上游 v1.0.0 的 text-only 路径也会急加载 voice 模块，需要补装以下轻量 voice 依赖
# （macOS 上 pyaudio 需先 `brew install portaudio`）：
pip install pyaudio elevenlabs deepgram-sdk websockets jiwer pydub aiohttp scipy
```

```{important}
- 数据集由 evalscope 自动从 ModelScope 拉取（数据集 ID：`evalscope/tau3-bench-data`），并自动设置 TAU2_DATA_DIR。
- 仅支持通过 OpenAI 兼容 API 服务评测被测模型。
- τ³-bench 与 τ²-bench 无法共存（同一 PyPI 包名 `tau2`）。
```

## 使用方法

以 qwen-plus 为例。官方榜单通常使用 `user model = gpt-4.1-2025-04-14`。

```python
import os
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='qwen-plus',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    eval_type='openai_api',

    datasets=['tau3_bench'],
    dataset_args={
        'tau3_bench': {
            'subset_list': ['airline', 'retail', 'telecom', 'banking_knowledge'],
            'extra_params': {
                'user_model': 'qwen-plus',
                'api_key': os.getenv('DASHSCOPE_API_KEY'),
                'api_base': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
                'generation_config': {
                    'temperature': 0.7,
                },
                # 仅 banking_knowledge 使用，其他领域忽略：
                'retrieval_config': 'bm25',
                'retrieval_config_kwargs': {},
            }
        }
    },

    eval_batch_size=5,
    limit=5,
    generation_config={'temperature': 0.6},
)

run_task(task_cfg)
```

## 检索配置（banking_knowledge）

`retrieval_config` 通过 `extra_params` 传入，控制 agent 如何访问 698 份知识库文档。该字段会原样转发给 `tau2.run.run_task(retrieval_config=...)`。

| 配置 | agent 拿到的工具 | 额外依赖 |
|---|---|---|
| `no_knowledge` | 无 | 无（离线） |
| `full_kb` | 无（整个 KB 塞进 prompt） | 无 |
| `golden_retrieval` | 无（gold docs 塞进 prompt） | 无 |
| `grep_only` | `grep` | 无 |
| `bm25`（默认） | `KB_search`（BM25 稀疏检索） | `rank-bm25`（在 `[knowledge]` extra 内） |
| `openai_embeddings` | `KB_search`（OpenAI 稠密向量） | `OPENAI_API_KEY` |
| `qwen_embeddings` | `KB_search`（OpenRouter/Qwen 稠密向量） | `OPENROUTER_API_KEY` |
| `*_reranker` | 增加 LLM reranker | 还需 `OPENAI_API_KEY` |
| `*_grep` | 增加 `grep` 工具 | 同基础配置 |
| `terminal_use` / `terminal_use_write` | 沙箱 `shell` | npm `@anthropic-ai/sandbox-runtime` + ripgrep（Linux 还要 bwrap、socat） |
| `alltools` / `alltools-qwen` | BM25 + 稠密 + shell | 上述各项依赖叠加 |

完整矩阵见上游 README：https://github.com/sierra-research/tau2-bench/blob/v1.0.0/src/tau2/knowledge/README.md

## 提示

- 使用 gpt-4.1 作为用户模拟模型时设置：
  - `extra_params.user_model='gpt-4.1-2025-04-14'`
  - `extra_params.api_base='https://api.openai.com/v1'`
  - `extra_params.api_key=<OPENAI_API_KEY>`
- 仅评测 airline/retail/telecom 时，从 `subset_list` 移除 `banking_knowledge` 即可避免拉取检索相关依赖。

## 评测流程

1. 任务初始化：为代理提供领域工具与策略
2. 用户模拟：用户模型按场景产生请求
3. 代理响应：被测模型生成带工具调用的响应
4. 多轮交互直至任务完成或失败
5. 由 tau3 evaluator 计算 reward，映射为 evalscope 报告中的 `acc`

## 指标说明

- Pass^1：首次尝试即完成任务的比例（越高越好）
- Aggregation：`mean_and_pass_hat_k`（设置 `repeats > 1` 可计算 pass@k）
