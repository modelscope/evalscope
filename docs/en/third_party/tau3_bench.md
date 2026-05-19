# τ³-bench

## Introduction

τ³-bench (Tau Cubed Bench) is the v1.0.0 release of the tau-bench family. It builds on τ²-bench with a new knowledge-retrieval domain (`banking_knowledge`), 75+ task quality fixes across the existing domains, and an audio-native evaluation mode.

- Project URL: https://github.com/sierra-research/tau2-bench (v1.0.0)
- The PyPI package name is still `tau2`. **τ³-bench cannot be installed in the same environment as τ²-bench (v0.2.0).** Pick one.

Core Features:
- Dynamic Interaction: Multi-turn user/agent simulation
- Tool Integration: Agents must use the provided API tools correctly
- Policy Adherence: Agents must follow domain policies
- Knowledge Retrieval (new): RAG pipeline with BM25, dense embeddings (OpenAI / Qwen), grep, sandbox shell, and rerankers
- Task Quality (new): 75+ corrected expected actions across airline / retail / banking

Supported Evaluation Domains:
- airline: Airline customer service
- retail: Retail customer service
- telecom: Telecom customer service
- banking_knowledge: Banking customer service backed by 698 policy/procedure documents (RAG)

## Installation

```bash
pip install evalscope
# Requires Python 3.12-3.13. Install tau3 with the [knowledge] extra so banking_knowledge works.
pip install "tau2[knowledge] @ git+https://github.com/sierra-research/tau2-bench@v1.0.0"

# Upstream v1.0.0 has eager voice imports even on the text-only path, so the
# following lightweight voice-related deps must be present (pyaudio needs
# `brew install portaudio` on macOS):
pip install pyaudio elevenlabs deepgram-sdk websockets jiwer pydub aiohttp scipy
```

```{important}
- The dataset is automatically pulled from ModelScope (Dataset ID: `evalscope/tau3-bench-data`), and TAU2_DATA_DIR is configured for you.
- Only supports evaluating the agent under test through OpenAI-compatible API services.
- τ³-bench and τ²-bench cannot coexist (same package name `tau2`).
```

## Usage

Taking qwen-plus as an example. The official leaderboard typically uses `user model = gpt-4.1-2025-04-14`.

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
                # banking_knowledge only; other domains ignore these:
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

## Retrieval Configs (banking_knowledge)

The `retrieval_config` extra param controls how the agent accesses the 698-document knowledge base. It is forwarded verbatim to `tau2.run.run_task(retrieval_config=...)`.

| Config | Tools given to agent | Extra requirements |
|--------|----------------------|--------------------|
| `no_knowledge` | none | none (offline) |
| `full_kb` | none (whole KB stuffed into prompt) | none |
| `golden_retrieval` | none (gold-labeled docs in prompt) | none |
| `grep_only` | `grep` | none |
| `bm25` *(default)* | `KB_search` (BM25 sparse retrieval) | `rank-bm25` (in `[knowledge]` extra) |
| `openai_embeddings` | `KB_search` (dense, OpenAI) | `OPENAI_API_KEY` |
| `qwen_embeddings` | `KB_search` (dense, OpenRouter/Qwen) | `OPENROUTER_API_KEY` |
| `*_reranker` | adds LLM reranker | also `OPENAI_API_KEY` |
| `*_grep` | adds `grep` tool | inherits base config |
| `terminal_use` / `terminal_use_write` | sandboxed `shell` | npm `@anthropic-ai/sandbox-runtime` + ripgrep (and on Linux: bwrap, socat) |
| `alltools` / `alltools-qwen` | BM25 + dense + shell | combination of above |

See the upstream README for the full matrix: https://github.com/sierra-research/tau2-bench/blob/v1.0.0/src/tau2/knowledge/README.md

## Tips

- If using gpt-4.1 as the user simulator, set:
  - `extra_params.user_model='gpt-4.1-2025-04-14'`
  - `extra_params.api_base='https://api.openai.com/v1'`
  - `extra_params.api_key=<OPENAI_API_KEY>`
- For airline/retail/telecom-only runs, drop `banking_knowledge` from `subset_list` to avoid pulling in retrieval dependencies.

## Evaluation Process

1. Task initialization with domain tools and policies
2. User simulation generates natural requests
3. Agent under test produces tool-augmented responses
4. Multi-turn loop until completion or failure
5. Reward computed by tau3's evaluator; mapped to `acc` in evalscope reports

## Metric Description

- Pass^1: proportion of tasks completed on the first attempt (higher is better)
- Aggregation: `mean_and_pass_hat_k` (supports `repeats > 1` for pass@k)
