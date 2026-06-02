# 自定义文本检索评测数据集

## 概述

当你需要在特定领域（如法律、医疗、金融）或私有数据上评估 Embedding 模型的检索能力时，可以构建自定义数据集进行评测。EvalScope 基于 MTEB 框架，支持通过简单的 JSONL 文件定义语料库、查询和相关性标注，快速完成端到端的检索评测。

## 第一步：准备数据

### 目录结构

创建一个目录，包含以下三个 JSONL 文件：

```
my_retrieval_data/
├── corpus.jsonl
├── queries.jsonl
└── qrels.jsonl
```

### corpus.jsonl

语料库文件，每行一个 JSON 对象，表示一篇待检索文档。

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `_id` | str | 是 | 文档唯一标识 |
| `text` | str | 是 | 文档正文内容 |
| `title` | str | 否 | 文档标题，用于辅助检索 |

```json
{"_id": "doc1", "text": "气候变化正在导致更极端的天气模式。", "title": "气候变化"}
{"_id": "doc2", "text": "今天股市大幅上涨，科技股领涨。"}
{"_id": "doc3", "text": "人工智能正在通过自动化任务和提供见解来改变各种行业。"}
```

### queries.jsonl

查询文件，每行一个 JSON 对象，表示一条检索查询。

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `_id` | str | 是 | 查询唯一标识 |
| `text` | str | 是 | 查询文本 |

```json
{"_id": "query1", "text": "气候变化的影响是什么？"}
{"_id": "query2", "text": "今天股市上涨的原因是什么？"}
{"_id": "query3", "text": "人工智能如何改变行业？"}
```

### qrels.jsonl

相关性标注文件，每行一个 JSON 对象，定义查询与文档之间的相关性。

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `query-id` | str | 是 | 对应 queries.jsonl 中的 `_id` |
| `corpus-id` | str | 是 | 对应 corpus.jsonl 中的 `_id` |
| `score` | int/float | 是 | 相关性分数 |

```json
{"query-id": "query1", "corpus-id": "doc1", "score": 1}
{"query-id": "query2", "corpus-id": "doc2", "score": 1}
{"query-id": "query3", "corpus-id": "doc3", "score": 1}
```

**score 含义说明：**

- **二值标注（0/1）**：`1` 表示相关，`0` 表示不相关。适用于大多数检索场景。
- **连续分数**：使用多级相关性（如 0、1、2、3），数值越大表示越相关。适用于需要区分"部分相关"和"高度相关"的精细评测。

### 数据质量建议

```{tip}
- 推荐语料库至少包含 **100** 篇文档，查询至少 **50** 条，以获得统计上有意义的评测结果。
- 确保每个查询至少有 **1** 个相关文档标注。
- 如果使用二值标注，建议同时标注负样本（score=0），以更准确地评估模型区分能力。
- 语料库中应包含足够多与查询无关的"干扰"文档，模拟真实检索场景。
```

## 第二步：配置并运行评测

### 完整配置示例

```python
from evalscope.run import run_task

task_cfg = {
    "work_dir": "outputs",
    "eval_backend": "RAGEval",
    "eval_config": {
        "tool": "MTEB",
        "models": [
            {
                "model_name_or_path": "AI-ModelScope/m3e-base",
                "pooling_mode": None,
                "max_seq_length": 512,
                "prompt": "",
                "model_kwargs": {"torch_dtype": "auto"},
                "encode_kwargs": {
                    "batch_size": 128,
                },
            }
        ],
        "eval": {
            "custom_tasks": [
                {
                    "name": "CustomRetrieval",
                    "data_path": "my_retrieval_data",
                }
            ],
            "overwrite_results": True,
            "limits": 500,
        },
    },
}

run_task(task_cfg=task_cfg)
```

### 运行命令

将上述代码保存为 `run_eval.py`，然后执行：

```bash
python run_eval.py
```

### 结果解读

评测完成后，结果保存在 `outputs/` 目录下，主要指标包括：

- **NDCG@k**：归一化折损累计增益，衡量排序质量（k=1,3,5,10）
- **MAP@k**：平均精度均值
- **Recall@k**：召回率
- **Precision@k**：精确率

分数范围为 0~1，越高表示检索效果越好。

## 参数参考

`custom_tasks` 列表中每个任务对应一个 `CustomTaskConfig`，可配置字段如下：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | str | `"CustomRetrieval"` | 任务名称，用于标识评测结果 |
| `data_path` | str | （必填） | 数据集目录路径，需包含 `corpus.jsonl`、`queries.jsonl`、`qrels.jsonl` |
| `eval_splits` | List[str] | `["test"]` | 评测分割列表 |

```{note}
其他评测参数（如 `models`、`limits`、`overwrite_results` 等）与默认配置一致，详见 [MTEB 评测参数说明](../../user_guides/backend/rageval_backend/mteb.md#参数说明)。
```

## 常见问题

### 字段格式错误

- 数据文件必须是 **JSONL** 格式（每行一个独立的 JSON 对象），不是标准 JSON 数组格式。
- `_id` 字段值必须是**字符串**类型，即使是数字 ID 也应写为 `"123"` 而非 `123`。
- `qrels.jsonl` 中的字段名带连字符：`query-id` 和 `corpus-id`，不是下划线。

### 数据量太少的影响

- 语料库文档过少（如少于 10 篇）时，Recall@10 等指标可能始终为 1.0，无法有效区分模型好坏。
- 查询过少会导致评测结果方差较大，不具备统计显著性。
- 建议根据实际业务场景，准备至少百条量级的数据。

### 从其他格式转换

如果已有 TSV 或 CSV 格式的数据，可以使用以下方式转换为 JSONL：

```python
import csv
import json

# TSV/CSV 转 corpus.jsonl 示例
with open("corpus.tsv", "r") as fin, open("corpus.jsonl", "w") as fout:
    reader = csv.DictReader(fin, delimiter="\t")  # CSV 使用 delimiter=","
    for row in reader:
        obj = {"_id": row["id"], "text": row["text"]}
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
```
