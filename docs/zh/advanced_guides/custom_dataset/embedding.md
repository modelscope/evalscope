# Embedding模型

## 自定义文本检索评测

### 1. 构建数据集

构建如下格式数据：

```
retrieval_data
├── corpus.jsonl
├── queries.jsonl
└── qrels
    └── test.tsv
```

其中：
- `corpus.jsonl`: 语料库文件，每行一个json，格式为`{"_id": "xxx", "text": "xxx"}`，_id为语料库的id，text为语料库的文本。例如：
  ``` json
  {"_id": "doc1", "text": "气候变化正在导致更极端的天气模式。"}
  {"_id": "doc2", "text": "今天股市大幅上涨，科技股领涨。"}
  {"_id": "doc3", "text": "人工智能正在通过自动化任务和提供见解来改变各种行业。"}
  {"_id": "doc4", "text": "随着技术的进步，风能和太阳能等可再生能源变得越来越普及。"}
  {"_id": "doc5", "text": "最新研究表明，均衡饮食和定期锻炼可以显著改善心理健康。"}
  {"_id": "doc6", "text": "虚拟现实正在教育、娱乐和培训方面创造新的机会。"}
  {"_id": "doc7", "text": "由于环保优势和电池技术的进步，电动汽车越来越受欢迎。"}
  {"_id": "doc8", "text": "太空探索任务正在揭示关于我们的太阳系及其以外的新信息。"}
  {"_id": "doc9", "text": "区块链技术在加密货币之外还有潜在的应用，包括供应链管理和安全投票系统。"}
  {"_id": "doc10", "text": "远程工作的好处包括更大的灵活性和减少通勤时间。"}
  ```

- `queries.jsonl`: 查询文件，每行一个json，格式为`{"_id": "xxx", "text": "xxx"}`，_id为查询的id，text为查询的文本。例如：

  ``` json
  {"_id": "query1", "text": "气候变化的影响是什么？"}
  {"_id": "query2", "text": "今天股市上涨的原因是什么？"}
  {"_id": "query3", "text": "人工智能如何改变行业？"}
  {"_id": "query4", "text": "可再生能源有哪些进展？"}
  {"_id": "query5", "text": "均衡饮食如何改善心理健康？"}
  {"_id": "query6", "text": "虚拟现实创造了哪些新机会？"}
  {"_id": "query7", "text": "为什么电动汽车越来越受欢迎？"}
  {"_id": "query8", "text": "太空探索任务揭示了哪些新信息？"}
  {"_id": "query9", "text": "区块链技术在加密货币之外有哪些应用？"}
  {"_id": "query10", "text": "远程工作的好处是什么？"}
  ```

- `qrels`: 评测文件，可包含多个`tsv`文件，格式为`query-id doc-id  score`，query-id为查询的id，doc-id为语料库的id，score为语料库与查询的相关度打分。例如：
  ```
  query-id	corpus-id	score
  query1	doc1	1
  query2	doc2	1
  query3	doc3	1
  query4	doc4	1
  query5	doc5	1
  query6	doc6	1
  query7	doc7	1
  query8	doc8	1
  query9	doc9	1
  query10	doc10	1
  ```

### 2. 构建配置文件

编写的配置文件示例如下：
``` python
task_cfg = {
    "work_dir": "outputs",
    "eval_backend": "RAGEval",
    "eval_config": {
        "tool": "MTEB",
        "model": [
            {
                "model_name_or_path": "AI-ModelScope/m3e-base",
                "pooling_mode": None,  # load from model config
                "max_seq_length": 512,
                "prompt": "",
                "model_kwargs": {"torch_dtype": "auto"},
                "encode_kwargs": {
                    "batch_size": 128,
                },
            }
        ],
        "eval": {
            "tasks": ["CustomRetrieval"],
            "dataset_path": "custom_eval/text/retrieval",
            "verbosity": 2,
            "overwrite_results": True,
            "limits": 500,
        },
    },
}
```

**参数说明**

基本参数与[默认配置](../../user_guides/backend/rageval_backend/mteb.md#参数说明)一致，需要修改的参数有：
- `eval`:
  - `tasks`: 评测任务，必须为`CustomRetrieval`。
  - `dataset_path`: 数据集路径，为自定义数据集的路径。

### 3. 运行评测

运行下面的代码，即可开始评测：
```python
from evalscope.run import run_task

run_task(task_cfg=task_cfg)
```
