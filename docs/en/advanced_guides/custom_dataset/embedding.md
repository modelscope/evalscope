# Embedding Model

## Custom Text Retrieval Evaluation

### 1. Construct Dataset

Create data in the following format:

```
retrieval_data
├── corpus.jsonl
├── queries.jsonl
└── qrels
    └── test.tsv
```

Where:
- `corpus.jsonl`: The corpus file, each line is a JSON object with the format `{"_id": "xxx", "text": "xxx"}`. `_id` is the corpus ID, and `text` is the corpus text. Example:
  ```json
  {"_id": "doc1", "text": "Climate change is leading to more extreme weather patterns."}
  {"_id": "doc2", "text": "The stock market surged today, led by tech stocks."}
  {"_id": "doc3", "text": "Artificial intelligence is transforming industries by automating tasks and providing insights."}
  {"_id": "doc4", "text": "With technological advances, renewable energy like wind and solar is becoming more prevalent."}
  {"_id": "doc5", "text": "Recent studies show that a balanced diet and regular exercise can significantly improve mental health."}
  {"_id": "doc6", "text": "Virtual reality is creating new opportunities in education, entertainment, and training."}
  {"_id": "doc7", "text": "Electric vehicles are gaining popularity due to environmental benefits and advancements in battery technology."}
  {"_id": "doc8", "text": "Space exploration missions are uncovering new information about our solar system and beyond."}
  {"_id": "doc9", "text": "Blockchain technology has potential applications beyond cryptocurrencies, including supply chain management and secure voting systems."}
  {"_id": "doc10", "text": "The benefits of remote work include greater flexibility and reduced commuting time."}
  ```

- `queries.jsonl`: The queries file, each line is a JSON object with the format `{"_id": "xxx", "text": "xxx"}`. `_id` is the query ID, and `text` is the query text. Example:

  ```json
  {"_id": "query1", "text": "What are the impacts of climate change?"}
  {"_id": "query2", "text": "What caused the stock market to rise today?"}
  {"_id": "query3", "text": "How is artificial intelligence changing industries?"}
  {"_id": "query4", "text": "What advancements have been made in renewable energy?"}
  {"_id": "query5", "text": "How does a balanced diet improve mental health?"}
  {"_id": "query6", "text": "What new opportunities has virtual reality created?"}
  {"_id": "query7", "text": "Why are electric vehicles becoming more popular?"}
  {"_id": "query8", "text": "What new information has been revealed by space exploration missions?"}
  {"_id": "query9", "text": "What are the applications of blockchain technology beyond cryptocurrencies?"}
  {"_id": "query10", "text": "What are the benefits of remote work?"}
  ```

- `qrels`: The relevance judgments file, which can include multiple `tsv` files with the format `query-id doc-id score`. `query-id` is the query ID, `doc-id` is the corpus ID, and `score` is the relevance score of the corpus to the query. Example:
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

### 2. Construct Configuration File

Here is an example configuration file:

```python
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

**Parameter Explanation**

The basic parameters are consistent with the [default configuration](../../user_guides/backend/rageval_backend/mteb.md#parameter-explanation). The parameters that need modification are:
- `eval`:
  - `tasks`: Evaluation task, must be `CustomRetrieval`.
  - `dataset_path`: Dataset path, which is the path to the custom dataset.

### 3. Run Evaluation

Run the following code to start the evaluation:
```python
from evalscope.run import run_task

run_task(task_cfg=task_cfg)
```
