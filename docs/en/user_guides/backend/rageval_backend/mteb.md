(mteb)=

# MTEB Text Embedding Evaluation

This framework supports [MTEB 2.x](https://github.com/embeddings-benchmark/mteb) (Massive Text Embedding Benchmark) for measuring text embedding model performance across retrieval, reranking, classification, clustering, semantic text similarity, and more. Supports 35+ Chinese datasets and 112+ languages.

```{warning}
**Breaking Changes (v1.9.0)**

If you are upgrading from an older version, please note:

1. **Config schema**: `model` -> `models`, `eval.tasks` -> `eval.task_names`, `eval.topk` -> `eval.top_k`
2. **Dependencies**: Requires `mteb>=2.7.0` (was 1.x) and `ragas>=0.4.0` (was 0.2.x)
3. **Dataset migration**: All Chinese datasets migrated from `C-MTEB/` to `mteb/` organization
4. **Custom Dataset format changes**: Unified to Retrieval format (JSONL), deprecated TSV qrels and Reranking/STS custom task types; qrels changed from `qrels/test.tsv` to flat `qrels.jsonl`; column names unified: `_id` (corpus/queries), `query-id` + `corpus-id` + `score` (qrels)
5. **Removed imports**: `from evalscope.backend.rag_eval import EmbeddingModel` -> use `load_model()`; `from evalscope.backend.rag_eval.cmteb import ...` -> use `evalscope.backend.rag_eval.mteb`
```

## Environment Setup

```bash
pip install evalscope[rag] -U
```

## Scenario 1: Evaluate Local Embedding Models (Quick Start)

> For: locally deployed sentence-transformer models

### Configuration and Running

```python
from evalscope.run import run_task

task_cfg = {
    "eval_backend": "RAGEval",
    "eval_config": {
        "tool": "mteb",
        "models": [
            {
                "model_name_or_path": "AI-ModelScope/m3e-base",
                "max_seq_length": 512,
                "model_kwargs": {"torch_dtype": "auto"},
                "encode_kwargs": {"batch_size": 128},
            }
        ],
        "eval": {
            "task_names": ["TNews", "CLSClusteringS2S", "T2Reranking", "T2Retrieval", "ATEC"],
            "verbosity": 2,
            "overwrite_results": True,
            "top_k": 10,
            "limits": 500,
        },
    },
}

run_task(task_cfg=task_cfg)
```

### Understanding Results

Results are saved to `outputs/<model_name>/<revision>/<task_name>.json`. Key fields:

```json
{
  "task_name": "TNews",
  "scores": {
    "validation": [
      {
        "main_score": 0.4744,
        "accuracy": 0.4744,
        "f1": 0.4456,
        "f1_weighted": 0.4754
      }
    ]
  }
}
```

The `main_score` is the primary metric for each task (accuracy for classification, ndcg@10 for retrieval).

### Key Parameters for This Scenario

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name_or_path` | `str` | `""` | Model name or path, supports auto-download from ModelScope |
| `max_seq_length` | `int` | `512` | Maximum sequence length |
| `model_kwargs` | `dict` | `{}` | Model loading args, e.g. `{"torch_dtype": "auto"}` |
| `encode_kwargs` | `dict` | `{"batch_size": 32}` | Encoding args |
| `task_names` | `List[str]` | `None` | List of tasks to evaluate |
| `limits` | `Optional[int]` | `None` | Limit sample count (recommended for debugging) |

## Scenario 2: Evaluate API Models (DashScope/OpenAI, etc.)

> For: Embedding/Reranker services accessed via API

### Prerequisites

Ensure the model service is deployed and accessible. Obtain the API endpoint and key.

### Configuration and Running

```python
import os
from evalscope.run import run_task
from evalscope import TaskConfig

task_cfg = TaskConfig(
    eval_backend='RAGEval',
    eval_config={
        'tool': 'MTEB',
        'models': [
            {
                'model_name': 'text-embedding-v3',
                'api_base': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
                'api_key': os.environ.get('DASHSCOPE_API_KEY', 'EMPTY'),
                'dimensions': 1024,
                'encode_kwargs': {
                    'batch_size': 10,
                },
            }
        ],
        'eval': {
            'task_names': ['T2Retrieval'],
            'verbosity': 2,
            'overwrite_results': True,
            'limits': 30,
        },
    },
)

run_task(task_cfg=task_cfg)
```

### Understanding Results

```json
{
  "task_name": "T2Retrieval",
  "scores": {
    "dev": [
      {
        "main_score": 0.73143,
        "ndcg_at_10": 0.73143,
        "recall_at_10": 0.73318,
        "precision_at_1": 0.78989
      }
    ]
  }
}
```

### Key Parameters for This Scenario

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | `str` | - | API model name |
| `api_base` | `str` | - | API service URL |
| `api_key` | `str` | - | API key |
| `dimensions` | `Optional[int]` | `None` | Model output dimensions |
| `is_cross_encoder` | `bool` | `False` | Set to `True` for API rerankers |

## Scenario 3: Two-Stage Evaluation (Retrieval + Reranking)

> For: pipelines that first retrieve with an Embedding model, then re-rank with a Reranker

### When to Choose Two-Stage

- **Single-stage**: Directly evaluate embedding models on retrieval/classification/clustering/STS tasks — suitable for most scenarios
- **Two-stage**: When you need to evaluate a reranker model's effect in a real retrieval pipeline; first retrieves top-K candidates with an embedding model, then re-ranks them with a reranker

### Configuration and Running

Pass two models in `models`: the first is the embedding model for retrieval, the second is the reranking model (set `is_cross_encoder: True`).

```python
from evalscope.run import run_task

task_cfg = {
    "eval_backend": "RAGEval",
    "eval_config": {
        "tool": "MTEB",
        "models": [
            {
                "model_name_or_path": "AI-ModelScope/m3e-base",
                "is_cross_encoder": False,
                "max_seq_length": 512,
                "model_kwargs": {"torch_dtype": "auto"},
                "encode_kwargs": {"batch_size": 64},
            },
            {
                "model_name_or_path": "OpenBMB/MiniCPM-Reranker",
                "is_cross_encoder": True,
                "max_seq_length": 512,
                "prompt": "Generate a retrieval representation for this question",
                "model_kwargs": {"torch_dtype": "auto"},
                "encode_kwargs": {"batch_size": 32},
            },
        ],
        "eval": {
            "task_names": ["T2Retrieval"],
            "verbosity": 2,
            "overwrite_results": True,
            "top_k": 5,
            "limits": 100,
        },
    },
}

run_task(task_cfg=task_cfg)
```

### Understanding Results

Two-stage evaluation outputs results for both stage 1 (retrieval) and stage 2 (reranking):

**Stage 1** (`outputs/stage1/<embedding_model>/...`):
```json
{
  "task_name": "T2Retrieval",
  "scores": {"dev": [{"main_score": 0.73143, "ndcg_at_10": 0.73143}]}
}
```

**Stage 2** (`outputs/stage2/<reranker_model>/...`):
```json
{
  "task_name": "T2Retrieval",
  "scores": {"dev": [{"main_score": 0.661, "ndcg_at_10": 0.661}]}
}
```

### Two-Stage Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `is_cross_encoder` | `bool` | `False` | Must be `True` for the second model (reranker) |
| `top_k` | `int` | `10` | Number of candidates retrieved in stage 1, passed to stage 2 for reranking |
| `prompt` | `Optional[str]` | `None` | Query prefix prompt for retrieval tasks |

## Scenario 4: Custom Dataset Evaluation

> For: evaluating models with private datasets

### Data Preparation

Custom datasets must follow the Retrieval format (JSONL). For detailed format specifications:

```{seealso}
[Custom Retrieval Dataset](../../../advanced_guides/custom_dataset/embedding.md)
```

### Configuration and Running

Use `custom_tasks` in `eval` to specify your custom dataset path:

```python
from evalscope.run import run_task

task_cfg = {
    "eval_backend": "RAGEval",
    "eval_config": {
        "tool": "MTEB",
        "models": [
            {
                "model_name_or_path": "AI-ModelScope/m3e-base",
                "max_seq_length": 512,
                "model_kwargs": {"torch_dtype": "auto"},
                "encode_kwargs": {"batch_size": 128},
            }
        ],
        "eval": {
            "custom_tasks": ["path/to/your/custom_dataset"],
            "verbosity": 2,
            "overwrite_results": True,
        },
    },
}

run_task(task_cfg=task_cfg)
```

## Supported Datasets

<details><summary>Click to expand full dataset list</summary>

| Name | Hub Link | Description | Type | Category | Number of Test Samples |
|-----|-----|---------------------------|-----|-----|-----|
| [T2Retrieval](https://arxiv.org/abs/2304.03679) | [mteb/T2Retrieval](https://modelscope.cn/datasets/mteb/T2Retrieval) | T2Ranking: A large-scale Chinese paragraph ranking benchmark | Retrieval | s2p | 24,832 |
| [MMarcoRetrieval](https://github.com/unicamp-dl/mMARCO) | [mteb/MMarcoRetrieval](https://modelscope.cn/datasets/mteb/MMarcoRetrieval) | mMARCO is the multilingual version of the MS MARCO paragraph ranking dataset | Retrieval | s2p | 7,437 |
| [DuRetrieval](https://aclanthology.org/2022.emnlp-main.357.pdf) | [mteb/DuRetrieval](https://modelscope.cn/datasets/mteb/DuRetrieval) | A large-scale Chinese web search engine paragraph retrieval benchmark | Retrieval | s2p | 4,000 |
| [CovidRetrieval](https://aclanthology.org/2022.emnlp-main.357.pdf) | [mteb/CovidRetrieval](https://modelscope.cn/datasets/mteb/CovidRetrieval) | COVID-19 news articles | Retrieval | s2p | 949 |
| [CmedqaRetrieval](https://aclanthology.org/2022.emnlp-main.357.pdf) | [mteb/CmedqaRetrieval](https://modelscope.cn/datasets/mteb/CmedqaRetrieval) | Online medical consultation texts | Retrieval | s2p | 3,999 |
| [EcomRetrieval](https://arxiv.org/abs/2203.03367) | [mteb/EcomRetrieval](https://modelscope.cn/datasets/mteb/EcomRetrieval) | Paragraph retrieval dataset collected from Alibaba e-commerce search engine systems | Retrieval | s2p | 1,000 |
| [MedicalRetrieval](https://arxiv.org/abs/2203.03367) | [mteb/MedicalRetrieval](https://modelscope.cn/datasets/mteb/MedicalRetrieval) | Paragraph retrieval dataset collected from Alibaba medical search engine systems | Retrieval | s2p | 1,000 |
| [VideoRetrieval](https://arxiv.org/abs/2203.03367) | [mteb/VideoRetrieval](https://modelscope.cn/datasets/mteb/VideoRetrieval) | Paragraph retrieval dataset collected from Alibaba video search engine systems | Retrieval | s2p | 1,000 |
| [T2Reranking](https://arxiv.org/abs/2304.03679) | [mteb/T2Reranking](https://modelscope.cn/datasets/mteb/T2Reranking) | T2Ranking: A large-scale Chinese paragraph ranking benchmark | Re-ranking | s2p | 24,382 |
| [MMarcoReranking](https://github.com/unicamp-dl/mMARCO) | [mteb/MMarco-reranking](https://modelscope.cn/datasets/mteb/Mmarco-reranking) | mMARCO is the multilingual version of the MS MARCO paragraph ranking dataset | Re-ranking | s2p | 7,437 |
| [CMedQAv1](https://github.com/zhangsheng93/cMedQA) | [mteb/CMedQAv1-reranking](https://modelscope.cn/datasets/mteb/CMedQAv1-reranking) | Chinese community medical Q&A | Re-ranking | s2p | 2,000 |
| [CMedQAv2](https://github.com/zhangsheng93/cMedQA2) | [mteb/CMedQAv2-reranking](https://modelscope.cn/datasets/mteb/CMedQAv2-reranking) | Chinese community medical Q&A | Re-ranking | s2p | 4,000 |
| [Ocnli](https://arxiv.org/abs/2010.05444) | [mteb/OCNLI](https://modelscope.cn/datasets/mteb/OCNLI) | Original Chinese natural language inference dataset | Pair Classification | s2s | 3,000 |
| [Cmnli](https://modelscope.cn/datasets/clue/viewer/cmnli) | [mteb/CMNLI](https://modelscope.cn/datasets/mteb/CMNLI) | Chinese multi-class natural language inference | Pair Classification | s2s | 139,000 |
| [CLSClusteringS2S](https://arxiv.org/abs/2209.05034) | [mteb/CLSClusteringS2S](https://modelscope.cn/datasets/mteb/CLSClusteringS2S) | Clustering titles from the CLS dataset. Clustering based on 13 sets of main categories. | Clustering | s2s | 10,000 |
| [CLSClusteringP2P](https://arxiv.org/abs/2209.05034) | [mteb/CLSClusteringP2P](https://modelscope.cn/datasets/mteb/CLSClusteringP2P) | Clustering titles + abstracts from the CLS dataset. Clustering based on 13 sets of main categories. | Clustering | p2p | 10,000 |
| [ThuNewsClusteringS2S](http://thuctc.thunlp.org/) | [mteb/ThuNewsClusteringS2S](https://modelscope.cn/datasets/mteb/ThuNewsClusteringS2S) | Clustering titles from the THUCNews dataset | Clustering | s2s | 10,000 |
| [ThuNewsClusteringP2P](http://thuctc.thunlp.org/) | [mteb/ThuNewsClusteringP2P](https://modelscope.cn/datasets/mteb/ThuNewsClusteringP2P) | Clustering titles + abstracts from the THUCNews dataset | Clustering | p2p | 10,000 |
| [ATEC](https://github.com/IceFlameWorm/NLP_Datasets/tree/master/ATEC) | [mteb/ATEC](https://modelscope.cn/datasets/mteb/ATEC) | ATEC NLP Sentence Pair Similarity Competition | STS | s2s | 20,000 |
| [BQ](https://huggingface.co/datasets/shibing624/nli_zh) | [mteb/BQ](https://modelscope.cn/datasets/mteb/BQ) | Banking Question Semantic Similarity | STS | s2s | 10,000 |
| [LCQMC](https://huggingface.co/datasets/shibing624/nli_zh) | [mteb/LCQMC](https://modelscope.cn/datasets/mteb/LCQMC) | Large-scale Chinese Question Matching Corpus | STS | s2s | 12,500 |
| [PAWSX](https://arxiv.org/pdf/1908.11828.pdf) | [mteb/PAWSX](https://modelscope.cn/datasets/mteb/PAWSX) | Translated PAWS evaluation pairs | STS | s2s | 2,000 |
| [STSB](https://github.com/pluto-junzeng/CNSD) | [mteb/STSB](https://modelscope.cn/datasets/mteb/STSB) | Translated STS-B into Chinese | STS | s2s | 1,360 |
| [AFQMC](https://github.com/CLUEbenchmark/CLUE) | [mteb/AFQMC](https://modelscope.cn/datasets/mteb/AFQMC) | Ant Financial Question Matching Corpus | STS | s2s | 3,861 |
| [QBQTC](https://github.com/CLUEbenchmark/QBQTC) | [mteb/QBQTC](https://modelscope.cn/datasets/mteb/QBQTC) | QQ Browser Query Title Corpus | STS | s2s | 5,000 |
| [TNews](https://github.com/CLUEbenchmark/CLUE) | [mteb/TNews-classification](https://modelscope.cn/datasets/mteb/TNews-classification) | News Short Text Classification | Classification | s2s | 10,000 |
| [IFlyTek](https://github.com/CLUEbenchmark/CLUE) | [mteb/IFlyTek-classification](https://modelscope.cn/datasets/mteb/IFlyTek-classification) | Long Text Classification of Application Descriptions | Classification | s2s | 2,600 |
| [Waimai](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/waimai_10k/intro.ipynb) | [mteb/waimai-classification](https://modelscope.cn/datasets/mteb/waimai-classification) | Sentiment Analysis of User Reviews on Food Delivery Platforms | Classification | s2s | 1,000 |
| [OnlineShopping](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/online_shopping_10_cats/intro.ipynb) | [mteb/OnlineShopping-classification](https://modelscope.cn/datasets/mteb/OnlineShopping-classification) | Sentiment Analysis of User Reviews on Online Shopping Websites | Classification | s2s | 1,000 |
| [MultilingualSentiment](https://github.com/tyqiangz/multilingual-sentiment-datasets) | [mteb/MultilingualSentiment-classification](https://modelscope.cn/datasets/mteb/MultilingualSentiment-classification) | A set of multilingual sentiment datasets grouped into three categories: positive, neutral, negative | Classification | s2s | 3,000 |
| [JDReview](https://huggingface.co/datasets/kuroneko5943/jd21) | [mteb/JDReview-classification](https://modelscope.cn/datasets/mteb/JDReview-classification) | Reviews of iPhone | Classification | s2s | 533 |

For retrieval tasks, a sample of 100,000 candidates (including the ground truth) is drawn from the entire corpus to reduce inference costs.

</details>

```{seealso}
- [Datasets Supported by MTEB](https://embeddings-benchmark.github.io/mteb/overview/)
```

## Full Parameter Reference

### Local Model Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name_or_path` | `str` | `""` | Model name or path, supports automatic download from ModelScope |
| `is_cross_encoder` | `bool` | `False` | Whether the model is a cross encoder; set to `True` for reranking models |
| `pooling_mode` | `Optional[str]` | `None` | Pooling mode. Options: cls / lasttoken / max / mean / mean_sqrt_len_tokens / weightedmean. Set to cls for `bge` series models |
| `max_seq_length` | `int` | `512` | Maximum sequence length |
| `prompt` | `Optional[str]` | `None` | Query prefix prompt for retrieval tasks |
| `prompts` | `Optional[Dict[str, str]]` | `None` | Per-task prompts (key=task name, value=prompt). Only effective when `prompt` is not set |
| `model_kwargs` | `dict` | `{}` | Model loading arguments, e.g. `{"torch_dtype": "auto"}` |
| `encode_kwargs` | `dict` | `{"batch_size": 32}` | Encoding arguments |
| `hub` | `str` | `"modelscope"` | Model source: `modelscope` or `huggingface` |

### Remote API Model Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | `str` | - | API model name |
| `is_cross_encoder` | `bool` | `False` | Whether the model is a reranker; set to `True` for API rerankers |
| `api_base` | `str` | - | API base URL; rerankers call the rerank endpoint — if the URL does not end with `/rerank` or `/reranks`, `/rerank` is appended by default (`/reranks` for DashScope domains) |
| `api_key` | `str` | - | API key |
| `dimensions` | `Optional[int]` | `None` | Model output dimensions |
| `max_seq_length` | `int` | `512` | Maximum sequence length; texts exceeding this limit are automatically truncated before sending to the API |
| `encode_kwargs` | `dict` | `{"batch_size": 10}` | Encoding arguments |

### Evaluation Configuration (eval)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task_names` | `Optional[List[str]]` | `None` | Task name list, see [Supported Datasets](#supported-datasets) |
| `task_types` | `Optional[List[str]]` | `None` | Filter by task type, e.g. `["Retrieval", "STS"]` |
| `languages` | `Optional[List[str]]` | `None` | Filter by language, e.g. `["cmn-Hans"]` |
| `custom_tasks` | `Optional[List]` | `None` | Custom task configuration list, see [Scenario 4](#scenario-4-custom-dataset-evaluation) |
| `top_k` | `int` | `10` | Select top K results for retrieval tasks |
| `overwrite_results` | `bool` | `True` | Whether to overwrite existing results |
| `limits` | `Optional[int]` | `None` | Limit on number of samples; for retrieval tasks, limits both queries and corpus (only keeps documents referenced by qrels) |
| `output_folder` | `str` | `"outputs"` | Output directory for results |
| `hub` | `str` | `"modelscope"` | Dataset source: `modelscope` or `huggingface` |

### Top-Level Configuration

- `eval_backend`: Fixed to `RAGEval`, indicating the RAGEval evaluation backend
- `eval_config.tool`: Fixed to `MTEB`
- `eval_config.models`: Model configuration list. Single-stage takes one model; two-stage takes two models (first for retrieval, second for reranking)

## FAQ

**Model loading failure / OOM**

- Ensure sufficient GPU memory; reduce `encode_kwargs.batch_size` if needed
- Use `model_kwargs: {"torch_dtype": "float16"}` to reduce memory usage
- Verify the `model_name_or_path` is correct

**Dataset download failure**

- Default source is ModelScope; ensure `modelscope.cn` is reachable
- Set `hub: "huggingface"` to switch to HuggingFace source
- For locally available datasets, specify a local path via `model_name_or_path`

**Understanding result metrics**

- `main_score`: The primary evaluation metric for the task
- Retrieval: `ndcg_at_10` (Normalized Discounted Cumulative Gain)
- Reranking: `map` (Mean Average Precision)
- Classification: `accuracy`
- STS: `cosine_spearman` (Spearman correlation of cosine similarity)
- Clustering: `v_measure`
