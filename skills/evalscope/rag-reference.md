# RAG Evaluation Reference

RAG evaluation uses `eval_backend: 'RAGEval'` with a Python dict config (not CLI flags). Three tools available.

Requires: `pip install 'evalscope[rag]'`

## Tool: RAGAS

RAG quality evaluation using RAGAS 0.4.x. Two modes: testset generation and evaluation.

### Testset Generation

```python
from evalscope import run_task
run_task({
    'eval_backend': 'RAGEval',
    'eval_config': {
        'tool': 'RAGAS',
        'testset_generation': {
            'docs': ['path/to/doc1.md', 'path/to/doc2.pdf'],
            'test_size': 10,
            'output_file': 'outputs/testset.json',
            'generator_llm': {
                'model_name': 'qwen-plus',
                'provider': 'openai',        # default: 'openai'
                'api_base': 'http://localhost:8000/v1',
                'api_key': 'EMPTY',
            },
            'embeddings': {
                'model_name_or_path': 'AI-ModelScope/bge-large-zh',
                'provider': 'huggingface',   # default: 'huggingface'
            },
            'language': 'chinese',           # default: 'english'
        },
    },
})
```

### Evaluation

```python
run_task({
    'eval_backend': 'RAGEval',
    'eval_config': {
        'tool': 'RAGAS',
        'eval': {
            'testset_file': 'outputs/testset_with_answer.json',
            'critic_llm': {
                'model_name': 'qwen-plus',
                'api_base': 'http://localhost:8000/v1',
                'api_key': 'EMPTY',
            },
            'embeddings': {
                'model_name_or_path': 'AI-ModelScope/bge-large-zh',
            },
            'metrics': [
                'Faithfulness',
                'AnswerRelevancy',
                'ContextPrecision',
                'AnswerCorrectness',
            ],
            'language': 'chinese',
        },
    },
})
```

Multimodal metrics: `MultiModalFaithfulness`, `MultiModalRelevance`.

## Tool: MTEB

Embedding model evaluation using MTEB 2.7+. Supports local models, API models, and two-stage retrieval (encoder + reranker).

### Single Model

```python
run_task({
    'eval_backend': 'RAGEval',
    'eval_config': {
        'tool': 'MTEB',
        'model': [{
            'model_name_or_path': 'AI-ModelScope/bge-large-zh',
            'pooling_mode': 'cls',
            'max_seq_length': 512,
            'prompt': '为这个句子生成表示以用于检索相关文章：',
            'encode_kwargs': {'batch_size': 512},
        }],
        'eval': {
            'tasks': ['MedicalRetrieval', 'T2Retrieval'],
            'overwrite_results': True,
            'top_k': 10,
            'limits': 1000,    # None for no limit
        },
    },
})
```

### API Embedding Model

```python
'model': [{
    'model_name': 'text-embedding-v3',
    'api_base': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    'api_key': 'xxx',
    'dimensions': 1024,
    'encode_kwargs': {'batch_size': 10},
}]
```

### Two-Stage Retrieval (Encoder + Reranker)

```python
'model': [
    {  # Stage 1: encoder
        'model_name_or_path': 'AI-ModelScope/m3e-base',
        'is_cross_encoder': False,
        'max_seq_length': 512,
    },
    {  # Stage 2: reranker
        'model_name_or_path': 'OpenBMB/MiniCPM-Reranker',
        'is_cross_encoder': True,
        'max_seq_length': 512,
    },
]
```

## Tool: clip_benchmark

Multimodal retrieval evaluation (CLIP models or API VLMs).

```python
run_task({
    'eval_backend': 'RAGEval',
    'eval_config': {
        'tool': 'clip_benchmark',
        'eval': {
            'models': [{
                'model_name': 'internvl2-8b',
                'api_base': 'http://localhost:8008/v1',
                'prompt': '简要描述这张图片',
            }],
            'dataset_name': ['muge'],
            'split': 'test',
            'task': 'image_caption',  # or zeroshot_classification, zeroshot_retrieval
            'batch_size': 2,
            'limit': 10,
        },
    },
})
```

Tasks: `image_caption`, `zeroshot_classification`, `zeroshot_retrieval`.
