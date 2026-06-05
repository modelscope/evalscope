# Examples

End-to-end examples showing natural language → evalscope commands.

## Example 1: Evaluate Local Checkpoint on Math

**User**: "Evaluate Qwen2.5-7B's math reasoning ability"

**Command**:
```bash
evalscope eval \
  --model Qwen/Qwen2.5-7B-Instruct \
  --datasets gsm8k math_500 \
  --limit 10 \
  --seed 42
```

---

## Example 2: Benchmark an API Model (General)

**User**: "I deployed qwen2.5-72b at localhost:8000, run a general eval"

**Command**:
```bash
evalscope eval \
  --model qwen2.5-72b \
  --datasets mmlu gsm8k bbh humaneval ifeval \
  --api-url http://localhost:8000/v1/chat/completions \
  --api-key EMPTY \
  --limit 50
```

---

## Example 3: Concurrency Gradient Perf Test

**User**: "Run a concurrency gradient test from 1 to 32 on qwen2.5-14b at localhost:8000"

**Command**:
```bash
evalscope perf \
  --model qwen2.5-14b \
  --url http://localhost:8000/v1/chat/completions \
  --api openai \
  --dataset openqa \
  --parallel 1 2 4 8 16 32 \
  --number 50 100 200 400 800 1600 \
  --stream
```

---

## Example 4: Discover Chinese Benchmarks

**User**: "What Chinese evaluation benchmarks are available?"

**Command**:
```bash
evalscope benchmark-info --list --tag Chinese
```

**Follow-up** (details on one):
```bash
evalscope benchmark-info ceval --format markdown
```

---

## Example 5: RAG Evaluation with MTEB

**User**: "Evaluate my embedding model bge-large-zh on retrieval tasks"

**Command** (Python config):
```python
from evalscope import run_task
run_task({
    'eval_backend': 'RAGEval',
    'eval_config': {
        'tool': 'MTEB',
        'model': [{
            'model_name_or_path': 'AI-ModelScope/bge-large-zh',
            'pooling_mode': 'cls',
            'max_seq_length': 512,
            'encode_kwargs': {'batch_size': 256},
        }],
        'eval': {
            'tasks': ['T2Retrieval', 'MedicalRetrieval'],
            'overwrite_results': True,
            'top_k': 10,
        },
    },
})
```

---

## Example 6: Resume Failed Evaluation

**User**: "My eval on MMLU crashed halfway, can I continue from where it stopped?"

**Steps**:
1. Find the incomplete output directory: `ls -lt outputs/ | head -5`
2. Resume with `--use-cache`:

```bash
evalscope eval \
  --model qwen-plus \
  --datasets mmlu \
  --api-url http://localhost:8000/v1/chat/completions \
  --use-cache outputs/20250601_120000
```

Cached predictions are reused; only missing samples are re-evaluated.

---

## Example 7: Debug Connection Issue

**User**: "Perf test just says connection failed, what do I do?"

**Steps**:
1. Verify endpoint manually:
```bash
curl -s http://localhost:8000/v1/models
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen-plus", "messages": [{"role": "user", "content": "hi"}]}'
```

2. If the server returns 200 but perf still fails, check model name matches exactly.

3. Run with `--debug` for verbose output:
```bash
evalscope perf --model qwen-plus \
  --url http://localhost:8000/v1/chat/completions \
  --api openai --dataset openqa --parallel 1 --number 1 --debug
```

4. If server is slow to start, skip pre-flight: add `--no-test-connection`.
