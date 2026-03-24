# Examples

End-to-end examples showing how to translate natural language requests into evalscope CLI commands.

## Example 1: Evaluate Math Ability

**User**: "Evaluate Qwen2.5-7B's math reasoning ability"

**Thinking**: User wants to evaluate math capability. From the quick lookup table: Math → gsm8k, math_500, aime24, competition_math. Use gsm8k and math_500 as a good baseline. Model is a local checkpoint.

**Command**:
```bash
evalscope eval \
  --model Qwen/Qwen2.5-7B-Instruct \
  --datasets gsm8k math_500 \
  --limit 10 \
  --seed 42
```

**Follow-up**: After results, read `outputs/<timestamp>/reports/` to summarize accuracy scores.

---

## Example 2: Benchmark an API-Served Model

**User**: "I deployed a model at localhost:8000, help me run a benchmark"

**Thinking**: User has an API service. Need to clarify: model name, what to evaluate. Suggest a general evaluation suite.

**Clarification**: "What's the model name? And would you like a general capability evaluation (knowledge, math, reasoning, instruction following) or a specific focus?"

**User**: "It's Qwen2.5-72B, do a general eval"

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

## Example 3: Throughput Stress Test

**User**: "Test my model's throughput, it's running at http://10.0.0.1:8080"

**Thinking**: User wants performance testing. Need model name. Use perf command with openqa dataset.

**Clarification**: "What's the model name served at that endpoint?"

**User**: "qwen2.5-72b"

**Command**:
```bash
evalscope perf \
  --model qwen2.5-72b \
  --url http://10.0.0.1:8080/v1/chat/completions \
  --api openai \
  --dataset openqa \
  --parallel 5 \
  --number 200 \
  --max-tokens 512 \
  --stream
```

---

## Example 4: Concurrency Gradient Test

**User**: "Run a concurrency gradient test from 1 to 32 on my model"

**Thinking**: User wants to see how performance changes with increasing concurrency. Use parallel with multiple values. Need URL and model name.

**Clarification**: "What's the model endpoint URL and model name?"

**User**: "localhost:8000, model is qwen2.5-14b"

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

## Example 5: View Previous Results

**User**: "Show me the results from my last evaluation"

**Thinking**: Check the outputs directory for the latest results.

**Steps**:
1. List output directories: `ls -lt outputs/ | head -10`
2. Find the latest report: `ls outputs/<latest_dir>/reports/`
3. Read the JSON report to summarize scores
4. Or launch the visualization UI:

```bash
evalscope app --outputs ./outputs --server-port 7860 --lang en
```

---

## Example 6: Find Chinese Benchmarks

**User**: "What Chinese evaluation benchmarks are available?"

**Thinking**: User wants to discover benchmarks by capability. Use `--tag` to filter by the Chinese tag directly.

**Command**:
```bash
evalscope benchmark-info --list --tag Chinese
```

This returns a filtered table showing only benchmarks tagged with "Chinese" (e.g. ceval, cmmlu, chinese_simpleqa).

**Follow-up**: For details on any specific benchmark:
```bash
evalscope benchmark-info ceval
```

---

## Example 7: Get Benchmark Details

**User**: "Tell me about the GSM8K benchmark"

**Command** (text format, default):
```bash
evalscope benchmark-info gsm8k
```

This prints a detailed text summary: name, dataset ID, category, tags, few-shot count, metrics, description, prompt template, configurable parameters, and data statistics.

**For structured data** (e.g. to extract specific fields programmatically):
```bash
evalscope benchmark-info gsm8k --format json
```

**For full markdown documentation**:
```bash
evalscope benchmark-info gsm8k --format markdown
```

---

## Example 8: Quick Config Validation

**User**: "I want to evaluate my model on MMLU but first just want to make sure it works"

**Thinking**: Suggest a dry-run with small limit to validate the configuration.

**Command**:
```bash
evalscope eval \
  --model qwen-plus \
  --datasets mmlu \
  --api-url http://localhost:8000/v1/chat/completions \
  --api-key sk-xxx \
  --limit 5
```

If this succeeds, remove `--limit` for the full evaluation.

---

## Example 9: Compare Two Models

**User**: "Compare Qwen2.5-7B and Qwen2.5-14B on math and coding"

**Thinking**: Run two separate evaluations on the same benchmarks, then compare scores.

**Commands**:
```bash
# Evaluate first model
evalscope eval \
  --model Qwen/Qwen2.5-7B-Instruct \
  --datasets gsm8k math_500 humaneval \
  --work-dir ./outputs/qwen7b \
  --no-timestamp

# Evaluate second model
evalscope eval \
  --model Qwen/Qwen2.5-14B-Instruct \
  --datasets gsm8k math_500 humaneval \
  --work-dir ./outputs/qwen14b \
  --no-timestamp
```

After both complete, read and compare the JSON reports from each output directory. Or use the visualization UI to compare:
```bash
evalscope app --outputs ./outputs --server-port 7860
```

---

## Example 10: SLA Auto-Tuning

**User**: "Find the maximum concurrency my model can handle while keeping latency under 3 seconds"

**Thinking**: Use SLA auto-tuning with parallel as the variable and latency constraint.

**Command**:
```bash
evalscope perf \
  --model qwen2.5-72b \
  --url http://localhost:8000/v1/chat/completions \
  --api openai \
  --dataset openqa \
  --max-tokens 512 \
  --sla-auto-tune \
  --sla-variable parallel \
  --sla-params '[{"name": "latency", "operator": "<=", "value": 3.0}]' \
  --sla-num-runs 3
```

This uses binary search to find the highest `--parallel` value where average latency stays under 3 seconds.
