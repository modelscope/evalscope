# ToolBench

## Description

We evaluate the effectiveness of the ToolBench benchmark: [ToolBench](https://arxiv.org/pdf/2307.16789) (Qin et al., 2023b). The task involves integrating API calls to complete tasks. The agent must accurately select appropriate APIs and construct necessary API requests.

Furthermore, we divide the ToolBench test set into in-domain and out-of-domain based on whether the tools used in the test instances were seen during training. This division allows us to evaluate performance in both in-distribution and out-of-distribution scenarios. We refer to this dataset as `ToolBench-Static`.

For more details, please refer to: [Small LLMs Are Weak Tool Learners: A Multi-LLM Agent](https://arxiv.org/abs/2401.07324)

## Dataset

Dataset [link](https://modelscope.cn/datasets/AI-ModelScope/ToolBench-Static/dataPeview)

- Dataset Statistics:
  - In-domain count: 1588
  - Out-of-domain count: 781

## Usage

```python
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='Qwen/Qwen3-1.7B',
    datasets=['tool_bench'],
    limit=5,
    eval_batch_size=5,
    generation_config={
        'max_tokens': 1000,  # Maximum number of tokens to generate, set to a large value to avoid truncation
        'temperature': 0.7,      # Sampling temperature (recommended value by qwen)
        'top_p': 0.8,            # Top-p sampling (recommended value by qwen)
        'top_k': 20,             # Top-k sampling (recommended value by qwen)
        'extra_body':{'chat_template_kwargs': {'enable_thinking': False}}  # Disable thinking mode
    }
)

run_task(task_cfg=task_cfg)
```

Results are as follows (some dirty data is automatically discarded, leading to a reduction in in_domain sample count):
```text
+------------+------------+-----------+---------------+-------+---------+---------+
| Model      | Dataset    | Metric    | Subset        |   Num |   Score | Cat.0   |
+============+============+===========+===============+=======+=========+=========+
| Qwen3-1.7B | tool_bench | Act.EM    | in_domain     |     2 |  0      | default |
+------------+------------+-----------+---------------+-------+---------+---------+
| Qwen3-1.7B | tool_bench | Act.EM    | out_of_domain |     5 |  0.2    | default |
+------------+------------+-----------+---------------+-------+---------+---------+
| Qwen3-1.7B | tool_bench | Plan.EM   | in_domain     |     0 |  0      | default |
+------------+------------+-----------+---------------+-------+---------+---------+
| Qwen3-1.7B | tool_bench | Plan.EM   | out_of_domain |     0 |  0      | default |
+------------+------------+-----------+---------------+-------+---------+---------+
| Qwen3-1.7B | tool_bench | F1        | in_domain     |     2 |  0      | default |
+------------+------------+-----------+---------------+-------+---------+---------+
| Qwen3-1.7B | tool_bench | F1        | out_of_domain |     5 |  0.2    | default |
+------------+------------+-----------+---------------+-------+---------+---------+
| Qwen3-1.7B | tool_bench | HalluRate | in_domain     |     2 |  0      | default |
+------------+------------+-----------+---------------+-------+---------+---------+
| Qwen3-1.7B | tool_bench | HalluRate | out_of_domain |     5 |  0.4    | default |
+------------+------------+-----------+---------------+-------+---------+---------+
| Qwen3-1.7B | tool_bench | Rouge-L   | in_domain     |     2 |  0      | default |
+------------+------------+-----------+---------------+-------+---------+---------+
| Qwen3-1.7B | tool_bench | Rouge-L   | out_of_domain |     5 |  0.1718 | default |
+------------+------------+-----------+---------------+-------+---------+---------+ 
```

## Results and Metrics

- Metrics:
  - `Plan.EM`: The exact match score for the agent's plan decision to use tool calls, generate answers, or give up at each step.
  - `Act.EM`: The exact match score for actions, including tool names and parameters.
  - `HalluRate` (lower is better): The hallucination rate of the agent when answering at each step.
  - `Avg.F1`: The average F1 score for the agent's tool calls at each step.
  - `Rouge-L`: The Rouge-L score for the agent's responses at each step.

Generally, we focus on the `Act.EM`, `HalluRate`, and `Avg.F1` metrics.