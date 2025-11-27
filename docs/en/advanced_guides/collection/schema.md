# Defining Your Schema

In a nutshell: Schema = your capability checklist + relative value (weight) for each capability + optional hierarchical grouping.

## Core Concepts & Objects
- **CollectionSchema**: A nestable grouping node that forms hierarchical structures and reusable modules.
- **DatasetInfo**: The final evaluable dataset entry (leaf node), containing name, weight, and adapter parameters.
- **flatten()**: Expands the entire tree to get a normalized list of leaf datasets (with hierarchical path `hierarchy`).

## Field Overview
**CollectionSchema**:
- `name`: str (required) - Group/index name
- `weight`: float (optional, default 1.0) - Relative weight for the entire subgroup (combined with child nodes, then normalized)
- `datasets`: list[DatasetInfo | CollectionSchema]

**DatasetInfo**:
- `name`: str (required) - Dataset registration name
- `weight`: float (required, > 0)
- `task_type`: str (optional) - Custom capability label
- `tags`: list[str] (optional) - Custom tags; parent CollectionSchema names are automatically appended
- `args`: dict (optional) - Pass-through to dataset adapter (e.g., subset_list, local_path, review_timeout, etc.)

## Why Set Weights?
**Sampling Phase**:
- Under weighted sampling strategy, sample proportion ∝ normalized weight.
- Normalization for flat (non-nested) structure:
$$
\mathbf{p_i} = \frac{\mathbf{w_i}}{\sum_{j=1}^{n} \mathbf{w_j}}
$$
- For multi-level nesting (product of weights along the leaf path, then overall normalization):
$$
\mathbf{p_i} = \frac{\prod_{k \in \text{path}(i)} \mathbf{w_k}}{\sum_{j=1}^{n} \prod_{k \in \text{path}(j)} \mathbf{w_k}}
$$

**Scoring Phase**:
- Aggregation uses linear weighting:
$$
\mathbf{S} = \sum_{i=1}^{n} \boldsymbol{\alpha_i} \, \mathbf{s_i}
$$
where:
$$
\boldsymbol{\alpha_i} = \mathbf{p_i}, \quad \sum_{i=1}^{n} \boldsymbol{\alpha_i} = 1
$$


## Minimal Python Example
```python
from evalscope.collections import CollectionSchema, DatasetInfo

simple_schema = CollectionSchema(
    name='reasoning_index',
    datasets=[
        DatasetInfo(name='arc', weight=2.0, task_type='reasoning', tags=['en']),
        DatasetInfo(name='ceval', weight=3.0, task_type='reasoning', tags=['zh'], args={'subset_list': ['logic']}),
    ],
)
```


## Nested & Multi-Capability Example
```python
from evalscope.collections import CollectionSchema, DatasetInfo

complex_schema = CollectionSchema(name='math_index', datasets=[
    CollectionSchema(name='math', weight=3.0, datasets=[
        DatasetInfo(name='gsm8k', weight=1.0, task_type='math', tags=['en']),
        DatasetInfo(name='aime25', weight=1.0, task_type='math', tags=['en']),
    ]),
    CollectionSchema(name='reasoning', weight=1.0, datasets=[
        DatasetInfo(name='arc', weight=1.0, task_type='reasoning', tags=['en']),
        DatasetInfo(name='ceval', weight=1.0, task_type='reasoning', tags=['zh'], args={'subset_list': ['logic']}),
    ]),
])
```
**Hierarchical Effects**:
- Parent names are appended to each leaf's tags/hierarchy, supporting multi-level aggregated analysis.
- Parent weight scales the entire group, then jointly normalizes with child nodes.

## Flatten / Serialize / Restore
```python
print(simple_schema.flatten())          # Flatten + normalize
print(complex_schema.flatten())         # View normalized weights after nesting
complex_schema.dump_json('outputs/complex_schema.json')
loaded = CollectionSchema.from_json('outputs/complex_schema.json')
```

**Output Example (automatically normalized weights)**:
```python
>>> [DatasetInfo(name='arc', weight=0.4, task_type='reasoning', tags=['en'], args={}, hierarchy=['reasoning_index']),
     DatasetInfo(name='ceval', weight=0.6, task_type='reasoning', tags=['zh'], args={'subset_list': ['logic']}, hierarchy=['reasoning_index'])]

>>> [DatasetInfo(name='gsm8k', weight=0.375, task_type='math', tags=['en'], args={}, hierarchy=['math_index', 'math']),
     DatasetInfo(name='aime25', weight=0.375, task_type='math', tags=['en'], args={}, hierarchy=['math_index', 'math']),
     DatasetInfo(name='arc', weight=0.125, task_type='reasoning', tags=['en'], args={}, hierarchy=['math_index', 'reasoning']),
     DatasetInfo(name='ceval', weight=0.125, task_type='reasoning', tags=['zh'], args={'subset_list': ['logic']}, hierarchy=['math_index', 'reasoning'])]
```

## Common Issues
- Weights must be > 0; directly remove entries with weight = 0.
- Deep nesting can easily cause some leaf weights to be "diluted"; verify manually by flattening first.
- Too many tags affect downstream visualization dimensions; keep them concise (e.g., en / zh / math / rag).
- `args` must match corresponding dataset adapter fields; incorrect fields will be ignored or trigger warnings.
- Cramming too many capabilities into one Schema hinders iteration; split into multiple versions for comparison.


## Reference Cases

### Case A: Enterprise RAG Assistant Index
- **Pain Point**: High scores on general leaderboards, but inaccurate internal knowledge retrieval.
- **Approach**: Knowledge tasks + long-context retrieval + instruction following.
- **Configuration Example**:
```python
from evalscope.collections import CollectionSchema, DatasetInfo

schema = CollectionSchema(name='rag_index', datasets=[
    DatasetInfo(name='chinese_simpleqa', weight=0.3, task_type='knowledge', tags=['rag']),
    DatasetInfo(name='aa_lcr', weight=0.3, task_type='long_context', tags=['rag']),
    DatasetInfo(name='ifeval', weight=0.4, task_type='instruction_following', tags=['rag']),
])
```


### Case B: Code Completion & Analysis Index
- **Pain Point**: General leaderboards cannot comprehensively reflect code understanding and generation capabilities.
- **Approach**: Code generation + real issues.
- **Configuration Example**:
```python
from evalscope.collections import CollectionSchema, DatasetInfo

schema = CollectionSchema(name='code_index', datasets=[
    DatasetInfo(name='humaneval', weight=2.0, task_type='code', tags=['python']),
    DatasetInfo(name='live_code_bench', weight=1.0, task_type='code', tags=['zh', 'finance'], args={'subset_list': ['v5'], 'review_timeout': 6, 'extra_params': {'start_date': '2024-08-01','end_date': '2025-02-28'},}),
])
```