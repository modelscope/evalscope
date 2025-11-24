# Sampling Your Index Data

In a nutshell: Based on the defined Collection Schema, sample multiple datasets using specified strategies to generate a mixed JSONL for unified one-time evaluation.

## Output
- A single JSONL file (mixed samples) for subsequent `evalscope eval`.
- Each line represents a standardized sample.

## Field Description
| Field | Description |
| ---- | ---- |
| index | Sample sequence number (renumbered in the mixed set) |
| prompt | Original input structure (may include question / context / code, etc.) |
| tags | Merged tags (dataset's own + upper hierarchy) |
| task_type | Task type or capability classification |
| weight | Leaf node normalized weight (for interpretation, not equal to actual sample repetition weight) |
| dataset_name | Original dataset name |
| subset_name | Subset or difficulty label (can be empty) |

Example:
```json
{
  "index": 0,
  "prompt": {"question": "What is the capital of France?"},
  "tags": ["en", "reasoning"],
  "task_type": "question_answering",
  "weight": 1.0,
  "dataset_name": "arc",
  "subset_name": "ARC-Easy"
}
```

## Three Sampling Strategies (Formulas & Use Cases)

Given:
- Total sample size: $N$
- Number of datasets: $K$
- Schema weights: $w_i$
- Original dataset sample size: $m_i$

### 1. Weighted Sampling
Formula (expected sample count):
$$
n_i \approx N \cdot \frac{w_i}{\sum_{j=1}^{K} w_j}
$$
**Use case**: You've expressed business priorities through weights and want "high-weight capabilities" to occupy more resources.  
**Characteristics**: Reinforces values; if a dataset has much higher weight, it will dominate proportionally.

### 2. Stratified Sampling
Formula:
$$
n_i \approx N \cdot \frac{m_i}{\sum_{j=1}^{K} m_j}
$$
**Use case**: Preserve original distribution (e.g., let datasets with larger corpus sizes be more representative).  
**Characteristics**: Doesn't reflect business bias; avoids over-amplifying small datasets. Guarantees at least ≥1 sample per dataset.

### 3. Uniform Sampling
Formula:
$$
n_i \approx \frac{N}{K}
$$
**Use case**: Want each capability to have "equal voice," facilitating model comparison across capabilities.  
**Characteristics**: Ignores original size and weights; makes results easier for comparing individual capability weaknesses.

**Strategy Selection Recommendation**:
- For "business decision index": Weighted
- For "objective coverage / preserving corpus structure": Stratified
- For "capability alignment / diagnosis": Uniform

## Prerequisite Schema Example
```python
from evalscope.collections import CollectionSchema, DatasetInfo  

schema = CollectionSchema(
    name='reasoning_index',
    datasets=[
        DatasetInfo(name='arc', weight=2.0, task_type='reasoning', tags=['en']),
        DatasetInfo(name='ceval', weight=3.0, task_type='reasoning', tags=['zh'], args={'subset_list': ['logic']}),
    ],
)
```

### Weighted Sampling (Emphasizing Business Importance)
```python
from evalscope.collections import WeightedSampler
from evalscope.utils.io_utils import dump_jsonl_data

sampler = WeightedSampler(schema)
mixed_data = sampler.sample(10)  # N=10
dump_jsonl_data(mixed_data, 'outputs/weighted_mixed_data.jsonl')
```
Expected: arc : ceval ≈ 2 : 3 → 4 : 6

### Stratified Sampling (Preserving Source Data Size Distribution)
```python
from evalscope.collections import StratifiedSampler
from evalscope.utils.io_utils import dump_jsonl_data

sampler = StratifiedSampler(schema)
mixed_data = sampler.sample(10)
dump_jsonl_data(mixed_data, 'outputs/stratified_mixed_data.jsonl')
```
Expected: Approximately allocated by original sample sizes (example: if arc=2000, ceval=10 → roughly 9 : 1)

### Uniform Sampling (Aligning Capability Comparison)
```python
from evalscope.collections import UniformSampler
from evalscope.utils.io_utils import dump_jsonl_data

sampler = UniformSampler(schema)
mixed_data = sampler.sample(10)
dump_jsonl_data(mixed_data, 'outputs/uniform_mixed_data.jsonl')
```
Expected: arc : ceval = 5 : 5


## Common Issues
- Weights don't affect Uniform / Stratified allocation (only Weighted uses them).
- Dataset with too few samples: Stratified sampling will force at least 1 sample; increase N or switch to Weighted.
- The `weight` field in JSONL is the leaf normalized weight, not equal to the sample's appearance probability (especially for Stratified and Uniform strategies).