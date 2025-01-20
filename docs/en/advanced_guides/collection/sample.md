# Sampling Data

In mixed data evaluation, sampling data is the second step, and currently, three sampling methods are supported: weighted sampling, stratified sampling, and uniform sampling.

## Data Format

The sampled data format is JSON Lines (jsonl), where each line is a JSON object containing properties such as `index`, `prompt`, `tags`, `task_type`, `weight`, `dataset_name`, and `subset_name`.

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

## Weighted Sampling

Weighted sampling is based on the weights of the datasets. The larger the weight, the more samples are taken from that dataset. For nested schemas, the weighted sampling scales according to the weight of each schema, ensuring that the total weight of all datasets sums to 1.

For example, if a total of 100 samples are to be taken, and there are two datasets in the schema, with Dataset A having a weight of 3 and Dataset B having a weight of 1, then Dataset A will have 75 samples and Dataset B will have 25 samples.

```python
from evalscope.collections import WeightedSampler
from evalscope.utils.io_utils import dump_jsonl_data

sampler = WeightedSampler(schema)
mixed_data = sampler.sample(100)
dump_jsonl_data(mixed_data, 'outputs/weighted_mixed_data.jsonl')
```

## Stratified Sampling

Stratified sampling takes samples based on the number of samples in each dataset in the schema, with the number of samples taken from each dataset being proportional to its number of samples.

For example, if a total of 100 samples are to be taken, and there are two datasets in the schema, with Dataset A having 800 samples and Dataset B having 200 samples, then Dataset A will have 80 samples and Dataset B will have 20 samples.

```python
from evalscope.collections import StratifiedSampler

sampler = StratifiedSampler(schema)
mixed_data = sampler.sample(100)
dump_jsonl_data(mixed_data, 'outputs/stratified_mixed_data.jsonl')
```

## Uniform Sampling

Uniform sampling takes the same number of samples from each dataset in the schema.

For example, if a total of 100 samples are to be taken, and there are two datasets in the schema, with Dataset A having 800 samples and Dataset B having 200 samples, then both Dataset A and Dataset B will have 50 samples each.

```python
from evalscope.collections import UniformSampler

sampler = UniformSampler(schema)
mixed_data = sampler.sample(100)
dump_jsonl_data(mixed_data, 'outputs/uniform_mixed_data.jsonl')
```
