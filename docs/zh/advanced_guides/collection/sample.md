# 采样你的指数数据
在 EvalScope 中，**Collection** 允许你根据业务场景定义“专属指数”，通过配置数据集和权重，采样出最能代表你需求的混合数据集。以下是如何操作的详细指南。

## 数据格式
采样得到的混合数据集是一个 JSONL 文件，每行代表一个样本，包含以下字段：
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

## 如何选择采样策略
- **加权采样 (Weighted)**：按 Schema 权重抽样。资源集中到更重要能力。
- **分层采样 (Stratified)**：按各数据集原始样本量比例，保持原分布。
- **均匀采样 (Uniform)**：各数据集样本数相同，适合“能力对齐”对比。

### 加权采样
```python
from evalscope.collections import WeightedSampler
from evalscope.utils.io_utils import dump_jsonl_data

sampler = WeightedSampler(schema)
mixed_data = sampler.sample(100)
dump_jsonl_data(mixed_data, 'outputs/weighted_mixed_data.jsonl')
```

### 分层采样
```python
from evalscope.collections import StratifiedSampler
from evalscope.utils.io_utils import dump_jsonl_data

sampler = StratifiedSampler(schema)
mixed_data = sampler.sample(100)
dump_jsonl_data(mixed_data, 'outputs/stratified_mixed_data.jsonl')
```

### 均匀采样
```python
from evalscope.collections import UniformSampler
from evalscope.utils.io_utils import dump_jsonl_data

sampler = UniformSampler(schema)
mixed_data = sampler.sample(100)
dump_jsonl_data(mixed_data, 'outputs/uniform_mixed_data.jsonl')
```
