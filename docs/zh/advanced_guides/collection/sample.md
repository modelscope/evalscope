# 采样数据

在数据混合评测中，采样数据是数据混合评测的第二步，目前支持加权采样、分层采样、均匀采样三种采样方式。

## 数据格式

采样数据格式为jsonl，每行是一个json对象，包含`index`、`prompt`、`tags`、`task_type`、`weight`、`dataset_name`、`subset_name`等属性。

```json
{
    "index": 0,
    "prompt": {"question": "What is the capital of France?"},
    "tags": ["en", "reasoning"],
    "task_type": "question_answering",
    "weight": 1.0,
    "dataset_name": "arc",
    "subset_name": "ARC-Easy",
}
```

## 加权采样

加权采样是根据数据集的权重进行采样，权重越大，采样数据越多。对于嵌套的schema，加权采样会根据每个schema的权重进行缩放，最终所有数据集的权重和为1。

例如，整体采样100条数据，schema中有两个数据集，数据集A的权重为3，数据集B的权重为1，那么数据集A的采样数量为75条，数据集B的采样数量为25条。

```python
from evalscope.collections import WeightedSampler
from evalscope.utils.io_utils import dump_jsonl_data

sampler = WeightedSampler(schema)
mixed_data = sampler.sample(100)
dump_jsonl_data(mixed_data, 'outputs/weighted_mixed_data.jsonl')
```

## 分层采样

分层采样是根据schema中每个数据集自身样本的数量进行采样，每个数据集的采样数量与自身样本数量成正比。

例如，整体采样100条数据，schema中有两个数据集，数据集A有800条数据，数据集B有200条数据，那么数据集A的采样数量为80条，数据集B的采样数量为20条。

```python
from evalscope.collections import StratifiedSampler

sampler = StratifiedSampler(schema)
mixed_data = sampler.sample(100)
dump_jsonl_data(mixed_data, 'outputs/stratified_mixed_data.jsonl')
```

## 均匀采样

均匀采样是根据schema中每个数据集的个数进行采样，每个数据集的采样数量相同。

例如，整体采样100条数据，schema中有两个数据集，数据集A有800条数据，数据集B有200条数据，那么数据集A的采样数量为50条，数据集B的采样数量为50条。

```python
from evalscope.collections import UniformSampler

sampler = UniformSampler(schema)
mixed_data = sampler.sample(100)
dump_jsonl_data(mixed_data, 'outputs/uniform_mixed_data.jsonl')
```
