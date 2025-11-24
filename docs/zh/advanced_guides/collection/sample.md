# 采样你的指数数据

一句话：基于已定义的 Collection Schema，将多个数据集按指定策略抽样生成一个混合 JSONL，用于一次性统一评测。

## 产出物
- 单一 JSONL 文件（混合样本），用于后续 `evalscope eval`。
- 每行代表一个标准化后的样本。

## 字段说明
| 字段 | 说明 |
| ---- | ---- |
| index | 样本序号（在混合集中重新编号） |
| prompt | 原始输入结构（可能包含 question / context / code 等） |
| tags | 合并的标签（数据集自身 + 上层层级） |
| task_type | 任务类型或能力分类 |
| weight | 叶子节点归一化权重（用于解释，不等于实际样本重复权重） |
| dataset_name | 原始数据集名称 |
| subset_name | 子集或难度标签（可为空） |

示例：
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

## 三种采样策略（公式与适用场景）

设：
- 总采样量：$N$
- 数据集数：$K$
- Schema 权重：$w_i$
- 数据集原始样本量：$m_i$

### 1. 加权采样 (Weighted)
公式（期望样本数）：
$$
n_i \approx N \cdot \frac{w_i}{\sum_{j=1}^{K} w_j}
$$
适用：你已经用权重表达业务优先级，希望“高权重能力”占更多资源。  
特点：强化价值观；若某数据集权重远大，会显著占比。

### 2. 分层采样 (Stratified)
公式：
$$
n_i \approx N \cdot \frac{m_i}{\sum_{j=1}^{K} m_j}
$$
适用：保留原始分布（例如让语料规模大的数据集更具代表性）。  
特点：不体现业务倾向；避免过度放大小数据集。至少保证每个数据集 ≥1 样本。

### 3. 均匀采样 (Uniform)
公式：
$$
n_i \approx \frac{N}{K}
$$
适用：希望各能力“平均发声”，便于模型在各能力上的对齐对比。  
特点：忽略原始规模与权重；让结果更容易比较单项能力短板。

策略选择建议：
- 做“业务决策指数”：Weighted
- 做“客观覆盖 / 保持语料结构”：Stratified
- 做“能力对齐 / 诊断”：Uniform

## 前置示例 Schema
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

### 加权采样（突出业务重要性）
```python
from evalscope.collections import WeightedSampler
from evalscope.utils.io_utils import dump_jsonl_data

sampler = WeightedSampler(schema)
mixed_data = sampler.sample(10)  # N=10
dump_jsonl_data(mixed_data, 'outputs/weighted_mixed_data.jsonl')
```
预期：arc : ceval ≈ 2 : 3 → 4 : 6

### 分层采样（保持源数据规模分布）
```python
from evalscope.collections import StratifiedSampler
from evalscope.utils.io_utils import dump_jsonl_data

sampler = StratifiedSampler(schema)
mixed_data = sampler.sample(10)
dump_jsonl_data(mixed_data, 'outputs/stratified_mixed_data.jsonl')
```
预期：按各数据集原始样本量近似分配（示例：若 arc=2000, ceval=10 → 约 9 : 1）

### 均匀采样（对齐能力对比）
```python
from evalscope.collections import UniformSampler
from evalscope.utils.io_utils import dump_jsonl_data

sampler = UniformSampler(schema)
mixed_data = sampler.sample(10)
dump_jsonl_data(mixed_data, 'outputs/uniform_mixed_data.jsonl')
```
预期：arc : ceval = 5 : 5


## 常见问题
- 权重不影响 Uniform / Stratified 分配（仅 Weighted 使用）。
- 某数据集样本过少：分层采样会强制至少 1 条，继续扩大 N 或改用 Weighted。
- JSONL 中的 weight 字段为叶子归一化权重，不等于该样本出现概率（分层与均匀策略下尤是）。
