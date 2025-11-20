# 定义你的 Schema

## 什么是 Schema？

Schema 决定“测试什么”和“更看重什么”。在 EvalScope 中：
- **CollectionSchema**：一个可递归的集合节点，便于分组与复用。
- **DatasetInfo**：一个具体数据集的条目，包含名称、权重、任务类型、标签与参数。

## 权重的数学含义
- 采样阶段：权重越大，被抽到的样本越多。对嵌套 Schema，会先逐层缩放，最终归一化为 p_i = w_i / Σw_i。
- 评分阶段：将各数据集的得分做一次线性加权，S = Σ(α_i · s_i)，其中 α_i 是 Schema 的归一化权重，s_i 是该数据集的评测分数。
- 一句话：设置**权重**就是设置你的**价值观**。


## 创建 Schema

### 简单示例
```python
from evalscope.collections import CollectionSchema, DatasetInfo

simple_schema = CollectionSchema(name='reasoning_index', datasets=[
    DatasetInfo(name='arc', weight=1.0, task_type='reasoning', tags=['en']),
    DatasetInfo(name='ceval', weight=1.0, task_type='reasoning', tags=['zh'], args={'subset_list': ['logic']}),
])
```
说明：
- name：Schema 名称。
- datasets：由 DatasetInfo 或子 Schema 组成。
  - name：数据集名称（支持列表见“支持数据集”）。
  - weight：浮点数，采样时归一化（需大于 0）。
  - task_type：可选，自定义任务类型标签。
  - tags：可选，自定义标签，支持嵌套 Schema 自动追加上层名称。
  - args：透传到数据集适配器的参数（如 subset_list、local_path 等）。

### 复杂示例（嵌套 + 多组能力）
```python
complex_schema = CollectionSchema(name='math&reasoning', datasets=[
    CollectionSchema(name='math', weight=3.0, datasets=[
        DatasetInfo(name='gsm8k', weight=1.0, task_type='math', tags=['en']),
        DatasetInfo(name='competition_math', weight=1.0, task_type='math', tags=['en']),
        DatasetInfo(name='cmmlu', weight=1.0, task_type='math_examination', tags=['zh'], args={'subset_list': ['college_mathematics', 'high_school_mathematics']}),
        DatasetInfo(name='ceval', weight=1.0, task_type='math_examination', tags=['zh'], args={'subset_list': ['advanced_mathematics', 'high_school_mathematics', 'discrete_mathematics', 'middle_school_mathematics']}),
    ]),
    CollectionSchema(name='reasoning', weight=1.0, datasets=[
        DatasetInfo(name='arc', weight=1.0, task_type='reasoning', tags=['en']),
        DatasetInfo(name='ceval', weight=1.0, task_type='reasoning_examination', tags=['zh'], args={'subset_list': ['logic']}),
        DatasetInfo(name='race', weight=1.0, task_type='reasoning', tags=['en']),
    ]),
])
```
提示：
- 嵌套会把上层 CollectionSchema 的名称递归追加到 tags 中，便于后续 tag_level 视图聚合。
- 不需要的能力请直接不纳入 Schema（采样器要求权重大于 0）。

## 查看与落盘
```python
print(simple_schema)              # 查看结构
print(complex_schema.flatten())   # 展开并查看归一化后的权重
complex_schema.dump_json('outputs/schema.json')  # 保存
loaded = CollectionSchema.from_json('outputs/schema.json')  # 加载
```

## 案例参考

### 案例 A：企业级 **RAG 助手指数**
- 痛点：通用榜单分高，但内部知识检索不准。
- 思路：知识类 + 推理类 + 自有长文本检索集，按业务占比加权。
- 配置示例：
```python
from evalscope.collections import CollectionSchema, DatasetInfo, WeightedSampler
from evalscope.utils.io_utils import dump_jsonl_data

schema = CollectionSchema(name='rag_index', datasets=[
    DatasetInfo(name='cmmlu', weight=0.3, task_type='knowledge', tags=['zh', 'rag'], args={'subset_list': ['finance', 'enterprise_management']}),
    DatasetInfo(name='ceval', weight=0.2, task_type='reasoning', tags=['zh', 'rag'], args={'subset_list': ['logic']}),
    # 自定义长文本/检索集，建议提供本地 JSONL 或已适配的 DataAdapter
    DatasetInfo(name='your_long_text_rag', weight=0.5, task_type='retrieval', tags=['internal', 'rag'], args={'local_path': 'data/rag_qa.jsonl'}),
])
```


### 案例 B：代码补全与分析指数
- 痛点：通用榜单无法全面反映代码理解与生成能力。
- 思路：代码理解 + 生成 + 真实场景。
- 配置示例：
```python
from evalscope.collections import CollectionSchema, DatasetInfo, WeightedSampler
from evalscope.utils.io_utils import dump_jsonl_data

schema = CollectionSchema(name='code_index', datasets=[
    DatasetInfo(name='cmmlu', weight=2.0, task_type='finance', tags=['zh', 'finance'], args={'subset_list': ['finance', 'accounting']}),
    DatasetInfo(name='ceval', weight=1.0, task_type='reasoning', tags=['zh', 'finance'], args={'subset_list': ['logic']}),

])
```

