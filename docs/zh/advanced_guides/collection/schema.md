# 定义数据混合schema

数据混合schema定义使用哪些数据进行评测，以及数据如何分组，是数据混合评测的第一步。

## 创建schema

数据混合schema (CollectionSchema)示例如下：

**简单示例**

```python
from evalscope.collections import CollectionSchema, DatasetInfo

simple_schema = CollectionSchema(name='reasoning', datasets=[
    DatasetInfo(name='arc', weight=1, task_type='reasoning', tags=['en']),
    DatasetInfo(name='ceval', weight=1, task_type='reasoning', tags=['zh'], args={'subset_list': ['logic']})
])
```
其中：
- `name` 是数据混合schema的名称
- `datasets` 是数据集列表，每个数据集(DatasetInfo)包含 `name`、`weight`、`task_type`、`tags` 和 `args` 等属性。
    - `name` 是数据集的名称，支持的数据集名称见[数据集列表](../../get_started/supported_dataset.md#1-原生支持的数据集)
    - `weight` 是数据集的权重，类型为float，用于加权采样，默认为1.0，采样时所有数据会归一化（数值需要大于0）
    - `task_type` 是数据集的任务类型，可自行填写
    - `tags` 是数据集的标签，可自行填写
    - `args` 是数据集的参数，可指定的参数见[数据集参数](../../get_started/parameters.md#数据集参数)
    - `hierarchy` 是数据集的层次，由schema自动生成

**复杂示例**

```python
complex_schema = CollectionSchema(name='math&reasoning', datasets=[
    CollectionSchema(name='math', weight=3, datasets=[
        DatasetInfo(name='gsm8k', weight=1, task_type='math', tags=['en']),
        DatasetInfo(name='competition_math', weight=1, task_type='math', tags=['en']),
        DatasetInfo(name='cmmlu', weight=1, task_type='math_examination', tags=['zh'], args={'subset_list': ['college_mathematics', 'high_school_mathematics']}),
        DatasetInfo(name='ceval', weight=1, task_type='math_examination', tags=['zh'], args={'subset_list': ['advanced_mathematics', 'high_school_mathematics', 'discrete_mathematics', 'middle_school_mathematics']}),
    ]),
    CollectionSchema(name='reasoning', weight=1, datasets=[
        DatasetInfo(name='arc', weight=1, task_type='reasoning', tags=['en']),
        DatasetInfo(name='ceval', weight=1, task_type='reasoning_examination', tags=['zh'], args={'subset_list': ['logic']}),
        DatasetInfo(name='race', weight=1, task_type='reasoning', tags=['en']),
    ]),
])
```
- `weight` 是数据混合schema的权重，类型为float，用于加权采样，默认为1.0，采样时所有数据会归一化（数值需要大于0）
- `datasets` 中可以包含CollectionSchema，从而实现数据集的嵌套；在评测时，`CollectionSchema`的名称会递归添加到每个样本的tag中

## 使用schema

- 查看创建的schema:

```python
print(simple_schema)
```
```json
{
    "name": "reasoning",
    "datasets": [
        {
            "name": "arc",
            "weight": 1,
            "task_type": "reasoning",
            "tags": [
                "en",
                "reasoning"
            ],
            "args": {}
        },
        {
            "name": "ceval",
            "weight": 1,
            "task_type": "reasoning",
            "tags": [
                "zh",
                "reasoning"
            ],
            "args": {
                "subset_list": [
                    "logic"
                ]
            }
        }
    ]
}
```

- 查看schema的flatten结果（自动归一化权重）:

```python
print(complex_schema.flatten())
```
```text
DatasetInfo(name='gsm8k', weight=0.1875, task_type='math', tags=['en', 'math&reasoning', 'math'], args={})
DatasetInfo(name='competition_math', weight=0.1875, task_type='math', tags=['en', 'math&reasoning', 'math'], args={})
DatasetInfo(name='cmmlu', weight=0.1875, task_type='math', tags=['zh', 'math&reasoning', 'math'], args={'subset_list': ['college_mathematics', 'high_school_mathematics']})
DatasetInfo(name='ceval', weight=0.1875, task_type='math', tags=['zh', 'math&reasoning', 'math'], args={'subset_list': ['advanced_mathematics', 'high_school_mathematics', 'discrete_mathematics', 'middle_school_mathematics']})
DatasetInfo(name='arc', weight=0.08333333333333333, task_type='reasoning', tags=['en', 'math&reasoning', 'reasoning'], args={})
DatasetInfo(name='ceval', weight=0.08333333333333333, task_type='reasoning', tags=['zh', 'math&reasoning', 'reasoning'], args={'subset_list': ['logic']})
DatasetInfo(name='race', weight=0.08333333333333333, task_type='reasoning', tags=['en', 'math&reasoning', 'reasoning'], args={})
```

- 保存schema:

```python
schema.dump_json('outputs/schema.json')
```

- 从json文件中加载schema:

```python
schema = CollectionSchema.from_json('outputs/schema.json')
```
