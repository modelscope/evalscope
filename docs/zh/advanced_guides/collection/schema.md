# 定义你的 Schema

一句话：Schema = 你关心的能力清单 + 各能力的相对价值（权重） + 可选的层次分组。

## 核心概念与对象
- CollectionSchema: 可以嵌套的分组节点，形成层次结构与可复用模块。
- DatasetInfo: 最终可评测的数据集条目（叶子节点），包含名称、权重与适配参数。
- flatten(): 展开整棵树得到归一化后的叶子数据集列表（含层级路径 hierarchy）。

## 字段速览
CollectionSchema:
- name: str（必填）分组/指数名
- weight: float（可选，默认 1.0）给该子分组整体的相对权重（与子节点组合后再归一化）
- datasets: list[DatasetInfo | CollectionSchema]

DatasetInfo:
- name: str（必填）数据集注册名
- weight: float（必填 > 0）
- task_type: str（可选）自定义能力标签
- tags: list[str]（可选）自定义标签，父级 CollectionSchema 名会自动追加
- args: dict（可选）透传到数据集适配器（例如 subset_list、local_path、review_timeout 等）

## 为什么要设置权重
采样阶段:
- 在加权采样策略下，样本份额 ∝ 归一化权重。
- 平铺（无嵌套）时的归一化：
$$
\mathbf{p_i} = \frac{\mathbf{w_i}}{\sum_{j=1}^{n} \mathbf{w_j}}
$$
- 多层嵌套时（沿叶子路径上各层权重乘积后再整体归一化）：
$$
\mathbf{p_i} = \frac{\prod_{k \in \text{path}(i)} \mathbf{w_k}}{\sum_{j=1}^{n} \prod_{k \in \text{path}(j)} \mathbf{w_k}}
$$

评分阶段:
- 汇总使用线性加权：
$$
\mathbf{S} = \sum_{i=1}^{n} \boldsymbol{\alpha_i} \, \mathbf{s_i}
$$
其中：
$$
\boldsymbol{\alpha_i} = \mathbf{p_i}, \quad \sum_{i=1}^{n} \boldsymbol{\alpha_i} = 1
$$


## 最小 Python 示例
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


## 嵌套与多能力示例
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
层次效果:
- 父级名称追加到每个叶子 tags/hierarchy，支持多层聚合分析。
- 父级 weight 会对整组起缩放作用，再与子节点联合归一化。

## 展开 / 序列化 / 还原
```python
print(simple_schema.flatten())          # 展开 + 归一化
print(complex_schema.flatten())         # 查看嵌套归一化后权重
complex_schema.dump_json('outputs/complex_schema.json')
loaded = CollectionSchema.from_json('outputs/complex_schema.json')
```

输出示例（自动归一化权重）:
```python
>>> [DatasetInfo(name='arc', weight=0.4, task_type='reasoning', tags=['en'], args={}, hierarchy=['reasoning_index']),
     DatasetInfo(name='ceval', weight=0.6, task_type='reasoning', tags=['zh'], args={'subset_list': ['logic']}, hierarchy=['reasoning_index'])]

>>> [DatasetInfo(name='gsm8k', weight=0.375, task_type='math', tags=['en'], args={}, hierarchy=['math_index', 'math']),
     DatasetInfo(name='aime25', weight=0.375, task_type='math', tags=['en'], args={}, hierarchy=['math_index', 'math']),
     DatasetInfo(name='arc', weight=0.125, task_type='reasoning', tags=['en'], args={}, hierarchy=['math_index', 'reasoning']),
     DatasetInfo(name='ceval', weight=0.125, task_type='reasoning', tags=['zh'], args={'subset_list': ['logic']}, hierarchy=['math_index', 'reasoning'])]
```

## 常见问题
- 权重必须 > 0，=0 的条目请直接删除。
- 深层嵌套容易导致某些叶子权重被“稀释”，建议先手动展开验证。
- tags 过多会影响后续可视化维度，保持简短（如 en / zh / math / rag）。
- args 需与对应数据集适配器字段匹配，错误字段会被忽略或触发警告。
- 大量能力塞进一个 Schema 不利于迭代，建议拆分多个版本并对比。


## 案例参考

### 案例 A：企业级 RAG 助手指数
- 痛点：通用榜单分高，但内部知识检索不准。
- 思路：知识类 + 长文本检索集 + 指令遵循。
- 配置示例：
```python
from evalscope.collections import CollectionSchema, DatasetInfo

schema = CollectionSchema(name='rag_index', datasets=[
    DatasetInfo(name='chinese_simpleqa', weight=0.3, task_type='knowledge', tags=['rag']),
    DatasetInfo(name='aa_lcr', weight=0.3, task_type='long_context', tags=['rag']),
    DatasetInfo(name='ifeval', weight=0.4, task_type='instruction_following', tags=['rag']),
])
```


### 案例 B：代码补全与分析指数
- 痛点：通用榜单无法全面反映代码理解与生成能力。
- 思路：代码生成 + 真实issue。
- 配置示例：
```python
from evalscope.collections import CollectionSchema, DatasetInfo

schema = CollectionSchema(name='code_index', datasets=[
    DatasetInfo(name='humaneval', weight=2.0, task_type='code', tags=['python']),
    DatasetInfo(name='live_code_bench', weight=1.0, task_type='code', tags=['zh', 'finance'], args={'subset_list': ['v5'], 'review_timeout': 6, 'extra_params': {'start_date': '2024-08-01','end_date': '2025-02-28'},}),
])
```

