# Defining the Data Mixing Schema

The data mixing schema defines which datasets are used for evaluation and how the data is grouped. This is the first step in the mixed data evaluation process.

## Creating the Schema

An example of a data mixing schema (CollectionSchema) is shown below:

**Simple Example**

```python
from evalscope.collections import CollectionSchema, DatasetInfo

simple_schema = CollectionSchema(name='reasoning', datasets=[
    DatasetInfo(name='arc', weight=1, task_type='reasoning', tags=['en']),
    DatasetInfo(name='ceval', weight=1, task_type='reasoning', tags=['zh'], args={'subset_list': ['logic']})
])
```
Where:
- `name` is the name of the data mixing schema.
- `datasets` is a list of datasets, where each dataset (DatasetInfo) includes attributes such as `name`, `weight`, `task_type`, `tags`, and `args`.
    - `name` is the name of the dataset. Supported dataset names can be found in the [dataset list](../../get_started/supported_dataset.md#1-native-supported-datasets).
    - `weight` is the weight of the dataset, used for weighted sampling. The default is 1, and all data will be normalized during sampling.
    - `task_type` is the task type of the dataset and can be filled in as needed.
    - `tags` are labels for the dataset, which can also be filled in as needed.
    - `args` are parameters for the dataset, and the configurable parameters can be found in the [dataset parameters](../../get_started/parameters.md#dataset-parameters).

**Complex Example**

```python
complex_schema = CollectionSchema(name='math&reasoning', datasets=[
    CollectionSchema(name='math', weight=3, datasets=[
        DatasetInfo(name='gsm8k', weight=1, task_type='math', tags=['en']),
        DatasetInfo(name='competition_math', weight=1, task_type='math', tags=['en']),
        DatasetInfo(name='cmmlu', weight=1, task_type='math', tags=['zh'], args={'subset_list': ['college_mathematics', 'high_school_mathematics']}),
        DatasetInfo(name='ceval', weight=1, task_type='math', tags=['zh'], args={'subset_list': ['advanced_mathematics', 'high_school_mathematics', 'discrete_mathematics', 'middle_school_mathematics']}),
    ]),
    CollectionSchema(name='reasoning', weight=1, datasets=[
        DatasetInfo(name='arc', weight=1, task_type='reasoning', tags=['en']),
        DatasetInfo(name='ceval', weight=1, task_type='reasoning', tags=['zh'], args={'subset_list': ['logic']}),
        DatasetInfo(name='race', weight=1, task_type='reasoning', tags=['en']),
    ]),
])
```
- `weight` is the weight of the data mixing schema, used for weighted sampling. The default is 1, and all data will be normalized during sampling.
- `datasets` can contain CollectionSchema, enabling the nesting of datasets. During evaluation, the name of the `CollectionSchema` will be recursively added to the tags of each sample.

## Using the Schema

- To view the created schema:

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

- To view the flatten result of the schema (automatically normalized weights):

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

- To save the schema:

```python
schema.dump_json('outputs/schema.json')
```

- To load the schema from a JSON file:

```python
schema = CollectionSchema.from_json('outputs/schema.json')
```