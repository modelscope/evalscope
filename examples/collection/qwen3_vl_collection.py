from evalscope.collections import CollectionSchema, DatasetInfo, WeightedSampler
from evalscope.utils.io_utils import dump_jsonl_data

schema = CollectionSchema(name='Qwen3-VL', datasets=[
    CollectionSchema(name='PureText', weight=1, datasets=[
        DatasetInfo(name='mmlu_pro', weight=1, task_type='exam', tags=['en'], args={'few_shot_num': 0}),
        DatasetInfo(name='ifeval', weight=1, task_type='instruction', tags=['en'], args={'few_shot_num': 0}),
        DatasetInfo(name='gsm8k', weight=1, task_type='math', tags=['en'], args={'few_shot_num': 0}),
    ]),
    CollectionSchema(name='Vision', weight=2, datasets=[
        DatasetInfo(name='math_vista', weight=1, task_type='math', tags=['en'], args={'few_shot_num': 0}),
        DatasetInfo(name='mmmu_pro', weight=1, task_type='exam', tags=['en'], args={'few_shot_num': 0}),
    ])
])

# get the mixed data
mixed_data = WeightedSampler(schema).sample(100000000)  # set a large number to ensure all datasets are sampled
# dump the mixed data to a jsonl file
dump_jsonl_data(mixed_data, 'outputs/qwen3_vl_test.jsonl')
