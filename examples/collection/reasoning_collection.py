from evalscope.collections import CollectionSchema, DatasetInfo, WeightedSampler
from evalscope.utils.io_utils import dump_jsonl_data

schema = CollectionSchema(name='DeepSeekDistill', datasets=[
            CollectionSchema(name='Math', datasets=[
                    DatasetInfo(name='math_500', weight=1, task_type='math', tags=['en'], args={'few_shot_num': 0}),
                    DatasetInfo(name='gpqa_diamond', weight=1, task_type='math', tags=['en'],  args={'few_shot_num': 0}),
                    DatasetInfo(name='aime24', weight=1, task_type='math', tags=['en'], args={'few_shot_num': 0}),
            ])
        ])

#  get the mixed data
mixed_data = WeightedSampler(schema).sample(100000)  # set a large number to ensure all datasets are sampled
dump_jsonl_data(mixed_data, 'test.jsonl')
