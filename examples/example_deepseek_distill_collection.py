from evalscope.collections import CollectionSchema, DatasetInfo, WeightedSampler
from evalscope.utils.io_utils import dump_jsonl_data

# define the schema
schema = CollectionSchema(name='DeepSeekDistill', datasets=[
            CollectionSchema(name='Math', datasets=[
                DatasetInfo(name='math_500', weight=1, task_type='math', tags=['en'], args={'few_shot_num': 0}),
                DatasetInfo(name='gpqa_diamond', weight=1, task_type='math', tags=['en'],  args={'few_shot_num': 0}),
                DatasetInfo(name='aime24', weight=1, task_type='math', tags=['en'], args={'few_shot_num': 0}),
            ])
        ])

# get the mixed data
mixed_data = WeightedSampler(schema).sample(100000)  # set a large number to ensure all datasets are sampled
# dump the mixed data to a jsonl file
dump_jsonl_data(mixed_data, 'outputs/deepseek_distill_test.jsonl')

from evalscope import TaskConfig, run_task

#  start the task
from evalscope.constants import EvalType

task_cfg = TaskConfig(
    model='DeepSeek-R1-Distill-Qwen-1.5B',
    api_url='http://127.0.0.1:8801/v1/chat/completions',
    api_key='EMPTY',
    eval_type=EvalType.SERVICE,
    datasets=[
        'data_collection',
    ],
    dataset_args={
        'data_collection': {
            'local_path': 'outputs/deepseek_distill_test.jsonl'
        }
    },
    eval_batch_size=32,
    generation_config={
        'max_tokens': 30000,  # avoid exceed max length
        'temperature': 0.6,
        'top_p': 0.95,
        'n': 1,  # num of responses per sample, note that lmdeploy only supports n=1
    },
)

run_task(task_cfg=task_cfg)
