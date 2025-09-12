from evalscope.collections import CollectionSchema, DatasetInfo, WeightedSampler
from evalscope.utils.io_utils import dump_jsonl_data

schema = CollectionSchema(name='Qwen3-VL', datasets=[
    CollectionSchema(name='PureText', weight=1, datasets=[
        DatasetInfo(name='mmlu_pro', weight=1, task_type='exam', tags=['en'], args={'few_shot_num': 0}),
        DatasetInfo(name='ifeval', weight=1, task_type='instruction', tags=['en'], args={'few_shot_num': 0}),
        DatasetInfo(name='gsm8k', weight=1, task_type='math', tags=['en'], args={'few_shot_num': 0}),
    ]),
    CollectionSchema(name='Vision', weight=1.5, datasets=[
        DatasetInfo(name='math_vista', weight=1, task_type='math', tags=['en'], args={'few_shot_num': 0}),
        DatasetInfo(name='mmmu_pro', weight=2, task_type='exam', tags=['en'], args={'few_shot_num': 0}),
    ])
])

# get the mixed data
mixed_data = WeightedSampler(schema).sample(1000)
# dump the mixed data to a jsonl file
dump_jsonl_data(mixed_data, 'outputs/qwen3_vl_test.jsonl')


from dotenv import dotenv_values

env = dotenv_values('.env')
from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType

task_cfg = TaskConfig(
    model='qwen-vl-plus-latest',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=env.get('DASHSCOPE_API_KEY'),
    eval_type=EvalType.SERVICE,
    datasets=[
        'data_collection',
    ],
    dataset_args={
        'data_collection': {
            'dataset_id': 'evalscope/Qwen3-VL-Test-Collection',
        }
    },
    eval_batch_size=5,
    generation_config={
        'max_tokens': 30000,  # 最大生成token数，建议设置为较大值避免输出截断
        'temperature': 0.6,  # 采样温度 (qwen 报告推荐值)
        'top_p': 0.95,  # top-p采样 (qwen 报告推荐值)
        'top_k': 20,  # top-k采样 (qwen 报告推荐值)
        'n': 1,  # 每个请求产生的回复数量
    },
    limit=100,  # 设置为100条数据进行测试
)

run_task(task_cfg=task_cfg)
