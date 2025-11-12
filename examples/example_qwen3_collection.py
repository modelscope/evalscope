
def generate_collection():
    from evalscope.collections import CollectionSchema, DatasetInfo, WeightedSampler
    from evalscope.utils.io_utils import dump_jsonl_data

    schema = CollectionSchema(name='Qwen3', datasets=[
        CollectionSchema(name='English', datasets=[
            DatasetInfo(name='mmlu_pro', weight=1, task_type='exam', tags=['en'], args={'few_shot_num': 0}),
            DatasetInfo(name='mmlu_redux', weight=1, task_type='exam', tags=['en'], args={'few_shot_num': 0}),
            DatasetInfo(name='ifeval', weight=1, task_type='instruction', tags=['en'], args={'few_shot_num': 0}),
        ]),
        CollectionSchema(name='Chinese', datasets=[
            DatasetInfo(name='ceval', weight=1, task_type='exam', tags=['zh'], args={'few_shot_num': 0}),
            DatasetInfo(name='iquiz', weight=1, task_type='exam', tags=['zh'], args={'few_shot_num': 0}),
        ]),
        CollectionSchema(name='Code', datasets=[
            DatasetInfo(name='live_code_bench', weight=1, task_type='code', tags=['en'], args={'few_shot_num': 0, 'subset_list': ['v5_v6'], 'extra_params': {'start_date': '2025-01-01', 'end_date': '2025-04-30'}}),
        ]),
        CollectionSchema(name='Math&Science', datasets=[
            DatasetInfo(name='math_500', weight=1, task_type='math', tags=['en'], args={'few_shot_num': 0}),
            DatasetInfo(name='aime24', weight=1, task_type='math', tags=['en'], args={'few_shot_num': 0}),
            DatasetInfo(name='aime25', weight=1, task_type='math', tags=['en'], args={'few_shot_num': 0}),
            DatasetInfo(name='gpqa_diamond', weight=1, task_type='knowledge', tags=['en'], args={'few_shot_num': 0})
        ])
    ])

    # get the mixed data
    mixed_data = WeightedSampler(schema).sample(100000000)  # set a large number to ensure all datasets are sampled
    # dump the mixed data to a jsonl file
    dump_jsonl_data(mixed_data, 'outputs/qwen3_test.jsonl')

def run_test_think():
    from evalscope import TaskConfig, run_task
    task_cfg = TaskConfig(
        model='Qwen3-32B',
        api_url='http://127.0.0.1:8801/v1/chat/completions',
        eval_type='openai_api',
        datasets=[
            'data_collection',
        ],
        dataset_args={
            'data_collection': {
                'dataset_id': 'evalscope/Qwen3-Test-Collection',
                'filters': {'remove_until': '</think>'}  # 过滤掉思考的内容
            }
        },
        eval_batch_size=128,
        generation_config={
            'max_tokens': 30000,  # 最大生成token数，建议设置为较大值避免输出截断
            'temperature': 0.6,  # 采样温度 (qwen 报告推荐值)
            'top_p': 0.95,  # top-p采样 (qwen 报告推荐值)
            'top_k': 20,  # top-k采样 (qwen 报告推荐值)
            'n': 1,  # 每个请求产生的回复数量
        },
        timeout=60000,  # 超时时间
        stream=True,  # 是否使用流式输出
        limit=100,  # 设置为100条数据进行测试
    )

    run_task(task_cfg=task_cfg)

def run_test_no_think():
    from evalscope import TaskConfig, run_task
    task_cfg = TaskConfig(
        model='Qwen3-32B-no-think',
        api_url='http://127.0.0.1:8801/v1/chat/completions',
        eval_type='openai_api',
        datasets=[
            'data_collection',
        ],
        dataset_args={
            'data_collection': {
                'dataset_id': 'evalscope/Qwen3-Test-Collection',
            }
        },
        eval_batch_size=128,
        generation_config={
            'max_tokens': 10000,  # 最大生成token数，建议设置为较大值避免输出截断
            'temperature': 0.7,  # 采样温度 (qwen 报告推荐值)
            'top_p': 0.8,  # top-p采样 (qwen 报告推荐值)
            'top_k': 20,  # top-k采样 (qwen 报告推荐值)
            'n': 1,  # 每个请求产生的回复数量
            'extra_body':{'chat_template_kwargs': {'enable_thinking': False}}  # 关闭思考模式
        },
        judge_worker_num=1,
        timeout=60000,  # 超时时间
        stream=True,  # 是否使用流式输出
        limit=10,  # 设置为1000条数据进行测试
    )

    run_task(task_cfg=task_cfg)

def run_math_thinking():
    from evalscope import TaskConfig, run_task
    task_cfg = TaskConfig(
        model='Qwen3-32B',
        api_url='http://127.0.0.1:8801/v1/chat/completions',
        eval_type='openai_api',
        datasets=[
            'math_500',
        ],
        dataset_args={
            'math_500': {
                'filters': {'remove_until': '</think>'}
            }
        },
        eval_batch_size=128,
        generation_config={
            'max_tokens': 30000,  # 最大生成token数，建议设置为较大值避免输出截断
            'temperature': 0.6,  # 采样温度 (qwen 报告推荐值)
            'top_p': 0.95,  # top-p采样 (qwen 报告推荐值)
            'top_k': 20,  # top-k采样 (qwen 报告推荐值)
            'n': 1,  # 每个请求产生的回复数量
        },
        timeout=60000,
        stream=True
        # use_cache='outputs/20250427_234222'
    )
    run_task(task_cfg=task_cfg)

if __name__ == '__main__':
    # generate_collection()
    # run_test_think()
    # run_math_thinking()
    run_test_no_think()
