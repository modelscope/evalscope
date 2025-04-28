
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
            DatasetInfo(name='gpqa', weight=1, task_type='knowledge', tags=['en'], args={'subset_list': ['gpqa_diamond'], 'few_shot_num': 0})
        ])
    ])

    # schema2 =  CollectionSchema(name='knowledge', datasets=[
    #         DatasetInfo(name='simple_qa', weight=1, task_type='knowledge', tags=['en'], args={'few_shot_num': 0}),
    #         DatasetInfo(name='chinese_simpleqa', weight=1, task_type='knowledge', tags=['zh'], args={'few_shot_num': 0}),
    #     ])

    # get the mixed data
    mixed_data = WeightedSampler(schema).sample(100000000)  # set a large number to ensure all datasets are sampled
    # dump the mixed data to a jsonl file
    dump_jsonl_data(mixed_data, 'outputs/qwen3_test.jsonl')

def run_test():
    from evalscope import TaskConfig, run_task
    task_cfg = TaskConfig(
        model='Qwen3-32B',
        api_url='http://127.0.0.1:8801/v1/chat/completions',
        eval_type='service',
        datasets=[
            'data_collection',
        ],
        dataset_args={
            'data_collection': {
                'local_path': 'outputs/qwen3_test.jsonl',
                'filters': {'remove_until': '</think>'}
            }
        },
        eval_batch_size=256,
        generation_config={
            'max_tokens': 20000,  # 最大生成token数，建议设置为较大值避免输出截断
            'temperature': 0.6,  # 采样温度 (qwen 报告推荐值)
            'top_p': 0.95,  # top-p采样 (qwen 报告推荐值)
            'top_k': 40,  # top-k采样 (qwen 报告推荐值)
            'n': 1,  # 每个请求产生的回复数量
        },
        timeout=60000,
        stream=True,
        use_cache='outputs/20250427_234222'
        # judge_strategy=JudgeStrategy.AUTO,
        # judge_worker_num=1,
        # judge_model_args={
        #     'model_id': 'qwen2.5-7b-instruct',
        #     'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        #     'api_key': env.get('DASHSCOPE_API_KEY'),
        # },
    )

    run_task(task_cfg=task_cfg)

if __name__ == '__main__':
    run_test()
