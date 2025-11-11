from dotenv import dotenv_values, load_dotenv

load_dotenv('.env')
env = dotenv_values('.env')

def run_test(model_name: str):
    from evalscope import TaskConfig, run_task
    task_cfg = TaskConfig(
        model=model_name,
        api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
        api_key=env.get('DASHSCOPE_API_KEY'),
        eval_type='openai_api',
        datasets=[
            'gsm8k',
            'iquiz',
            'ifeval',
            'humaneval',
            'gpqa_diamond',
        ],
        generation_config={
            'temperature': 0.7,
        },
        eval_batch_size=10,
        limit=10,  # 设置为10条数据进行测试
        use_cache=f'examples/viz/{model_name}' # 缓存路径
    )

    run_task(task_cfg=task_cfg)

run_test('qwen2.5-14b-instruct')
run_test('qwen2.5-7b-instruct')
