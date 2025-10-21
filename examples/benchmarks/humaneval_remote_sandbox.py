from dotenv import dotenv_values

env = dotenv_values('.env')
from evalscope import TaskConfig, run_task

task_config = TaskConfig(
    model='qwen-plus',
    datasets=['humaneval'],
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=env.get('DASHSCOPE_API_KEY'),
    eval_type='openai_api',
    eval_batch_size=5,
    limit=5,
    generation_config={
        'max_tokens': 4096,
        'temperature': 0.0,
        'seed': 42,
    },
    use_sandbox=True, # enable sandbox
    sandbox_type='docker', # specify sandbox type
    sandbox_manager_config={
        'base_url': 'http://127.0.0.1:1234'  # remote sandbox manager URL
    }
)

run_task(task_config)
