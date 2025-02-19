from evalscope import TaskConfig, run_task


def eval_distill_qwen():
    model_name = 'DeepSeek-R1-Distill-Qwen-7B'
    dataset_name = 'math_500'
    subsets = ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5']

    task_config = TaskConfig(
        api_url='http://0.0.0.0:8801/v1/chat/completions',
        model=model_name,
        eval_type='service',
        datasets=[dataset_name],
        dataset_args={dataset_name: {'few_shot_num': 0, 'subset_list': subsets}},
        eval_batch_size=32,
        generation_config={
            'max_tokens': 20000,  # avoid exceed max length
            'temperature': 0.6,
            'top_p': 0.95,
            'n': 3,
        },
    )
    run_task(task_config)


def eval_math_qwen():
    model_name = 'Qwen2.5-Math-7B-Instruct'
    dataset_name = 'math_500'
    subsets = ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5']

    task_config = TaskConfig(
        api_url='http://0.0.0.0:8801/v1/chat/completions',
        model=model_name,
        eval_type='service',
        datasets=[dataset_name],
        dataset_args={dataset_name: {'few_shot_num': 0, 'subset_list': subsets}},
        eval_batch_size=32,
        generation_config={
            'max_tokens': 3000,  # avoid exceed max length
            'temperature': 0.6,
            'top_p': 0.95,
            'n': 3,
        },
    )
    run_task(task_config)

if __name__ == '__main__':
    # eval_distill_qwen()
    eval_math_qwen()
