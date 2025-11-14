import os

from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='qwen-plus',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    eval_type='openai_api',  # 使用API模型服务
    datasets=['swe_bench_verified'], # 选择评测数据集, 也可以选择'swe_bench_verified_mini'或'swe_bench_lite'
    dataset_args={
        'swe_bench_verified': {
            'extra_params': {
                'build_docker_images': True, # 是否构建评测所需的Docker镜像, 首次运行建议设置为True
                'pull_remote_images_if_available': True, # 如果远程有可用镜像则拉取, 建议设置为True
                'inference_dataset_id': 'princeton-nlp/SWE-bench_oracle' # 选择推理数据集, 可选 'princeton-nlp/SWE-bench_bm25_13K', 'princeton-nlp/SWE-bench_bm25_27K', 'princeton-nlp/SWE-bench_bm25_40K' 或 'princeton-nlp/SWE-bench_oracle'
            }
        }
    },
    eval_batch_size=5,  # 推理时的批处理大小
    judge_worker_num=5, # 并行评测任务的工作线程数, docker container数量
    limit=5,  # 限制评测数量，便于快速测试，正式评测时建议去掉此项
    generation_config={
        'temperature': 0.1,
    }
)
run_task(task_cfg=task_cfg)