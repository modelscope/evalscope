from dotenv import dotenv_values

env = dotenv_values('.env')

from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType, JudgeStrategy, ModelTask

task_config = TaskConfig(
    model_id='offline-model',
    model_task=ModelTask.IMAGE_GENERATION,  # 模型任务类型
    eval_type=EvalType.IMAGE_EDITING,  # 评测任务类型
    datasets=['gedit'],  # 使用的benchmark
    dataset_args={  # benmark的具体参数
        'gedit':{
            'subset_list': ['color_alter', 'material_alter'], # 选取评测的子集
            'extra_params':{
                'language': 'cn', # 使用中文的指令
                'local_file': 'outputs/example_edit.jsonl' # 使用本地已经生成的图片
            }
        }
    },
    eval_batch_size=1,
    limit=5,
    judge_strategy=JudgeStrategy.AUTO,
    judge_worker_num=5,
    judge_model_args={  # 需要配置一个VLM模型用于自动打分
        'model_id': 'qwen2.5-vl-72b-instruct',
        'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'api_key': env.get('DASHSCOPE_API_KEY'),
        'generation_config': {
            'temperature': 0.0,
            'max_tokens': 4096,
        }
    },
)

run_task(task_config)
