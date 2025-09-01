from dotenv import dotenv_values

env = dotenv_values('.env')

from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType, JudgeStrategy, ModelTask

task_config = TaskConfig(
    model='Qwen/Qwen-Image-Edit', # 模型ID 或 本地路径
    model_args={
        'pipeline_cls': 'QwenImageEditPipeline',  #  在diffusers中的pipeline类
        'precision': 'bfloat16',  # 模型精度
        'device_map': 'cuda:2'  # 设备映射，Qwen Image Edit需要大概60G显存
    },
    model_task=ModelTask.IMAGE_GENERATION,  # 模型任务类型
    eval_type=EvalType.IMAGE_EDITING,  # 评测任务类型
    generation_config={  # 推理参数
        'true_cfg_scale': 4.0,
        'num_inference_steps': 50,
        'negative_prompt': ' ',
    },
    datasets=['gedit'],  # 使用的benchmark
    dataset_args={  # benmark的具体参数
        'gedit':{
            'subset_list': ['color_alter', 'material_alter'], # 选取评测的子集
            'extra_params':{
                'language': 'cn', # 使用中文的指令
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
