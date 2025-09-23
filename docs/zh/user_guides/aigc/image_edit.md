# 图片编辑评测

图片编辑任务是对图像内容的理解和生成能力的评测，输入一个图片和一个指令，模型根据指令生成相应的图像。EvalScope框架支持多种图片编辑模型的评测，包括Qwen-Image-Edit、FLUX.1-Kontext等。用户可以通过EvalScope框架对这些模型进行评测，获取模型的性能指标。

## 支持的评测数据集

请参考[文档](../../get_started/supported_dataset/aigc.md#aigc评测集)，请根据`ImageEditing`标签进行查找。

## 安装依赖

用户可以通过以下命令安装相关依赖：

```bash
pip install evalscope[aigc] -U
```

## 端到端评测

用户可以通过以下命令配置图片编辑模型，一键完成模型下载、数据集下载、模型推理、自动评测。

以`Qwen-Image-Edit`模型和`gedit`评测基准为例：

```python
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
```

输出示例如下：
```text
+-----------------+-----------+----------------------+-------------------+-------+---------+---------+
| Model           | Dataset   | Metric               | Subset            |   Num |   Score | Cat.0   |
+=================+===========+======================+===================+=======+=========+=========+
| Qwen-Image-Edit | gedit     | Semantic Consistency | background_change |     5 |  8      | default |
+-----------------+-----------+----------------------+-------------------+-------+---------+---------+
| Qwen-Image-Edit | gedit     | Semantic Consistency | color_alter       |     5 |  8      | default |
+-----------------+-----------+----------------------+-------------------+-------+---------+---------+
| Qwen-Image-Edit | gedit     | Semantic Consistency | material_alter    |     5 |  6.4    | default |
+-----------------+-----------+----------------------+-------------------+-------+---------+---------+
| Qwen-Image-Edit | gedit     | Semantic Consistency | motion_change     |     5 |  5.8    | default |
+-----------------+-----------+----------------------+-------------------+-------+---------+---------+
| Qwen-Image-Edit | gedit     | Semantic Consistency | ps_human          |     5 |  6.4    | default |
+-----------------+-----------+----------------------+-------------------+-------+---------+---------+
| Qwen-Image-Edit | gedit     | Semantic Consistency | style_change      |     5 |  6.2    | default |
+-----------------+-----------+----------------------+-------------------+-------+---------+---------+
| Qwen-Image-Edit | gedit     | Semantic Consistency | subject-add       |     5 |  8      | default |
+-----------------+-----------+----------------------+-------------------+-------+---------+---------+
| Qwen-Image-Edit | gedit     | Semantic Consistency | subject-remove    |     5 |  8.4    | default |
+-----------------+-----------+----------------------+-------------------+-------+---------+---------+
| Qwen-Image-Edit | gedit     | Semantic Consistency | subject-replace   |     5 |  8.4    | default |
+-----------------+-----------+----------------------+-------------------+-------+---------+---------+
| Qwen-Image-Edit | gedit     | Semantic Consistency | text_change       |     5 |  8.8    | default |
+-----------------+-----------+----------------------+-------------------+-------+---------+---------+
| Qwen-Image-Edit | gedit     | Semantic Consistency | tone_transfer     |     5 |  7.6    | default |
+-----------------+-----------+----------------------+-------------------+-------+---------+---------+
| Qwen-Image-Edit | gedit     | Semantic Consistency | OVERALL           |    55 |  7.4545 | -       |
+-----------------+-----------+----------------------+-------------------+-------+---------+---------+
| Qwen-Image-Edit | gedit     | Perceptual Quality   | background_change |     5 |  7.8    | default |
+-----------------+-----------+----------------------+-------------------+-------+---------+---------+
| Qwen-Image-Edit | gedit     | Perceptual Quality   | color_alter       |     5 |  6.8    | default |
+-----------------+-----------+----------------------+-------------------+-------+---------+---------+
| Qwen-Image-Edit | gedit     | Perceptual Quality   | material_alter    |     5 |  7.6    | default |
+-----------------+-----------+----------------------+-------------------+-------+---------+---------+
| Qwen-Image-Edit | gedit     | Perceptual Quality   | motion_change     |     5 |  7.8    | default |
+-----------------+-----------+----------------------+-------------------+-------+---------+---------+
| Qwen-Image-Edit | gedit     | Perceptual Quality   | ps_human          |     5 |  6.8    | default |
+-----------------+-----------+----------------------+-------------------+-------+---------+---------+
| Qwen-Image-Edit | gedit     | Perceptual Quality   | style_change      |     5 |  8      | default |
+-----------------+-----------+----------------------+-------------------+-------+---------+---------+
| Qwen-Image-Edit | gedit     | Perceptual Quality   | subject-add       |     5 |  7      | default |
+-----------------+-----------+----------------------+-------------------+-------+---------+---------+
| Qwen-Image-Edit | gedit     | Perceptual Quality   | subject-remove    |     5 |  7.8    | default |
+-----------------+-----------+----------------------+-------------------+-------+---------+---------+
| Qwen-Image-Edit | gedit     | Perceptual Quality   | subject-replace   |     5 |  7.8    | default |
+-----------------+-----------+----------------------+-------------------+-------+---------+---------+
| Qwen-Image-Edit | gedit     | Perceptual Quality   | text_change       |     5 |  7.2    | default |
+-----------------+-----------+----------------------+-------------------+-------+---------+---------+
| Qwen-Image-Edit | gedit     | Perceptual Quality   | tone_transfer     |     5 |  6.2    | default |
+-----------------+-----------+----------------------+-------------------+-------+---------+---------+
| Qwen-Image-Edit | gedit     | Perceptual Quality   | OVERALL           |    55 |  7.3455 | -       |
+-----------------+-----------+----------------------+-------------------+-------+---------+---------+
| Qwen-Image-Edit | gedit     | Overall              | background_change |     5 |  7.8967 | default |
+-----------------+-----------+----------------------+-------------------+-------+---------+---------+
| Qwen-Image-Edit | gedit     | Overall              | color_alter       |     5 |  7.317  | default |
+-----------------+-----------+----------------------+-------------------+-------+---------+---------+
| Qwen-Image-Edit | gedit     | Overall              | material_alter    |     5 |  6.7933 | default |
+-----------------+-----------+----------------------+-------------------+-------+---------+---------+
| Qwen-Image-Edit | gedit     | Overall              | motion_change     |     5 |  6.5765 | default |
+-----------------+-----------+----------------------+-------------------+-------+---------+---------+
| Qwen-Image-Edit | gedit     | Overall              | ps_human          |     5 |  5.6765 | default |
+-----------------+-----------+----------------------+-------------------+-------+---------+---------+
| Qwen-Image-Edit | gedit     | Overall              | style_change      |     5 |  6.2967 | default |
+-----------------+-----------+----------------------+-------------------+-------+---------+---------+
| Qwen-Image-Edit | gedit     | Overall              | subject-add       |     5 |  7.3798 | default |
+-----------------+-----------+----------------------+-------------------+-------+---------+---------+
| Qwen-Image-Edit | gedit     | Overall              | subject-remove    |     5 |  8.0908 | default |
+-----------------+-----------+----------------------+-------------------+-------+---------+---------+
| Qwen-Image-Edit | gedit     | Overall              | subject-replace   |     5 |  8.0827 | default |
+-----------------+-----------+----------------------+-------------------+-------+---------+---------+
| Qwen-Image-Edit | gedit     | Overall              | text_change       |     5 |  7.9547 | default |
+-----------------+-----------+----------------------+-------------------+-------+---------+---------+
| Qwen-Image-Edit | gedit     | Overall              | tone_transfer     |     5 |  6.7417 | default |
+-----------------+-----------+----------------------+-------------------+-------+---------+---------+
| Qwen-Image-Edit | gedit     | Overall              | OVERALL           |    55 |  7.1642 | -       |
+-----------------+-----------+----------------------+-------------------+-------+---------+---------+
```

## 自定义评测

当前AIGC模型的使用流程较为复杂，部分开发者需依赖ComfyUI等可视化工作流工具进行多模块组合生成，或通过API接口调用云端模型服务（如Stable Diffusion WebUI、商业API平台）。针对这类场景，EvalScope支持“无模型介入”的评测模式 ，仅需用户提供生成图片的存储路径以及对应Benchmark的`id`，即可直接启动评测流程，无需本地下载模型权重或执行推理计算。

### 数据格式

以`gedit`为例，需要提供下面的jsonl文件：

```json
{"id": "c18278ee2b0b3d8bd18c5279f4a8c636", "image_path": "Qwen-Image-Edit/images/c18278ee2b0b3d8bd18c5279f4a8c636_3.png"}
{"id": "15f01f7d55ad4f8695218594277e451f", "image_path": "Qwen-Image-Edit/images/15f01f7d55ad4f8695218594277e451f_0.png"}
{"id": "d83dad4db56f5c6c1270708a74311725", "image_path": "Qwen-Image-Edit/images/d83dad4db56f5c6c1270708a74311725_3.png"}
{"id": "caabd082c0ed1757df58db3eaea5ac73", "image_path": "Qwen-Image-Edit/images/caabd082c0ed1757df58db3eaea5ac73_1.png"}
{"id": "4a7d36259ad94d238a6e7e7e0bd6b643", "image_path": "Qwen-Image-Edit/images/4a7d36259ad94d238a6e7e7e0bd6b643_0.png"}
```
其中：
- `id` 为使用的评测基准中的样本唯一标识ID
- `image_path` 为生成图片的存储路径

### 配置任务

使用如下脚本可以加载本地数据进行评测：

```python
from dotenv import dotenv_values

env = dotenv_values('.env')

from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType, JudgeStrategy, ModelTask

task_config = TaskConfig(
    model_id='offline-model', # 无需配置model，只需提供自定义的ID
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
```

输出示例如下：
```text
+---------------+-----------+----------------------+----------------+-------+---------+---------+
| Model         | Dataset   | Metric               | Subset         |   Num |   Score | Cat.0   |
+===============+===========+======================+================+=======+=========+=========+
| offline-model | gedit     | Semantic Consistency | color_alter    |     1 |       8 | default |
+---------------+-----------+----------------------+----------------+-------+---------+---------+
| offline-model | gedit     | Semantic Consistency | material_alter |     1 |       2 | default |
+---------------+-----------+----------------------+----------------+-------+---------+---------+
| offline-model | gedit     | Semantic Consistency | OVERALL        |     2 |       5 | -       |
+---------------+-----------+----------------------+----------------+-------+---------+---------+
| offline-model | gedit     | Perceptual Quality   | color_alter    |     1 |       8 | default |
+---------------+-----------+----------------------+----------------+-------+---------+---------+
| offline-model | gedit     | Perceptual Quality   | material_alter |     1 |       8 | default |
+---------------+-----------+----------------------+----------------+-------+---------+---------+
| offline-model | gedit     | Perceptual Quality   | OVERALL        |     2 |       8 | -       |
+---------------+-----------+----------------------+----------------+-------+---------+---------+
| offline-model | gedit     | Overall              | color_alter    |     1 |       8 | default |
+---------------+-----------+----------------------+----------------+-------+---------+---------+
| offline-model | gedit     | Overall              | material_alter |     1 |       4 | default |
+---------------+-----------+----------------------+----------------+-------+---------+---------+
| offline-model | gedit     | Overall              | OVERALL        |     2 |       6 | -       |
+---------------+-----------+----------------------+----------------+-------+---------+---------+
```

## 可视化

EvalScope框架支持对评测结果进行可视化，用户可以通过以下命令生成可视化报告，方便对比模型的图像编辑效果：

```bash
evalscope app
```

使用文档请参考[可视化文档](../../get_started/visualization.md)。

示例如下：

![image](https://sail-moe.oss-cn-hangzhou.aliyuncs.com/yunlin/images/evalscope/image_edit/scores.png)
*打分分布*

![image](https://sail-moe.oss-cn-hangzhou.aliyuncs.com/yunlin/images/evalscope/image_edit/compare.png)
*效果对比*