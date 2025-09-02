# Image Editing Evaluation

The image editing task evaluates the model's ability to comprehend and generate image content. By inputting an image and a directive, the model generates a corresponding image based on the instruction. The EvalScope framework supports the evaluation of various image editing models, including Qwen-Image-Edit and FLUX.1-Kontext. Users can assess these models through the EvalScope framework to obtain performance metrics.

## Supported Evaluation Datasets

Please refer to the [documentation](../../get_started/supported_dataset/aigc.md#aigc-benchmarks) and search using the `ImageEditing` tag.

## Installation of Dependencies

Users can install the necessary dependencies using the following command:

```bash
pip install evalscope[aigc] -U
```

## End-to-End Evaluation

Users can configure image editing models and complete model downloading, dataset downloading, model inference, and automatic evaluation with a single command. Taking the `Qwen-Image-Edit` model and `gedit` evaluation benchmark as an example:

```python
from dotenv import dotenv_values

env = dotenv_values('.env')

from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType, JudgeStrategy, ModelTask

task_config = TaskConfig(
    model='Qwen/Qwen-Image-Edit', # Model ID or local path
    model_args={
        'pipeline_cls': 'QwenImageEditPipeline',  # Pipeline class in diffusers
        'precision': 'bfloat16',  # Model precision
        'device_map': 'cuda:2'  # Device mapping, Qwen Image Edit requires approximately 60G of VRAM
    },
    model_task=ModelTask.IMAGE_GENERATION,  # Model task type
    eval_type=EvalType.IMAGE_EDITING,  # Evaluation task type
    generation_config={  # Inference parameters
        'true_cfg_scale': 4.0,
        'num_inference_steps': 50,
        'negative_prompt': ' ',
    },
    datasets=['gedit'],  # Benchmark used
    dataset_args={  # Specific parameters for the benchmark
        'gedit':{
            'extra_params':{
                'language': 'cn', # Use Chinese instructions
            }
        }
    },
    eval_batch_size=1,
    limit=5,
    judge_strategy=JudgeStrategy.AUTO,
    judge_worker_num=5,
    judge_model_args={  # Configure a VLM model for automatic scoring
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

Example output:
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

## Custom Evaluation

Given the complexity of current AIGC model usage, some developers rely on visual workflow tools like ComfyUI for multi-module generation or use API interfaces to call cloud model services (such as Stable Diffusion WebUI, commercial API platforms). For these scenarios, EvalScope supports a "model-free" evaluation mode, where users only need to provide the storage path of the generated images and the corresponding benchmark `id` to start the evaluation process without downloading model weights or performing inference locally.

### Data Format

For `gedit`, you need to provide the following jsonl file:

```json
{"id": "c18278ee2b0b3d8bd18c5279f4a8c636", "image_path": "Qwen-Image-Edit/images/c18278ee2b0b3d8bd18c5279f4a8c636_3.png"}
{"id": "15f01f7d55ad4f8695218594277e451f", "image_path": "Qwen-Image-Edit/images/15f01f7d55ad4f8695218594277e451f_0.png"}
{"id": "d83dad4db56f5c6c1270708a74311725", "image_path": "Qwen-Image-Edit/images/d83dad4db56f5c6c1270708a74311725_3.png"}
{"id": "caabd082c0ed1757df58db3eaea5ac73", "image_path": "Qwen-Image-Edit/images/caabd082c0ed1757df58db3eaea5ac73_1.png"}
{"id": "4a7d36259ad94d238a6e7e7e0bd6b643", "image_path": "Qwen-Image-Edit/images/4a7d36259ad94d238a6e7e7e0bd6b643_0.png"}
```
Where:
- `id` is the unique sample ID in the evaluation benchmark
- `image_path` is the storage path of the generated image

### Task Configuration

Use the following script to load local data for evaluation:

```python
from dotenv import dotenv_values

env = dotenv_values('.env')

from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType, JudgeStrategy, ModelTask

task_config = TaskConfig(
    model_id='offline-model', # No need to configure a model, just provide a custom ID
    model_task=ModelTask.IMAGE_GENERATION,  # Model task type
    eval_type=EvalType.IMAGE_EDITING,  # Evaluation task type
    datasets=['gedit'],  # Benchmark used
    dataset_args={  # Specific parameters for the benchmark
        'gedit':{
            'subset_list': ['color_alter', 'material_alter'], # Select evaluation subsets
            'extra_params':{
                'language': 'cn', # Use Chinese instructions
                'local_file': 'outputs/example_edit.jsonl' # Use locally generated images
            }
        }
    },
    eval_batch_size=1,
    limit=5,
    judge_strategy=JudgeStrategy.AUTO,
    judge_worker_num=5,
    judge_model_args={  # Configure a VLM model for automatic scoring
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

Example output:
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

## Visualization

The EvalScope framework supports visualization of evaluation results, allowing users to generate visual reports for easy comparison of the image editing effects of models:

```bash
evalscope app
```

For documentation, please refer to the [visualization documentation](../../get_started/visualization.md).

Examples:

![image](https://sail-moe.oss-cn-hangzhou.aliyuncs.com/yunlin/images/evalscope/image_edit/scores.png)
*Score Distribution*

![image](https://sail-moe.oss-cn-hangzhou.aliyuncs.com/yunlin/images/evalscope/image_edit/compare.png)
*Effect Comparison*