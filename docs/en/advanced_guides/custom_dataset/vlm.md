# Multimodal Large Models

This framework supports two custom multimodal evaluation methods:

- **General-VQA Format**: Based on OpenAI message format, supports multi-image input, system prompts, and base64 images, suitable for Q&A-based multimodal evaluation tasks.
- **General-VMCQ Format**: Similar to MMMU format, question text can contain image placeholders `<image x>`, suitable for multiple-choice multimodal evaluation tasks.

## General-VQA Format

### 1. Data Preparation

Prepare data files conforming to OpenAI message format, supporting **JSONL** or **TSV** formats:

**JSONL Format Example** (`example_openai.jsonl`):
```json
{"messages": [{"role": "user", "content": [{"type": "text", "text": "What animal is this?"}, {"type": "image_url", "image_url": {"url": "custom_eval/multimodal/images/dog.jpg"}}]}], "answer": "Dog"}
{"messages": [{"role": "user", "content": [{"type": "text", "text": "What building is this?"}, {"type": "image_url", "image_url": {"url": "custom_eval/multimodal/images/AMNH.jpg"}}]}], "answer": "Museum"}
```

**TSV Format Example** (`example_openai.tsv`):
```text
messages	answer
[{"role": "user", "content": [{"type": "text", "text": "What animal is this?"}, {"type": "image_url", "image_url": {"url": "custom_eval/multimodal/images/dog.jpg"}}]}]	Dog
[{"role": "user", "content": [{"type": "text", "text": "What building is this?"}, {"type": "image_url", "image_url": {"url": "custom_eval/multimodal/images/AMNH.jpg"}}]}]	Museum
```

**Field Descriptions**:
- `messages`: OpenAI format message array, supporting:
  - Text content: `{"type": "text", "text": "question text"}`
  - Image URL: `{"type": "image_url", "image_url": {"url": "path or base64"}}`
  - System message: `{"role": "system", "content": "system prompt"}`
- `answer`: Reference answer (optional, used to calculate BLEU and Rouge scores)

**Supported Image Formats**:
- Local path: `"url": "custom_eval/multimodal/images/dog.jpg"`
- HTTP URL: `"url": "https://example.com/image.jpg"` (requires model service support)
- Base64 encoding: `"url": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."`

**Multi-image Input**

Supports using multiple images in one question:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Compare these two images:"},
        {"type": "image_url", "image_url": {"url": "image1.jpg"}},
        {"type": "text", "text": "and"},
        {"type": "image_url", "image_url": {"url": "image2.jpg"}},
        {"type": "text", "text": "What are the differences?"}
      ]
    }
  ],
  "answer": "The main differences are..."
}
```

**System Prompt**

You can add system messages to set the evaluation context:

```json
{
  "messages": [
    {"role": "system", "content": "You are a medical AI assistant."},
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Analyze this X-ray:"},
        {"type": "image_url", "image_url": {"url": "xray.jpg", "detail": "high"}}
      ]
    }
  ],
  "answer": "The X-ray shows..."
}
```

**Base64 Images**

Supports directly using base64 encoded images:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What's in this image?"},
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD..."
          }
        }
      ]
    }
  ],
  "answer": "A beautiful landscape"
}
```

### 2. Configure Evaluation Task

Evaluate using Python API or CLI:

**Python API**:
```python
from evalscope.run import run_task
from evalscope.config import TaskConfig
from os import environ as env

task_cfg = TaskConfig(
    model='qwen-vl-plus',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=env.get('DASHSCOPE_API_KEY'),
    eval_type='openai_api',
    datasets=['general_vqa'],
    dataset_args={
        'general_vqa': {
            'local_path': 'custom_eval/multimodal/vqa',  # Dataset directory
            'subset_list': ['example_openai'],  # Filename (without extension)
        }
    },
    limit=5,  # Optional: limit number of evaluation samples
)

result = run_task(task_cfg=task_cfg)
```

**CLI**:
```bash
evalscope eval \
    --model qwen-vl-plus \
    --api-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
    --api-key "$DASHSCOPE_API_KEY" \
    --eval-type openai_api \
    --datasets general_vqa \
    --dataset-args '{"general_vqa": {"local_path": "custom_eval/multimodal/vqa", "subset_list": ["example_openai"]}}' \
    --limit 5
```

Evaluation will output BLEU and Rouge metrics:
```text
+--------------+-------------+----------------+----------------+-------+---------+---------+
| Model        | Dataset     | Metric         | Subset         |   Num |   Score | Cat.0   |
+==============+=============+================+================+=======+=========+=========+
| qwen-vl-plus | general_vqa | mean_bleu-1    | example_openai |     5 |  0.0067 | default |
+--------------+-------------+----------------+----------------+-------+---------+---------+
| qwen-vl-plus | general_vqa | mean_bleu-2    | example_openai |     5 |  0      | default |
+--------------+-------------+----------------+----------------+-------+---------+---------+
| qwen-vl-plus | general_vqa | mean_bleu-3    | example_openai |     5 |  0      | default |
+--------------+-------------+----------------+----------------+-------+---------+---------+
| qwen-vl-plus | general_vqa | mean_bleu-4    | example_openai |     5 |  0      | default |
+--------------+-------------+----------------+----------------+-------+---------+---------+
| qwen-vl-plus | general_vqa | mean_Rouge-1-R | example_openai |     5 |  0.4    | default |
+--------------+-------------+----------------+----------------+-------+---------+---------+
| qwen-vl-plus | general_vqa | mean_Rouge-1-P | example_openai |     5 |  0.0062 | default |
+--------------+-------------+----------------+----------------+-------+---------+---------+
| qwen-vl-plus | general_vqa | mean_Rouge-1-F | example_openai |     5 |  0.0121 | default |
+--------------+-------------+----------------+----------------+-------+---------+---------+
| qwen-vl-plus | general_vqa | mean_Rouge-2-R | example_openai |     5 |  0      | default |
+--------------+-------------+----------------+----------------+-------+---------+---------+
| qwen-vl-plus | general_vqa | mean_Rouge-2-P | example_openai |     5 |  0      | default |
+--------------+-------------+----------------+----------------+-------+---------+---------+
| qwen-vl-plus | general_vqa | mean_Rouge-2-F | example_openai |     5 |  0      | default |
+--------------+-------------+----------------+----------------+-------+---------+---------+
| qwen-vl-plus | general_vqa | mean_Rouge-L-R | example_openai |     5 |  0.4    | default |
+--------------+-------------+----------------+----------------+-------+---------+---------+
| qwen-vl-plus | general_vqa | mean_Rouge-L-P | example_openai |     5 |  0.0047 | default |
+--------------+-------------+----------------+----------------+-------+---------+---------+
| qwen-vl-plus | general_vqa | mean_Rouge-L-F | example_openai |     5 |  0.0093 | default |
+--------------+-------------+----------------+----------------+-------+---------+---------+
```

### 3. Configure Judge Model

You can specify a judge model through the `judge_model` parameter to generate reference answers for evaluation, which will obtain accuracy metrics:

```python
from evalscope.run import run_task
from evalscope.constants import EvalType, JudgeStrategy
from os import environ as env

task_cfg = TaskConfig(
    model='qwen-vl-plus',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=env.get('DASHSCOPE_API_KEY'),
    eval_type='openai_api',
    datasets=['general_vqa'],
    dataset_args={
        'general_vqa': {
            'local_path': 'custom_eval/multimodal/vqa',
            'subset_list': ['example_openai'],
        }
    },
    limit=5,
    judge_model_args={
        'model_id': 'qwen-plus',  # Does not need to be a multimodal model
        'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'api_key': env.get('DASHSCOPE_API_KEY'),
        'generation_config': {
            'temperature': 0.0,
            'max_tokens': 4096
        },
    },
    judge_worker_num=5,
    judge_strategy=JudgeStrategy.LLM,
)
result = run_task(task_cfg=task_cfg)
```

**CLI** (equivalent):
```bash
evalscope eval \
  --model qwen-vl-plus \
  --api-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --api-key "$DASHSCOPE_API_KEY" \
  --eval-type openai_api \
  --datasets general_vqa \
  --dataset-args '{"general_vqa": {"local_path": "custom_eval/multimodal/vqa", "subset_list": ["example_openai"]}}' \
  --limit 5 \
  --judge-model-args '{"model_id": "qwen-plus", "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "api_key": "$DASHSCOPE_API_KEY", "generation_config": {"temperature": 0.0, "max_tokens": 4096}}' \
  --judge-worker-num 5 \
  --judge-strategy llm
```

Evaluation will output accuracy metrics:
```text
+--------------+-------------+----------+----------------+-------+---------+---------+
| Model        | Dataset     | Metric   | Subset         |   Num |   Score | Cat.0   |
+==============+=============+==========+================+=======+=========+=========+
| qwen-vl-plus | general_vqa | mean_acc | example_openai |     5 |       1 | default |
+--------------+-------------+----------+----------------+-------+---------+---------+ 
```

## General-VMCQ Format

### 1. Data Preparation

General-VMCQ adopts a structure similar to MMMU: question text can contain image placeholders `<image x>`; `options` is a Python list string, options can be text or image placeholders.

Images support two forms (both strings):
- Local or remote path/URL: `"custom_eval/multimodal/images/dog.jpg"` or `"https://.../dog.jpg"`
- Base64 Data URL: `"data:image/jpeg;base64,/9j/4AAQSk..."`

Supports up to 100 images (`image_1` to `image_100`). When text contains a non-existent image placeholder, parsing will stop directly (break).

**JSONL Example** (`example.jsonl`):
```json
{"question": "Which image shows a dog?", "options": ["<image 1>", "<image 2>", "<image 3>", "<image 4>"], "image_1": "custom_eval/multimodal/images/dog.jpg", "image_2": "custom_eval/multimodal/images/AMNH.jpg", "image_3": "custom_eval/multimodal/images/tesla.jpg", "image_4": "custom_eval/multimodal/images/tokyo.jpg", "answer": "A"}
{"question": "<image 1> What building is this?", "options": ["School", "Hospital", "Park", "Museum"], "image_1": "custom_eval/multimodal/images/AMNH.jpg", "answer": "D"}
```

**TSV Example** (`example.tsv`):
```text
question	options	answer	image_1	image_2	image_3	image_4
Which image shows a dog?	["<image 1>", "<image 2>", "<image 3>", "<image 4>"]	A	custom_eval/multimodal/images/dog.jpg	custom_eval/multimodal/images/AMNH.jpg	custom_eval/multimodal/images/tesla.jpg	custom_eval/multimodal/images/tokyo.jpg
<image 1> What building is this?	["School", "Hospital", "Park", "Museum"]	D	custom_eval/multimodal/images/AMNH.jpg			
```

**Field Descriptions**:
- `question`: Question text, can contain `<image x>` placeholders
- `options`: List (JSON array), elements can be text (e.g., `"School"`) or image placeholders (e.g., `"<image 1>"`), no need to add prefixes like `A.`, `B.`
- `answer`: Correct answer letter (e.g., `"A"`, `"B"`)
- `image_k`: Image string (local/remote path or base64 Data URL), k ∈ [1, 100]

### 2. Configure Evaluation Task

**Python API**:
```python
from evalscope.run import run_task
from evalscope.config import TaskConfig
from os import environ as env

task_cfg = TaskConfig(
    model='qwen-vl-plus',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=env.get('DASHSCOPE_API_KEY'),
    eval_type='openai_api',
    datasets=['general_vmcq'],
    dataset_args={
        'general_vmcq': {
            'local_path': 'custom_eval/multimodal/mcq',
            'subset_list': ['example'],
        }
    },
    limit=10,
)

result = run_task(task_cfg=task_cfg)
print(result)
```

**CLI**:
```bash
evalscope eval \
    --model qwen-vl-plus \
    --api-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
    --api-key "$DASHSCOPE_API_KEY" \
    --eval-type openai_api \
    --datasets general_vmcq \
    --dataset-args '{"general_vmcq": {"local_path": "custom_eval/multimodal/mcq", "subset_list": ["example"]}}' \
    --limit 10
```

### 3. Evaluation Results

Evaluation will output accuracy metrics:
```text
+--------------+--------------+----------+----------+-------+---------+---------+
| Model        | Dataset      | Metric   | Subset   |   Num |   Score | Cat.0   |
+==============+==============+==========+==========+=======+=========+=========+
| qwen-vl-plus | general_vmcq | mean_acc | example  |     3 |       1 | default |
+--------------+--------------+----------+----------+-------+---------+---------+ 
```

---

## Based on VLMEvalKit (Deprecated)

````{warning}
The following format is the Legacy version. It is recommended to use the **General Multimodal Format** described above.

Legacy format requires additional VLMEvalKit dependencies:
```bash
pip install evalscope[vlmeval]
```
Reference: [Evaluating with VLMEvalKit Backend](../../user_guides/backend/vlmevalkit_backend.md)
````


### Multiple Choice Format (MCQ)

#### 1. Data Preparation
The evaluation metric is accuracy, and you need to define a tsv file in the following format (separated by `\t`):
```text
index	category	answer	question	A	B	C	D	image_path
1	Animals	A	What animal is this?	Dog	Cat	Tiger	Elephant	/root/LMUData/images/custom_mcq/dog.jpg
2	Buildings	D	What building is this?	School	Hospital	Park	Museum	/root/LMUData/images/custom_mcq/AMNH.jpg
3	Cities	B	Which city's skyline is this?	New York	Tokyo	Shanghai	Paris	/root/LMUData/images/custom_mcq/tokyo.jpg
4	Vehicles	C	What is the brand of this car?	BMW	Audi	Tesla	Mercedes	/root/LMUData/images/custom_mcq/tesla.jpg
5	Activities	A	What is the person in the picture doing?	Running	Swimming	Reading	Singing	/root/LMUData/images/custom_mcq/running.jpg
```
Where:
- `index` is the question number
- `question` is the question
- `answer` is the answer
- `A`, `B`, `C`, `D` are options, must have at least two options
- `answer` is the answer option
- `image_path` is the image path (absolute path recommended); can also be replaced with `image` field, which should be base64 encoded image
- `category` is the category (optional field)

Place this file in the `~/LMUData` path, and you can use the filename for evaluation. For example, if the filename is `custom_mcq.tsv`, use `custom_mcq` for evaluation.

#### 2. Configuration File
The configuration file can be in `python dict`, `yaml`, or `json` format. For example, the following `config.yaml` file:
```yaml
eval_backend: VLMEvalKit
eval_config:
  model: 
    - type: qwen-vl-chat   # Deployed model name
      name: CustomAPIModel # Fixed value
      api_base: http://localhost:8000/v1/chat/completions
      key: EMPTY
      temperature: 0.0
      img_size: -1
  data:
    - custom_mcq # Custom dataset name, placed in `~/LMUData` path
  mode: all
  limit: 10
  reuse: false
  work_dir: outputs
  nproc: 1
```

#### 3. Run Evaluation

Run the following code to start evaluation:
```python
from evalscope.run import run_task

run_task(task_cfg='config.yaml')
```

Evaluation results:
```text
----------  ----
split       none
Overall     1.0
Activities  1.0
Animals     1.0
Buildings   1.0
Cities      1.0
Vehicles    1.0
----------  ----
```

### Custom Question-Answer Format (VQA)

#### 1. Data Preparation

Prepare a tsv file in question-answer format as follows:
```text
index	answer	question	image_path
1	Dog	What animal is this?	/root/LMUData/images/custom_mcq/dog.jpg
2	Museum	What building is this?	/root/LMUData/images/custom_mcq/AMNH.jpg
3	Tokyo	Which city's skyline is this?	/root/LMUData/images/custom_mcq/tokyo.jpg
4	Tesla	What is the brand of this car?	/root/LMUData/images/custom_mcq/tesla.jpg
5	Running	What is the person in the picture doing?	/root/LMUData/images/custom_mcq/running.jpg
```
This file is the same format as the multiple-choice format, where:
- `index` is the question number
- `question` is the question
- `answer` is the answer
- `image_path` is the image path (absolute path recommended); can also be replaced with `image` field, which should be base64 encoded image

Place this file in the `~/LMUData` path, and you can use the filename for evaluation. For example, if the filename is `custom_vqa.tsv`, use `custom_vqa` for evaluation.

#### 2. Custom Evaluation Script

The following is an example of a custom dataset. This example implements a custom evaluation script for question-answer format, which automatically loads the dataset, uses default prompts for Q&A, and finally calculates accuracy as the evaluation metric.


```python
import os
import numpy as np
from vlmeval.dataset.image_base import ImageBaseDataset
from vlmeval.dataset.image_vqa import CustomVQADataset
from vlmeval.smp import load, dump, d2df

class CustomDataset:
    def load_data(self, dataset):
        # Custom dataset loading
        data_path = os.path.join(os.path.expanduser("~/LMUData"), f'{dataset}.tsv')
        return load(data_path)
        
    def build_prompt(self, line):
        msgs = ImageBaseDataset.build_prompt(self, line)
        # Add prompts or custom instructions here
        msgs[-1]['value'] += '\nAnswer the question with one word or phrase.'
        return msgs
    
    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        assert 'answer' in data and 'prediction' in data
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x) for x in data['answer']]
        
        print(data)
        
        # ========Calculate evaluation metrics as needed=========
        # Exact match
        result = np.mean(data['answer'] == data['prediction'])
        ret = {'Overall': result}
        ret = d2df(ret).round(2)
        # Save results
        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(ret, result_file)
        return ret
        # ====================================
        
# Need to keep the following code to override default dataset class
CustomVQADataset.load_data = CustomDataset.load_data
CustomVQADataset.build_prompt = CustomDataset.build_prompt
CustomVQADataset.evaluate = CustomDataset.evaluate
```

#### 3. Configuration File
The configuration file can be in `python dict`, `yaml`, or `json` format. For example, the following `config.yaml` file:
```{code-block} yaml 
:caption: config.yaml

eval_backend: VLMEvalKit
eval_config:
  model: 
    - type: qwen-vl-chat   
      name: CustomAPIModel 
      api_base: http://localhost:8000/v1/chat/completions
      key: EMPTY
      temperature: 0.0
      img_size: -1
  data:
    - custom_vqa # Custom dataset name, placed in `~/LMUData` path
  mode: all
  limit: 10
  reuse: false
  work_dir: outputs
  nproc: 1
```

#### 4. Run Evaluation

The complete evaluation script is as follows:
```{code-block} python
:emphasize-lines: 1

from custom_dataset import CustomDataset  # Import custom dataset
from evalscope.run import run_task

run_task(task_cfg='config.yaml')
```

Evaluation results:
```text
{'qwen-vl-chat_custom_vqa_acc': {'Overall': '1.0'}}
```