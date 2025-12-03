# 多模态大模型

本框架支持两种自定义多模态评测方式：

- **通用问答题格式（General-VQA）**：基于 OpenAI 消息格式，支持多图片输入、系统提示和 base64 图片，适用于问答类多模态评测任务。
- **通用选择题格式（General-VMCQ）**：类似 MMMU 格式，问题文本中可包含图片占位符 `<image x>`，适用于选择题类多模态评测任务。

## 通用问答题格式（General-VQA）

### 1. 数据准备

准备符合 OpenAI 消息格式的数据文件，支持 **JSONL** 或 **TSV** 格式：

**JSONL 格式示例** (`example_openai.jsonl`):
```json
{"messages": [{"role": "user", "content": [{"type": "text", "text": "What animal is this?"}, {"type": "image_url", "image_url": {"url": "custom_eval/multimodal/images/dog.jpg"}}]}], "answer": "Dog"}
{"messages": [{"role": "user", "content": [{"type": "text", "text": "What building is this?"}, {"type": "image_url", "image_url": {"url": "custom_eval/multimodal/images/AMNH.jpg"}}]}], "answer": "Museum"}
```

**TSV 格式示例** (`example_openai.tsv`):
```text
messages	answer
[{"role": "user", "content": [{"type": "text", "text": "What animal is this?"}, {"type": "image_url", "image_url": {"url": "custom_eval/multimodal/images/dog.jpg"}}]}]	Dog
[{"role": "user", "content": [{"type": "text", "text": "What building is this?"}, {"type": "image_url", "image_url": {"url": "custom_eval/multimodal/images/AMNH.jpg"}}]}]	Museum
```

**字段说明**：
- `messages`: OpenAI 格式的消息数组，支持：
  - 文本内容：`{"type": "text", "text": "问题文本"}`
  - 图片 URL：`{"type": "image_url", "image_url": {"url": "路径或base64"}}`
  - 系统消息：`{"role": "system", "content": "系统提示"}`
- `answer`: 参考答案（可选，用于计算 BLEU 和 Rouge 分数）

**支持的图片格式**：
- 本地路径：`"url": "custom_eval/multimodal/images/dog.jpg"`
- HTTP URL：`"url": "https://example.com/image.jpg"`（需模型服务侧支持）
- Base64 编码：`"url": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."`

**多图片输入**

支持在一个问题中使用多张图片：

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

**系统提示**

可以添加系统消息来设置评测上下文：

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

**Base64 图片**

支持直接使用 base64 编码的图片：

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

### 2. 配置评测任务

使用 Python API 或 CLI 进行评测：

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
            'local_path': 'custom_eval/multimodal/vqa',  # 数据集目录
            'subset_list': ['example_openai'],  # 文件名（不含扩展名）
        }
    },
    limit=5,  # 可选：限制评测样本数
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

评测将输出 BLEU 和 Rouge 指标：
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

### 3. 配置裁判模型

可以通过 `judge_model` 参数指定裁判模型，用于生成参考答案进行评测，将获取准确率指标：

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
        'model_id': 'qwen-plus', # 无需是多模态模型
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

**CLI**（等效配置）:
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

评测将输出准确率指标（输出指标较ROUGE和BLEU更直观）：
```text
+--------------+-------------+----------+----------------+-------+---------+---------+
| Model        | Dataset     | Metric   | Subset         |   Num |   Score | Cat.0   |
+==============+=============+==========+================+=======+=========+=========+
| qwen-vl-plus | general_vqa | mean_acc | example_openai |     5 |       1 | default |
+--------------+-------------+----------+----------------+-------+---------+---------+ 
```

## 通用选择题格式（General-VMCQ）

### 1. 数据准备

General-VMCQ 采用与 MMMU 相似的结构：问题文本中可包含图片占位符 `<image x>`；`options` 为 Python 列表字符串，选项可为文本或图片占位符。

支持图片两种形式（均为字符串）：
- 本地或远程路径/URL：`"custom_eval/multimodal/images/dog.jpg"` 或 `"https://.../dog.jpg"`
- Base64 Data URL：`"data:image/jpeg;base64,/9j/4AAQSk..."`

支持最多 100 张图片（`image_1` 到 `image_100`）。当文本中出现不存在的图片占位符时，会直接停止解析后续内容（break）。

**JSONL 示例**（`example.jsonl`）：
```json
{"question": "Which image shows a dog?", "options": ["<image 1>", "<image 2>", "<image 3>", "<image 4>"], "image_1": "custom_eval/multimodal/images/dog.jpg", "image_2": "custom_eval/multimodal/images/AMNH.jpg", "image_3": "custom_eval/multimodal/images/tesla.jpg", "image_4": "custom_eval/multimodal/images/tokyo.jpg", "answer": "A"}
{"question": "<image 1> What building is this?", "options": ["School", "Hospital", "Park", "Museum"], "image_1": "custom_eval/multimodal/images/AMNH.jpg", "answer": "D"}
```

**TSV 示例**（`example.tsv`）：
```text
question	options	answer	image_1	image_2	image_3	image_4
Which image shows a dog?	["<image 1>", "<image 2>", "<image 3>", "<image 4>"]	A	custom_eval/multimodal/images/dog.jpg	custom_eval/multimodal/images/AMNH.jpg	custom_eval/multimodal/images/tesla.jpg	custom_eval/multimodal/images/tokyo.jpg
<image 1> What building is this?	["School", "Hospital", "Park", "Museum"]	D	custom_eval/multimodal/images/AMNH.jpg			
```

**字段说明**：
- `question`: 问题文本，可包含 `<image x>` 占位符
- `options`: 列表（JSON 数组），元素可以是文本（如 `"School"`）或图片占位符（如 `"<image 1>"`），不需要添加 `A.`、`B.` 等前缀
- `answer`: 正确答案字母（如 `"A"`、`"B"`）
- `image_k`: 图片字符串（本地/远程路径或 base64 Data URL），k ∈ [1, 100]

### 2. 配置评测任务

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

### 3. 评测结果

评测将输出准确率指标：
```text
+--------------+--------------+----------+----------+-------+---------+---------+
| Model        | Dataset      | Metric   | Subset   |   Num |   Score | Cat.0   |
+==============+==============+==========+==========+=======+=========+=========+
| qwen-vl-plus | general_vmcq | mean_acc | example  |     3 |       1 | default |
+--------------+--------------+----------+----------+-------+---------+---------+ 
```

---

## 基于 VLMEvalKit (已废弃)

````{warning}
以下格式为 Legacy 版本，推荐使用上述的**通用多模态格式**。

Legacy 格式需要额外依赖 VLMEvalKit：
```bash
pip install evalscope[vlmeval]
```
参考：[使用VLMEvalKit评测后端](../../user_guides/backend/vlmevalkit_backend.md)
````


### 选择题格式（MCQ）

#### 1. 数据准备
评测指标为准确率（accuracy），需要定义如下格式的tsv文件（使用`\t`分割）：
```text
index	category	answer	question	A	B	C	D	image_path
1	Animals	A	What animal is this?	Dog	Cat	Tiger	Elephant	/root/LMUData/images/custom_mcq/dog.jpg
2	Buildings	D	What building is this?	School	Hospital	Park	Museum	/root/LMUData/images/custom_mcq/AMNH.jpg
3	Cities	B	Which city's skyline is this?	New York	Tokyo	Shanghai	Paris	/root/LMUData/images/custom_mcq/tokyo.jpg
4	Vehicles	C	What is the brand of this car?	BMW	Audi	Tesla	Mercedes	/root/LMUData/images/custom_mcq/tesla.jpg
5	Activities	A	What is the person in the picture doing?	Running	Swimming	Reading	Singing	/root/LMUData/images/custom_mcq/running.jpg
```
其中：
- `index`为问题序号
- `question`为问题
- `answer`为答案
- `A`、`B`、`C`、`D`为选项，不得少于两个选项
- `answer`为答案选项
- `image_path`为图片路径（建议使用绝对路径）；也可替换为`image`字段，需为base64编码的图片
- `category`为类别（可选字段）

将该文件放在`~/LMUData`路径中，即可使用文件名来进行评测。例如该文件名为`custom_mcq.tsv`，则使用`custom_mcq`即可评测。

#### 2. 配置文件
配置文件，可以为`python dict`、`yaml`或`json`格式，例如如下`config.yaml`文件：
```yaml
eval_backend: VLMEvalKit
eval_config:
  model: 
    - type: qwen-vl-chat   # 部署的模型名称
      name: CustomAPIModel # 固定值
      api_base: http://localhost:8000/v1/chat/completions
      key: EMPTY
      temperature: 0.0
      img_size: -1
  data:
    - custom_mcq # 自定义数据集名称，放在`~/LMUData`路径中
  mode: all
  limit: 10
  reuse: false
  work_dir: outputs
  nproc: 1
```

#### 3. 运行评测

运行下面的代码，即可开始评测：
```python
from evalscope.run import run_task

run_task(task_cfg='config.yaml')
```

评测结果如下：
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

### 自定义问答题格式（VQA）

#### 1. 数据准备

准备一个问答题格式的tsv文件，格式如下：
```text
index	answer	question	image_path
1	Dog	What animal is this?	/root/LMUData/images/custom_mcq/dog.jpg
2	Museum	What building is this?	/root/LMUData/images/custom_mcq/AMNH.jpg
3	Tokyo	Which city's skyline is this?	/root/LMUData/images/custom_mcq/tokyo.jpg
4	Tesla	What is the brand of this car?	/root/LMUData/images/custom_mcq/tesla.jpg
5	Running	What is the person in the picture doing?	/root/LMUData/images/custom_mcq/running.jpg
```
该文件与选择题格式相同，其中：
- `index`为问题序号
- `question`为问题
- `answer`为答案
- `image_path`为图片路径（建议使用绝对路径）；也可替换为`image`字段，需为base64编码的图片

将该文件放在`~/LMUData`路径中，即可使用文件名来进行评测。例如该文件名为`custom_vqa.tsv`，则使用`custom_vqa`即可评测。

#### 2. 自定义评测脚本

以下是一个自定义数据集的示例，该示例实现了一个自定义的问答题格式的评测脚本，该脚本会自动加载数据集，并使用默认的提示进行问答，最后计算准确率作为评测指标。


```python
import os
import numpy as np
from vlmeval.dataset.image_base import ImageBaseDataset
from vlmeval.dataset.image_vqa import CustomVQADataset
from vlmeval.smp import load, dump, d2df

class CustomDataset:
    def load_data(self, dataset):
        # 自定义数据集的加载
        data_path = os.path.join(os.path.expanduser("~/LMUData"), f'{dataset}.tsv')
        return load(data_path)
        
    def build_prompt(self, line):
        msgs = ImageBaseDataset.build_prompt(self, line)
        # 这里添加提示或自定义指令
        msgs[-1]['value'] += '\n用一个单词或短语回答问题。'
        return msgs
    
    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        assert 'answer' in data and 'prediction' in data
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x) for x in data['answer']]
        
        print(data)
        
        # ========根据需要计算评测指标=========
        # 精确匹配
        result = np.mean(data['answer'] == data['prediction'])
        ret = {'Overall': result}
        ret = d2df(ret).round(2)
        # 保存结果
        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(ret, result_file)
        return ret
        # ====================================
        
# 需保留以下代码，重写默认的数据集类
CustomVQADataset.load_data = CustomDataset.load_data
CustomVQADataset.build_prompt = CustomDataset.build_prompt
CustomVQADataset.evaluate = CustomDataset.evaluate
```

#### 3. 配置文件
配置文件，可以为`python dict`、`yaml`或`json`格式，例如如下`config.yaml`文件：
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
    - custom_vqa # 自定义数据集名称，放在`~/LMUData`路径中
  mode: all
  limit: 10
  reuse: false
  work_dir: outputs
  nproc: 1
```

#### 4. 运行评测

完整评测脚本如下：
```{code-block} python
:emphasize-lines: 1

from custom_dataset import CustomDataset  # 导入自定义数据集
from evalscope.run import run_task

run_task(task_cfg='config.yaml')
```

评测结果如下：
```text
{'qwen-vl-chat_custom_vqa_acc': {'Overall': '1.0'}}
```
