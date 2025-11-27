# 多模态大模型

本框架支持两种自定义多模态评测方式：

1. **通用格式（推荐）**：使用 OpenAI 兼容的消息格式，支持多模态内容（文本、图片等）
2. **Legacy 格式**：基于 VLMEvalKit 后端的传统格式

## 通用多模态格式（推荐）

### 简介

通用格式使用 OpenAI 兼容的消息结构，提供了更灵活和标准化的多模态评测方案。支持：

- **标准化格式**：遵循 OpenAI Chat Completion API 的消息格式
- **灵活的多模态输入**：支持文本、图片（本地路径或 base64 编码）
- **多种数据格式**：支持 TSV 和 JSONL 两种文件格式
- **无需额外依赖**：使用 EvalScope 原生评测引擎，无需安装 VLMEvalKit

### 通用问答题格式（General-VQA）

#### 1. 数据准备

准备符合 OpenAI 消息格式的数据文件，支持 **JSONL** 或 **TSV** 格式：

**JSONL 格式示例** (`example_openai.jsonl`):
```json
{"messages": [{"role": "user", "content": [{"type": "text", "text": "What animal is this?"}, {"type": "image_url", "image_url": {"url": "custom_eval/multimodal/images/dog.jpg"}}]}], "answer": "Dog"}
{"messages": [{"role": "user", "content": [{"type": "text", "text": "What building is this?"}, {"type": "image_url", "image_url": {"url": "custom_eval/multimodal/images/AMNH.jpg"}}]}], "answer": "Museum"}
```

**TSV 格式示例** (`example_openai.tsv`):
```tsv
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
- HTTP URL：`"url": "https://example.com/image.jpg"`
- Base64 编码：`"url": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."`

#### 2. 配置评测任务

使用 Python API 或 CLI 进行评测：

**Python API**:
```python
from evalscope.run import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='Qwen/Qwen2-VL-2B-Instruct',
    datasets=['general_vqa'],
    dataset_args={
        'general_vqa': {
            'local_path': 'custom_eval/multimodal/vqa',  # 数据集目录
            'subset_list': ['example_openai'],  # 文件名（不含扩展名）
        }
    },
    limit=10,  # 可选：限制评测样本数
)

result = run_task(task_cfg=task_cfg)
print(result)
```

**CLI**:
```bash
evalscope eval \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --datasets general_vqa \
    --dataset-args '{"general_vqa": {"local_path": "custom_eval/multimodal/vqa", "subset_list": ["example_openai"]}}' \
    --limit 10
```

#### 3. 评测结果

评测将输出 BLEU 和 Rouge 指标：
```text
----------  -------
Rouge-L-P   0.85
Rouge-L-R   0.82
Rouge-L-F   0.83
BLEU-1      0.75
BLEU-2      0.68
BLEU-3      0.62
BLEU-4      0.58
----------  -------
```

### 通用选择题格式（General-VMCQ）

#### 1. 数据准备

多选题数据格式类似，但需要在问题文本中包含选项：

**JSONL 格式示例** (`example_openai.jsonl`):
```json
{"messages": [{"role": "user", "content": [{"type": "text", "text": "What animal is this?\nA. Dog\nB. Cat\nC. Tiger\nD. Elephant"}, {"type": "image_url", "image_url": {"url": "custom_eval/multimodal/images/dog.jpg"}}]}], "answer": "A"}
{"messages": [{"role": "user", "content": [{"type": "text", "text": "What building is this?\nA. School\nB. Hospital\nC. Park\nD. Museum"}, {"type": "image_url", "image_url": {"url": "custom_eval/multimodal/images/AMNH.jpg"}}]}], "answer": "D"}
```

**TSV 格式示例** (`example_openai.tsv`):
```tsv
messages	answer
[{"role": "user", "content": [{"type": "text", "text": "What animal is this?\nA. Dog\nB. Cat\nC. Tiger\nD. Elephant"}, {"type": "image_url", "image_url": {"url": "custom_eval/multimodal/images/dog.jpg"}}]}]	A
[{"role": "user", "content": [{"type": "text", "text": "What building is this?\nA. School\nB. Hospital\nC. Park\nD. Museum"}, {"type": "image_url", "image_url": {"url": "custom_eval/multimodal/images/AMNH.jpg"}}]}]	D
```

**字段说明**：
- `messages`: 包含问题和选项的完整文本，以及图片
- `answer`: 正确答案的选项字母（如 "A", "B", "AB" 等）

#### 2. 配置评测任务

**Python API**:
```python
from evalscope.run import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='Qwen/Qwen2-VL-2B-Instruct',
    datasets=['general_vmcq'],
    dataset_args={
        'general_vmcq': {
            'local_path': 'custom_eval/multimodal/mcq',
            'subset_list': ['example_openai'],
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
    --model Qwen/Qwen2-VL-2B-Instruct \
    --datasets general_vmcq \
    --dataset-args '{"general_vmcq": {"local_path": "custom_eval/multimodal/mcq", "subset_list": ["example_openai"]}}' \
    --limit 10
```

#### 3. 评测结果

评测将输出准确率指标：
```text
------  ----
acc     0.80
------  ----
```

### 高级用法

#### 多图片输入

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

#### 系统提示

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

#### Base64 图片

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

---

## Legacy 格式（基于 VLMEvalKit）

````{warning}
以下格式为 Legacy 版本，推荐使用上述的**通用多模态格式**。

Legacy 格式需要额外依赖 VLMEvalKit：
```bash
pip install evalscope[vlmeval]
```
参考：[使用VLMEvalKit评测后端](../../user_guides/backend/vlmevalkit_backend.md)
````

---

## Legacy 格式（基于 VLMEvalKit）

````{warning}
以下格式为 Legacy 版本，推荐使用上述的**通用多模态格式**。

Legacy 格式需要额外依赖 VLMEvalKit：
```bash
pip install evalscope[vlmeval]
```
参考：[使用VLMEvalKit评测后端](../../user_guides/backend/vlmevalkit_backend.md)
````

## Legacy 选择题格式（MCQ）

### 1. 数据准备
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

### 2. 配置文件
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
```{seealso}
VLMEvalKit[参数说明](../../user_guides/backend/vlmevalkit_backend.md#参数说明)
```
### 3. 运行评测

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

## Legacy 自定义问答题格式（VQA）

### 1. 数据准备

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

### 2. 自定义评测脚本

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

### 3. 配置文件
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

### 4. 运行评测

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
