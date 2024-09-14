# 自定义数据集评估

## LLM 数据集

本框架支持选择题和问答题，两种预定义的数据集格式，使用流程如下：

### 选择题格式（MCQ）
适合用户是选择题的场景，评测指标为准确率（accuracy）。

#### 1. 数据准备
准备选择题格式的csv文件，该目录包含了两个文件：
```text
custom/
├── example_dev.csv  # 名称需为*_dev.csv，用于fewshot评测，如果是0-shot评测，该csv可以为空
└── example_val.csv  # 名称需为*_val.csv，用于实际评测的数据
```

其中csv文件需要为下面的格式：

```text
id,question,A,B,C,D,answer,explanation
1,通常来说，组成动物蛋白质的氨基酸有____,4种,22种,20种,19种,C,1. 目前已知构成动物蛋白质的的氨基酸有20种。
2,血液内存在的下列物质中，不属于代谢终产物的是____。,尿素,尿酸,丙酮酸,二氧化碳,C,"代谢终产物是指在生物体内代谢过程中产生的无法再被利用的物质，需要通过排泄等方式从体内排出。丙酮酸是糖类代谢的产物，可以被进一步代谢为能量或者合成其他物质，并非代谢终产物。"
```
其中：
- `id`是评测序号
- `question`是问题
- `A` `B` `C` `D`是可选项（如果选项少于四个则对应留空）
- `answer`是正确选项
- `explanation`是解释（可选）

#### 2. 配置文件
```python
# 1. 配置自定义数据集文件
TaskConfig.registry(
    name='custom_dataset',      # 任务名称，可自定义
    data_pattern='ceval',       # 数据格式，选择题格式固定为 'ceval'
    dataset_dir='custom',       # 数据集路径
    subset_list=['example']     # 评测数据集名称，上述 *_dev.csv 中的 *
)

# 2. 配置任务，通过任务名称获取配置
task_cfg = registry_tasks['custom_dataset']

# 3. 配置模型和其他配置
task_cfg.update({
    'model_args': {'revision': None, 'precision': torch.float16, 'device_map': 'auto'},
    'eval_type': 'checkpoint',                 # 评测类型，需保留，固定为checkpoint
    'model': '../models/Qwen2-0.5B-Instruct',  # 模型路径
    'template_type': 'qwen',                   # 模型模板类型
    'outputs': 'outputs',
    'mem_cache': False,
    'limit': 10,
})
```

#### 3. 运行评测
```python
from evalscope.run import run_task
run_task(task_cfg=task_cfg)
```

运行结果：
```text
2024-08-27 11:33:58,917 - evalscope - INFO - ** Report table: 
 +---------+------------------+
| Model   | custom           |
+=========+==================+
|         | (custom/acc) 0.6 |
+---------+------------------+
```

### 问答题格式（QA）
适合用户是问答题的场景，评测指标是`ROUGE`和`BLEU`。

#### 1. 数据准备
准备一个问答题格式的jsonline文件，该目录包含了一个文件：

```text
custom_qa/
└── example.jsonl
```

该jsonline文件需要为下面的格式：

```json
{"query": "中国的首都是哪里？", "response": "中国的首都是北京"}
{"query": "世界上最高的山是哪座山？", "response": "是珠穆朗玛峰"}
{"query": "为什么北极见不到企鹅？", "response": "因为企鹅大多生活在南极"}
```

#### 2. 配置文件
```python
# 1. 配置自定义数据集文件
TaskConfig.registry(
    name='custom_dataset',      # 任务名称，可自定义
    data_pattern='general_qa',  # 数据格式，问答题格式固定为 'general_qa'
    dataset_dir='custom_qa',    # 数据集路径
    subset_list=['example']     # 评测数据集名称，上述 example.jsonl
)

# 2. 配置任务，通过任务名称获取配置
task_cfg = registry_tasks['custom_dataset']

# 3. 配置模型和其他配置
task_cfg.update({
    'model_args': {'revision': None, 'precision': torch.float16, 'device_map': 'auto'},
    'eval_type': 'checkpoint',                 # 评测类型，需保留，固定为checkpoint
    'model': '../models/Qwen2-0.5B-Instruct',  # 模型路径
    'template_type': 'qwen',                   # 模型模板类型
    'outputs': 'outputs',
    'mem_cache': False,
    'limit': 10,
})
```

#### 3. 运行评测
```python
from evalscope.run import run_task
run_task(task_cfg=task_cfg)
```

运行结果：
```text
2024-08-27 14:14:20,556 - evalscope - INFO - ** Report table: 
 +----------------------------------+-------------------------------------------+
| Model                            | custom_qa                                 |
+==================================+===========================================+
| 7aeeebf3d029ba4207e53c759be833e2 | (custom_qa/rouge-1-r) 0.8166666666666668  |
|                                  | (custom_qa/rouge-1-p) 0.22123015873015875 |
|                                  | (custom_qa/rouge-1-f) 0.31796037032374336 |
|                                  | (custom_qa/rouge-2-r) 0.54                |
|                                  | (custom_qa/rouge-2-p) 0.13554945054945053 |
|                                  | (custom_qa/rouge-2-f) 0.187063490583231   |
|                                  | (custom_qa/rouge-l-r) 0.8166666666666668  |
|                                  | (custom_qa/rouge-l-p) 0.21021876271876275 |
|                                  | (custom_qa/rouge-l-f) 0.30170995423739666 |
|                                  | (custom_qa/bleu-1) 0.21021876271876275    |
|                                  | (custom_qa/bleu-2) 0.1343230354551109     |
|                                  | (custom_qa/bleu-3) 0.075                  |
|                                  | (custom_qa/bleu-4) 0.06666666666666667    |
+----------------------------------+-------------------------------------------+ 
```

### (可选) 使用ms-swift框架自定义评估

```{seealso}
支持两种pattern的评测集：选择题格式的`CEval`和问答题格式的`General-QA`

参考：[ms-swift评估自定义评测集](../best_practice/swift_integration.md#自定义评测集)
```

--------------


## VLM 数据集

本框架支持选择题和问答题，两种预定义的数据集格式，使用流程如下：

````{note}
自定义数据集的评测需要使用`VLMEvalKit`，需要安装额外依赖：
```shell
pip install evalscope[vlmeval]
```
参考：[使用VLMEvalKit评测后端](../user_guides/vlmevalkit_backend.md)
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
  rerun: false
  work_dir: outputs
  nproc: 1
```
#### 3. 运行评测

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
        data_path = os.path.join("~/LMUData", f'{dataset}.tsv')
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
        
        # ========根据需要计算评估指标=========
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
  rerun: false
  work_dir: outputs
  nproc: 1
```

#### 4. 运行评测
````{important}
脚本中需导入自定义数据集，
```python
# 导入自定义数据集
from custom_dataset import CustomDataset
```
````
完整评测脚本如下：
```python
from custom_dataset import CustomDataset
from evalscope.run import run_task

run_task(task_cfg='config.yaml')
```

评测结果如下：
```text
{'qwen-vl-chat_custom_vqa_acc': {'Overall': '1.0'}}
```
