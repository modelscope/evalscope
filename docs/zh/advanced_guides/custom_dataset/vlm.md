# 多模态大模型

本框架支持多模态选择题和问答题，两种预定义的数据集格式，使用流程如下：

````{note}
自定义数据集的评测需要使用`VLMEvalKit`，需要安装额外依赖：
```shell
pip install evalscope[vlmeval]
```
参考：[使用VLMEvalKit评测后端](../../user_guides/backend/vlmevalkit_backend.md)
````

## 选择题格式（MCQ）

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

## 自定义问答题格式（VQA）

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