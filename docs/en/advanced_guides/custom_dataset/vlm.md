# Multimodal Large Model

This framework supports multiple-choice questions and QA questions, two predefined dataset formats. The usage process is as follows:

````{note}
Custom dataset evaluation requires using `VLMEvalKit`, which requires additional dependencies:
```shell
pip install evalscope[vlmeval]
```
Reference: [Evaluation Backend with VLMEvalKit](../../user_guides/backend/vlmevalkit_backend.md)
````

## Multiple-Choice Question Format (MCQ)

### 1. Data Preparation
The evaluation metric is accuracy, and you need to define a tsv file in the following format (using `\t` as the separator):
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
- `A`, `B`, `C`, `D` are the options, with at least two options
- `answer` is the answer option
- `image_path` is the image path (absolute paths are recommended); this can also be replaced with the `image` field, which should be base64 encoded
- `category` is the category (optional field)

Place this file in the `~/LMUData` path, and you can use the filename for evaluation. For example, if the filename is `custom_mcq.tsv`, you can use `custom_mcq` for evaluation.

### 2. Configuration Task
The configuration file can be in `python dict`, `yaml`, or `json` format, for example, the following `config.yaml` file:
```yaml
eval_backend: VLMEvalKit
eval_config:
  model: 
    - type: qwen-vl-chat   # Name of the deployed model
      name: CustomAPIModel # Fixed value
      api_base: http://localhost:8000/v1/chat/completions
      key: EMPTY
      temperature: 0.0
      img_size: -1
  data:
    - custom_mcq # Name of the custom dataset, placed in `~/LMUData`
  mode: all
  limit: 10
  reuse: false
  work_dir: outputs
  nproc: 1
```
```{seealso}
VLMEvalKit [Parameter Description](../../user_guides/backend/vlmevalkit_backend.md#parameter-explanation)
```
### 3. Running Evaluation

Run the following code to start the evaluation:
```python
from evalscope.run import run_task

run_task(task_cfg='config.yaml')
```

The evaluation results are as follows:
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

## Custom QA Question Format (VQA)

### 1. Data Preparation

Prepare a QA formatted tsv file as follows:
```text
index	answer	question	image_path
1	Dog	What animal is this?	/root/LMUData/images/custom_mcq/dog.jpg
2	Museum	What building is this?	/root/LMUData/images/custom_mcq/AMNH.jpg
3	Tokyo	Which city's skyline is this?	/root/LMUData/images/custom_mcq/tokyo.jpg
4	Tesla	What is the brand of this car?	/root/LMUData/images/custom_mcq/tesla.jpg
5	Running	What is the person in the picture doing?	/root/LMUData/images/custom_mcq/running.jpg
```
This file is similar to the MCQ format, where:
- `index` is the question number
- `question` is the question
- `answer` is the answer
- `image_path` is the image path (absolute paths are recommended); this can also be replaced with the `image` field, which should be base64 encoded

Place this file in the `~/LMUData` path, and you can use the filename for evaluation. For example, if the filename is `custom_vqa.tsv`, you can use `custom_vqa` for evaluation.

### 2. Custom Evaluation Script

Below is an example of a custom dataset, implementing a custom QA format evaluation script. This script will automatically load the dataset, use default prompts for QA, and finally compute accuracy as the evaluation metric.

```python
import os
import numpy as np
from vlmeval.dataset.image_base import ImageBaseDataset
from vlmeval.dataset.image_vqa import CustomVQADataset
from vlmeval.smp import load, dump, d2df

class CustomDataset:
    def load_data(self, dataset):
        # Load custom dataset
        data_path = os.path.join(os.path.expanduser("~/LMUData"), f'{dataset}.tsv')
        return load(data_path)
        
    def build_prompt(self, line):
        msgs = ImageBaseDataset.build_prompt(self, line)
        # Add prompts or custom instructions here
        msgs[-1]['value'] += '\nAnswer the question in one word or phrase.'
        return msgs
    
    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        assert 'answer' in data and 'prediction' in data
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x) for x in data['answer']]
        
        print(data)
        
        # ========Compute the evaluation metric as needed=========
        # Exact match
        result = np.mean(data['answer'] == data['prediction'])
        ret = {'Overall': result}
        ret = d2df(ret).round(2)
        # Save the result
        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(ret, result_file)
        return ret
        # ========================================================
        
# Keep the following code and override the default dataset class
CustomVQADataset.load_data = CustomDataset.load_data
CustomVQADataset.build_prompt = CustomDataset.build_prompt
CustomVQADataset.evaluate = CustomDataset.evaluate
```

### 3. Configuration File
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
    - custom_vqa # Name of the custom dataset, placed in `~/LMUData`
  mode: all
  limit: 10
  reuse: false
  work_dir: outputs
  nproc: 1
```

### 4. Running Evaluation

The complete evaluation script is as follows:
```{code-block} python
:emphasize-lines: 1

from custom_dataset import CustomDataset  # Import the custom dataset
from evalscope.run import run_task

run_task(task_cfg='config.yaml')
```

The evaluation results are as follows:
```text
{'qwen-vl-chat_custom_vqa_acc': {'Overall': '1.0'}}
```