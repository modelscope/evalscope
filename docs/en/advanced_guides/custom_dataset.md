# Custom Dataset Evaluation

## LLM Dataset
This framework supports two types of pre-defined dataset formats for multiple-choice questions and open-ended questions. The usage process is as follows:
### Multiple-Choice Question Format (MCQ)
Suitable for scenarios where the user encounters multiple-choice questions. The evaluation metric is accuracy.

#### 1. Data Preparation
Prepare a CSV file in the multiple-choice question format. The directory should contain two files:
```text
custom/
├── example_dev.csv  # This file should be named *_dev.csv and is used for few-shot evaluation. If it is a 0-shot evaluation, this CSV can be empty.
└── example_val.csv  # This file should be named *_val.csv and is used for actual evaluation data.
```

The CSV file should be in the following format:
```text
id,question,A,B,C,D,answer,explanation
1,In general, amino acids that make up animal proteins are ____,4 kinds,22 types,20 types,19 types,C,1. It is currently known that there are 20 kinds of amino acids that make up animal proteins.
2,Of the following substances found in the blood, the one that is not a metabolic end product is ____. ,urea,uric acid,pyruvic acid,carbon dioxide,C,"Metabolic end products refer to substances produced during the metabolic process in an organism that cannot be further utilized and need to be excreted from the body through excretion, etc. Pyruvic acid is a product of carbohydrate metabolism, which can be further metabolized for energy or for the synthesis of other substances, and is not a metabolic end product."
```
Where:
- `id` is the evaluation number.
- `question` is the question.
- `A` `B` `C` `D` are the options (if there are fewer than four options, leave the corresponding ones blank).
- `answer` is the correct option.
- `explanation` is the explanation (optional).

#### 2. Configuration File
```python
# 1. Configure custom dataset file
TaskConfig.registry(
    name='custom_dataset',      # Task name, can be customized
    data_pattern='ceval',       # Data format, multiple-choice question format is fixed as 'ceval'
    dataset_dir='custom',       # Dataset path
    subset_list=['example']     # Evaluation dataset name, * in the above *_dev.csv
)

# 2. Configure the task, get configuration by task name
task_cfg = registry_tasks['custom_dataset']

# 3. Configure the model and other settings
task_cfg.update({
    'model_args': {'revision': None, 'precision': torch.float16, 'device_map': 'auto'},
    'eval_type': 'checkpoint',                 # Evaluation type, must be retained, fixed as checkpoint
    'model': '../models/Qwen2-0.5B-Instruct',  # Model path
    'template_type': 'qwen',                   # Model template type
    'outputs': 'outputs',
    'mem_cache': False,
    'limit': 10,
})
```
#### 3. Run Evaluation
```python
from evalscope.run import run_task
run_task(task_cfg=task_cfg)
```
Run result:
```text
2024-08-27 11:33:58,917 - evalscope - INFO - ** Report table: 
 +---------+------------------+
| Model   | custom           |
+=========+==================+
|         | (custom/acc) 0.6 |
+---------+------------------+
```

### Question Answering Format (QA)
Suitable for scenarios where the user encounters essay questions. The evaluation metrics are `ROUGE` and `BLEU`.
#### 1. Data Preparation
Prepare a JSON-line format file for essay questions, and the directory should contain one file:
```text
custom_qa/
└── example.jsonl
```
The JSON-line file should be in the following format:
```json
{"query": "What is the capital of China?", "response": "The capital of China is Beijing"}
{"query": "What is the highest mountain in the world?", "response": "It is Mount Everest"}
{"query": "Why can't penguins be found in the Arctic?", "response": "Because most penguins live in the Antarctic"}
```
#### 2. Configuration File
```python
# 1. Configure custom dataset file
TaskConfig.registry(
    name='custom_dataset',      # Task name, can be customized
    data_pattern='general_qa',  # Data format, essay question format is fixed as 'general_qa'
    dataset_dir='custom_qa',    # Dataset path
    subset_list=['example']     # Evaluation dataset name, example.jsonl as mentioned above
)

# 2. Configure the task, get configuration by task name
task_cfg = registry_tasks['custom_dataset']

# 3. Configure the model and other settings
task_cfg.update({
    'model_args': {'revision': None, 'precision': torch.float16, 'device_map': 'auto'},
    'eval_type': 'checkpoint',                 # Evaluation type, must be retained, fixed as checkpoint
    'model': '../models/Qwen2-0.5B-Instruct',  # Model path
    'template_type': 'qwen',                   # Model template type
    'outputs': 'outputs',
    'mem_cache': False,
    'limit': 10,
})
```
#### 3. Run Evaluation
```python
from evalscope.run import run_task
run_task(task_cfg=task_cfg)
```
Run result:
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

### (Optional) Custom Evaluation Using ms-swift Framework
```{seealso}
Supports two types of pattern evaluation sets: `CEval` for the multiple-choice question format and `General-QA` for the open-ended question format.

Reference: [Custom Evaluation Set with ms-swift](../best_practice/swift_integration.md#custom-evaluation-sets)
```
--------------

## VLM Dataset
This framework supports two pre-defined dataset formats for multiple-choice questions and open-ended questions. The usage process is as follows:
````{note}
Custom dataset evaluation requires the use of `VLMEvalKit` and requires the installation of additional dependencies:
```shell
pip install evalscope[vlmeval]
```
Reference: [Using VLMEvalKit for Evaluation Backend](../user_guides/vlmevalkit_backend.md)
````

### Multiple-Choice Question Format (MCQ)

#### 1. Data Preparation
The evaluation metric is accuracy, and it requires a `tsv` file with the following format (separated by `\t`):
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
- `answer` is the correct answer
- `A`, `B`, `C`, `D` are the options, with a minimum of two options
- `image_path` is the path to the image (it is recommended to use absolute path); it can also be replaced with the `image` field, which should be a base64-encoded image
- `category` is the category (optional field)

Place this file in the `~/LMUData` path and use the file name for evaluation. For example, if the file name is `custom_mcq.tsv`, then use `custom_mcq` for evaluation.

#### 2. Configuration File
The configuration file can be in Python dict, YAML, or JSON format. For example, the `config.yaml` file looks like this:
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
    - custom_mcq # Custom dataset name, placed in the `~/LMUData` path
  mode: all
  limit: 10
  rerun: false
  work_dir: outputs
  nproc: 1
```
#### 3. Run Evaluation
```python
from evalscope.run import run_task
run_task(task_cfg='config.yaml')
```
The evaluation result is as follows:
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

### Question Answering Format (VQA)
#### 1. Data Preparation
Prepare a `tsv` file of the question-answer format, with the following format:
```text
index	answer	question	image_path
1	Dog	What animal is this?	/root/LMUData/images/custom_mcq/dog.jpg
2	Museum	What building is this?	/root/LMUData/images/custom_mcq/AMNH.jpg
3	Tokyo	Which city's skyline is this?	/root/LMUData/images/custom_mcq/tokyo.jpg
4	Tesla	What is the brand of this car?	/root/LMUData/images/custom_mcq/tesla.jpg
5	Running	What is the person in the picture doing?	/root/LMUData/images/custom_mcq/running.jpg
```
The file follows the same format as multiple-choice questions, where:
- `index` is the question's number.
- `question` is the question.
- `answer` is the answer.
- `image_path` is the image path (recommended to use absolute path); can also be replaced with the `image` field, which needs to be a base64 encoded image.

Place this file in the `~/LMUData` path, and it can be used for evaluation with its file name. For example, if the file name is `custom_vqa.tsv`, it can be evaluated using `custom_vqa`.

#### 2. Custom Evaluation Script
Below is an example of a custom dataset with a custom question-answer format evaluation script. The script automatically loads the dataset, uses default prompts for question-answering, and finally calculates accuracy as the evaluation metric.

```python
import os
import numpy as np
from vlmeval.dataset.image_base import ImageBaseDataset
from vlmeval.dataset.image_vqa import CustomVQADataset
from vlmeval.smp import load, dump, d2df

class CustomDataset:
    def load_data(self, dataset):
        # Loading the custom dataset
        data_path = os.path.join("~/LMUData", f'{dataset}.tsv')
        return load(data_path)
        
    def build_prompt(self, line):
        msgs = ImageBaseDataset.build_prompt(self, line)
        # Add prompts or custom instructions here
        msgs[-1]['value'] += '\nAnswer the question using one word or phrase.'
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
        # =======================================================
        
# The following code must be kept to override the default dataset class
CustomVQADataset.load_data = CustomDataset.load_data
CustomVQADataset.build_prompt = CustomDataset.build_prompt
CustomVQADataset.evaluate = CustomDataset.evaluate
```
#### 3. Configuration File
The configuration file can be in `python dict`, `yaml`, or `json` format, for example, the `config.yaml` file below:
```yaml
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
    - custom_vqa # Name of the custom dataset, placed in the `~/LMUData` path
  mode: all
  limit: 10
  rerun: false
  work_dir: outputs
  nproc: 1
```
#### 4. Run Evaluation
````{note}
To import the custom dataset, the evaluation script is as follows:
```python
# Import the custom dataset
from custom_dataset import CustomDataset
from evalscope.run import run_task

run_task(task_cfg='config.yaml')
```
````

The evaluation result is as follows:
```text
{'qwen-vl-chat_custom_vqa_acc': {'Overall': '1.0'}}
```