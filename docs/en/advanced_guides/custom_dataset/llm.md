# Large Language Model

This framework supports multiple-choice questions and question-answering questions, with two predefined dataset formats. The usage process is as follows:

## Multiple-Choice Question Format (MCQ)
Suitable for scenarios where users need multiple-choice questions. The evaluation metric is accuracy.

### 1. Data Preparation

Prepare files in multiple-choice question format, supporting both CSV and JSONL formats. The directory structure is as follows:

**CSV Format**

```text
mcq/
├── example_dev.csv   # (Optional) File name composed of `{subset_name}_dev.csv`, used for few-shot evaluation
└── example_val.csv   # File name composed of `{subset_name}_val.csv`, used for actual evaluation data
```

CSV files should be in the following format:

```text
id,question,A,B,C,D,answer
1,Generally speaking, the amino acids that make up animal proteins are ____,4 types,22 types,20 types,19 types,C
2,Among the substances present in the blood, which one is not a metabolic end product?____,Urea,Uric acid,Pyruvic acid,Carbon dioxide,C
```

**JSONL Format**

```text
mcq/
├── example_dev.jsonl # (Optional) File name composed of `{subset_name}_dev.jsonl`, used for few-shot evaluation
└── example_val.jsonl # File name composed of `{subset_name}_val.jsonl`, used for actual evaluation data
```

JSONL files should be in the following format:

```json
{"id": "1", "question": "Generally speaking, the amino acids that make up animal proteins are ____", "A": "4 types", "B": "22 types", "C": "20 types", "D": "19 types", "answer": "C"}
{"id": "2", "question": "Among the substances present in the blood, which one is not a metabolic end product?____", "A": "Urea", "B": "Uric acid", "C": "Pyruvic acid", "D": "Carbon dioxide", "answer": "C"}
```

Where:
- `id` is the serial number (optional field)
- `question` is the query
- `A`, `B`, `C`, `D`, etc., are the options, supporting up to 10 choices
- `answer` is the correct option

### 2. Configuration Task

Run the following code to start the evaluation:
```python
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='Qwen/Qwen2-0.5B-Instruct',
    datasets=['general_mcq'],  # Data format, fixed as 'general_mcq' for multiple-choice format
    dataset_args={
        'general_mcq': {
            "local_path": "custom_eval/text/mcq",  # Custom dataset path
            "subset_list": [
                "example"  # Evaluation dataset name, mentioned subset_name
            ]
        }
    },
)
run_task(task_cfg=task_cfg)
```

Results:
```text
+---------------------+-------------+-----------------+----------+-------+---------+---------+
| Model               | Dataset     | Metric          | Subset   |   Num |   Score | Cat.0   |
+=====================+=============+=================+==========+=======+=========+=========+
| Qwen2-0.5B-Instruct | general_mcq | AverageAccuracy | example  |    12 |  0.5833 | default |
+---------------------+-------------+-----------------+----------+-------+---------+---------+
```

## Question-Answering Format (QA)
Suitable for scenarios where users need question-answering questions. The evaluation metrics are `ROUGE` and `BLEU`.

### 1. Data Preparation
Prepare a JSON lines file for the question-answering format. The directory contains a single file:

```text
qa/
└── example.jsonl
```

The JSON lines file needs to be in the following format:

```json
{"system": "You are a geographer", "query": "What is the capital of China?", "response": "The capital of China is Beijing"}
{"query": "What is the highest mountain in the world?", "response": "It's Mount Everest"}
{"query": "Why can't penguins be seen in the Arctic?", "response": "Because penguins mostly live in Antarctica"}
```

Where:
- `system` is the system prompt (optional field)
- `query` is the question (required)
- `response` is the correct answer (required)

### 2. Configuration Task

Run the following code to start the evaluation:
```python
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='qwen/Qwen2-0.5B-Instruct',
    datasets=['general_qa'],  # Data format, fixed as 'general_qa' for question-answering format
    dataset_args={
        'general_qa': {
            "local_path": "custom_eval/text/qa",  # Custom dataset path
            "subset_list": [
                "example"       # Evaluation dataset name, corresponding to * in the above *.jsonl
            ]
        }
    },
)

run_task(task_cfg=task_cfg)
```

Results:
```text
+---------------------+-------------+-----------------+----------+-------+---------+---------+
| Model               | Dataset     | Metric          | Subset   |   Num |   Score | Cat.0   |
+=====================+=============+=================+==========+=======+=========+=========+
| Qwen2-0.5B-Instruct | general_qa  | bleu-1          | example  |    12 |  0.2324 | default |
+---------------------+-------------+-----------------+----------+-------+---------+---------+
| Qwen2-0.5B-Instruct | general_qa  | bleu-2          | example  |    12 |  0.1451 | default |
+---------------------+-------------+-----------------+----------+-------+---------+---------+
| Qwen2-0.5B-Instruct | general_qa  | bleu-3          | example  |    12 |  0.0625 | default |
+---------------------+-------------+-----------------+----------+-------+---------+---------+
| Qwen2-0.5B-Instruct | general_qa  | bleu-4          | example  |    12 |  0.0556 | default |
+---------------------+-------------+-----------------+----------+-------+---------+---------+
| Qwen2-0.5B-Instruct | general_qa  | rouge-1-f       | example  |    12 |  0.3441 | default |
+---------------------+-------------+-----------------+----------+-------+---------+---------+
| Qwen2-0.5B-Instruct | general_qa  | rouge-1-p       | example  |    12 |  0.2393 | default |
+---------------------+-------------+-----------------+----------+-------+---------+---------+
| Qwen2-0.5B-Instruct | general_qa  | rouge-1-r       | example  |    12 |  0.8889 | default |
+---------------------+-------------+-----------------+----------+-------+---------+---------+
| Qwen2-0.5B-Instruct | general_qa  | rouge-2-f       | example  |    12 |  0.2062 | default |
+---------------------+-------------+-----------------+----------+-------+---------+---------+
| Qwen2-0.5B-Instruct | general_qa  | rouge-2-p       | example  |    12 |  0.1453 | default |
+---------------------+-------------+-----------------+----------+-------+---------+---------+
| Qwen2-0.5B-Instruct | general_qa  | rouge-2-r       | example  |    12 |  0.6167 | default |
+---------------------+-------------+-----------------+----------+-------+---------+---------+
| Qwen2-0.5B-Instruct | general_qa  | rouge-l-f       | example  |    12 |  0.333  | default |
+---------------------+-------------+-----------------+----------+-------+---------+---------+
| Qwen2-0.5B-Instruct | general_qa  | rouge-l-p       | example  |    12 |  0.2324 | default |
+---------------------+-------------+-----------------+----------+-------+---------+---------+
| Qwen2-0.5B-Instruct | general_qa  | rouge-l-r       | example  |    12 |  0.8889 | default |
+---------------------+-------------+-----------------+----------+-------+---------+---------+ 
```
