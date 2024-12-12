# Large Language Model

This framework supports two predefined dataset formats: multiple-choice questions (MCQ) and question-answering (QA). The workflow is as follows:

## Multiple-Choice Question Format (MCQ)
This format is suitable for scenarios where users are dealing with multiple-choice questions, and the evaluation metric is accuracy.

### 1. Data Preparation
Prepare a CSV file in the MCQ format. This directory should contain two files:
```text
mcq/
├── example_dev.csv  # Name must be *_dev.csv for few-shot evaluation; this CSV can be empty for zero-shot evaluation.
└── example_val.csv  # Name must be *_val.csv for the actual evaluation data.
```

The CSV file must follow this format:

```text
id,question,A,B,C,D,answer,explanation
1,Generally, the amino acids that make up animal proteins are____,4,22,20,19,C,1. Currently, there are 20 known amino acids that constitute animal proteins.
2,Among the substances present in the blood, which of the following is not a metabolic end product____?,Urea,Uric acid,Pyruvate,Carbon dioxide,C,"Metabolic end products refer to substances produced during metabolism in the organism that cannot be reused and need to be excreted. Pyruvate is a product of carbohydrate metabolism that can be further metabolized for energy or to synthesize other substances, and is not a metabolic end product."
```
Where:
- `id` is the evaluation sequence number
- `question` is the question
- `A`, `B`, `C`, `D` are the options (leave empty if there are fewer than four options)
- `answer` is the correct option
- `explanation` is the explanation (optional)

### 2. Configuration File
```python
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='qwen/Qwen2-0.5B-Instruct',
    datasets=['ceval'],  # Dataset format, fixed as 'ceval' for multiple-choice questions
    dataset_args={
        'ceval': {
            "local_path": "custom_eval/text/mcq",  # Custom dataset path
            "subset_list": [
                "example"  # Evaluation dataset name, the * in the above *_dev.csv
            ]
        }
    },
)
run_task(task_cfg=task_cfg)
```

Run Result:
```text
+---------------------+---------------+
| Model               | mcq           |
+=====================+===============+
| Qwen2-0.5B-Instruct | (mcq/acc) 0.5 |
+---------------------+---------------+ 
```

## Question-Answering Format (QA)
This format is suitable for scenarios where users are dealing with question-answer pairs, and the evaluation metrics are `ROUGE` and `BLEU`.

### 1. Data Preparation
Prepare a JSONL file in the QA format. This directory should contain one file:

```text
qa/
└── example.jsonl
```

The JSONL file must follow this format:

```json
{"query": "What is the capital of China?", "response": "The capital of China is Beijing."}
{"query": "What is the tallest mountain in the world?", "response": "It is Mount Everest."}
{"query": "Why can't you find penguins in the Arctic?", "response": "Because most penguins live in the Antarctic."}
```

### 2. Configuration File
```python
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='qwen/Qwen2-0.5B-Instruct',
    datasets=['general_qa'],  # Dataset format, fixed as 'general_qa' for question-answering
    dataset_args={
        'general_qa': {
            "local_path": "custom_eval/text/qa",  # Custom dataset path
            "subset_list": [
                "example"       # Evaluation dataset name, the * in the above *.jsonl
            ]
        }
    },
)

run_task(task_cfg=task_cfg)
```

Run Result:
```text
+---------------------+------------------------------------+
| Model               | qa                                 |
+=====================+====================================+
| Qwen2-0.5B-Instruct | (qa/rouge-1-r) 0.888888888888889   |
|                     | (qa/rouge-1-p) 0.2386966558963222  |
|                     | (qa/rouge-1-f) 0.3434493794481033  |
|                     | (qa/rouge-2-r) 0.6166666666666667  |
|                     | (qa/rouge-2-p) 0.14595543345543344 |
|                     | (qa/rouge-2-f) 0.20751474380718113 |
|                     | (qa/rouge-l-r) 0.888888888888889   |
|                     | (qa/rouge-l-p) 0.23344334652802393 |
|                     | (qa/rouge-l-f) 0.33456027373987435 |
|                     | (qa/bleu-1) 0.23344334652802393    |
|                     | (qa/bleu-2) 0.14571148341640142    |
|                     | (qa/bleu-3) 0.0625                 |
|                     | (qa/bleu-4) 0.05555555555555555    |
+---------------------+------------------------------------+
```

## (Optional) Custom Evaluation Using the ms-swift Framework

```{seealso}
Supports two patterns of evaluation sets: Multiple-choice format `CEval` and Question-answering format `General-QA`

Reference: [ms-swift Custom Evaluation Sets](../../best_practice/swift_integration.md#custom-evaluation-sets)
```