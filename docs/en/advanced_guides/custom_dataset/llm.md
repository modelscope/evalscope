# Large Language Model

This framework supports multiple-choice questions, question-answering questions, and function calling, with three predefined dataset formats. The usage process is as follows:

## Multiple-Choice Question Format (MCQ)
Suitable for scenarios where users need multiple-choice questions. The evaluation metric is accuracy.

### Data Preparation

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

### Configuration Task

````{note}
The default `prompt_template` supports four options, as shown below, where `{query}` is the placeholder for the prompt. If you need fewer or more options, you can customize the `prompt_template`.
```text
Please answer the question and select the correct answer from the options. The last line of your answer should be in the following format: "答案是：LETTER" (without quotes), where LETTER is one of A, B, C, or D.
{query}
```
````

Run the following code to start the evaluation:

```python
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='Qwen/Qwen2-0.5B-Instruct',
    datasets=['general_mcq'],  # Data format: for multiple-choice questions, use 'general_mcq'
    dataset_args={
        'general_mcq': {
            # 'prompt_template': 'xxx',  # You can customize the prompt template if needed
            "local_path": "custom_eval/text/mcq",  # Path to your custom dataset
            "subset_list": [
                "example"  # Name of the evaluation dataset (the subset_name above)
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

This framework accommodates two formats for question-and-answer tasks: those with reference answers and those without.

1. **Reference Answer** Q&A: Suitable for questions with clear correct answers, with default evaluation metrics being `ROUGE` and `BLEU`. It can also be configured with an LLM judge for semantic correctness assessment.
2. **Reference-free Answer** Q&A: Suitable for questions without definitive correct answers, such as open-ended questions. By default, no evaluation metrics are provided, but an LLM judge can be configured to score the generated answers.

Here's how to use it:

### Data Preparation

Prepare a JSONL file in the Q&A format, for example, a directory containing a file:

```text
qa/
└── example.jsonl
```

This JSONL file contains one sample per line and supports the following three structures (please choose one and keep it consistent within the same file):

```json
{"system": "You are a geographer", "query": "What is the capital of China?", "response": "The capital of China is Beijing"}
{"query": "What is the highest mountain in the world?", "response": "It is Mount Everest"}
{"messages": [{"role": "system", "content": "You are a geographer"}, {"role": "user", "content": "What is the largest desert in the world?"}], "response": "It is the Sahara Desert"}
```

Field descriptions and required fields:
- Format 1 (system + query [+ response])
  - query: Required
  - response: Required when evaluating with reference answers; can be omitted or left empty when evaluating without reference answers
  - system: Optional
- Format 2 (query [+ response])
  - query: Required
  - response: Required when evaluating with reference answers; can be omitted or left empty when evaluating without reference answers
- Format 3 (messages [+ response])
  - messages: Required, array elements are {"role": "system"|"user"|"assistant", "content": "<text>"}; the last entry is recommended to be a user question
  - response: Required when evaluating with reference answers; can be omitted or left empty when evaluating without reference answers


### Reference Answer Q&A

Below is how to configure the evaluation of reference answer Q&A tasks using the `Qwen2.5` model on `example.jsonl`.

**Method 1: Evaluation based on `ROUGE` and `BLEU`**

Simply run the following code:
```python
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='Qwen/Qwen2.5-0.5B-Instruct',
    datasets=['general_qa'],  # Data format, fixed as 'general_qa' for Q&A tasks
    dataset_args={
        'general_qa': {
            "local_path": "custom_eval/text/qa",  # Custom dataset path
            "subset_list": [
                # Evaluation dataset name, the * in *.jsonl above, multiple subsets can be configured
                "example"       
            ]
        }
    },
)

run_task(task_cfg=task_cfg)
```

<details><summary>Click to view evaluation results</summary>

```text
+----------------+------------+-----------+----------+-------+---------+---------+
| Model                 | Dataset    | Metric    | Subset   |   Num |   Score | Cat.0   |
+================+============+===========+==========+=======+=========+=========+
| Qwen2.5-0.5B-Instruct | general_qa | Rouge-1-R | example  |   12 | 0.694 | default |
+----------------+------------+-----------+----------+-------+---------+---------+
| Qwen2.5-0.5B-Instruct | general_qa | Rouge-1-P | example  |   12 | 0.176  | default |
+----------------+------------+-----------+----------+-------+---------+---------+
| Qwen2.5-0.5B-Instruct | general_qa | Rouge-1-F | example  |   12 | 0.2276 | default |
+----------------+------------+-----------+----------+-------+---------+---------+
| Qwen2.5-0.5B-Instruct | general_qa | Rouge-2-R | example  |   12 | 0.4667 | default |
+----------------+------------+-----------+----------+-------+---------+---------+
| Qwen2.5-0.5B-Instruct | general_qa | Rouge-2-P | example  |   12 | 0.0939 | default |
+----------------+------------+-----------+----------+-------+---------+---------+
| Qwen2.5-0.5B-Instruct | general_qa | Rouge-2-F | example  |   12 | 0.1226 | default |
+----------------+------------+-----------+----------+-------+---------+---------+
| Qwen2.5-0.5B-Instruct | general_qa | Rouge-L-R | example  |   12 | 0.6528 | default |
+----------------+------------+-----------+----------+-------+---------+---------+
| Qwen2.5-0.5B-Instruct | general_qa | Rouge-L-P | example  |   12 | 0.1628 | default |
+----------------+------------+-----------+----------+-------+---------+---------+
| Qwen2.5-0.5B-Instruct | general_qa | Rouge-L-F | example  |   12 | 0.2063 | default |
+----------------+------------+-----------+----------+-------+---------+---------+
| Qwen2.5-0.5B-Instruct | general_qa | bleu-1    | example  |   12 | 0.164 | default |
+----------------+------------+-----------+----------+-------+---------+---------+
| Qwen2.5-0.5B-Instruct | general_qa | bleu-2    | example  |   12 | 0.0935 | default |
+----------------+------------+-----------+----------+-------+---------+---------+
| Qwen2.5-0.5B-Instruct | general_qa | bleu-3    | example  |   12 | 0.065 | default |
+----------------+------------+-----------+----------+-------+---------+---------+
| Qwen2.5-0.5B-Instruct | general_qa | bleu-4    | example  |   12 | 0.0556 | default |
+----------------+------------+-----------+----------+-------+---------+---------+ 
```
</details>

**Method 2: Evaluation based on LLM**

LLM-based evaluation can conveniently assess the correctness of model outputs (or other dimensions of metrics, requiring custom prompt settings). Below is an example configuring `judge_model_args` parameters, using the preset `pattern` mode to determine the correctness of model outputs.

For a complete explanation of judge parameters, please refer to [documentation](../../get_started/parameters.md#judge-parameters).

```python
import os
from evalscope import TaskConfig, run_task
from evalscope.constants import JudgeStrategy

task_cfg = TaskConfig(
    model='Qwen/Qwen2.5-0.5B-Instruct',
    datasets=[
        'general_qa',
    ],
    dataset_args={
        'general_qa': {
            'dataset_id': 'custom_eval/text/qa',
            'subset_list': [
                'example'
            ],
        }
    },
    # judge related parameters
    judge_model_args={
        'model_id': 'qwen2.5-72b-instruct',
        'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'api_key': os.getenv('DASHSCOPE_API_KEY'),
        'generation_config': {
            'temperature': 0.0,
            'max_tokens': 4096
        },
        # Determine if the model output is correct based on reference answers and model output
        'score_type': 'pattern',
    },
    # judge concurrency number
    judge_worker_num=5,
    # Use LLM for evaluation
    judge_strategy=JudgeStrategy.LLM,
)

run_task(task_cfg=task_cfg)
```

<details><summary>Click to view evaluation results</summary>

```text
+----------------+------------+----------------+----------+-------+---------+---------+
| Model                 | Dataset    | Metric          | Subset   |   Num |   Score | Cat.0   |
+================+============+================+==========+=======+=========+=========+
| Qwen2.5-0.5B-Instruct | general_qa | AverageAccuracy | example  |   12 | 0.583 | default |
+----------------+------------+----------------+----------+-------+---------+---------+ 
```
</details>

### Reference-free Answer Q&A

If the dataset lacks reference answers, an LLM judge can be used to evaluate the model's output answers. Without configuring an LLM, no scoring results will be available.

Below is an example configuring `judge_model_args` parameters, using the preset `numeric` mode to automatically assess model output scores from dimensions such as accuracy, relevance, and usefulness. Higher scores indicate better model output.

For a complete explanation of judge parameters, please refer to [documentation](../../get_started/parameters.md#judge-parameters).
```python
import os
from evalscope import TaskConfig, run_task
from evalscope.constants import JudgeStrategy

task_cfg = TaskConfig(
    model='Qwen/Qwen2.5-0.5B-Instruct',
    datasets=[
        'general_qa',
    ],
    dataset_args={
        'general_qa': {
            'dataset_id': 'custom_eval/text/qa',
            'subset_list': [
                'example'
            ],
        }
    },
    # judge related parameters
    judge_model_args={
        'model_id': 'qwen2.5-72b-instruct',
        'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'api_key': os.getenv('DASHSCOPE_API_KEY'),
        'generation_config': {
            'temperature': 0.0,
            'max_tokens': 4096
        },
        # Direct scoring
        'score_type': 'numeric',
    },
    # judge concurrency number
    judge_worker_num=5,
    # Use LLM for evaluation
    judge_strategy=JudgeStrategy.LLM,
)

run_task(task_cfg=task_cfg)
```

<details><summary>Click to view evaluation results</summary>

```text
+----------------+------------+----------------+----------+-------+---------+---------+
| Model                 | Dataset    | Metric          | Subset   |   Num |   Score | Cat.0   |
+================+============+================+==========+=======+=========+=========+
| Qwen2.5-0.5B-Instruct | general_qa | AverageAccuracy | example  |   12 | 0.6375 | default |
+----------------+------------+----------------+----------+-------+---------+---------+
```

</details>

## Function Calling Format (FC)

Applicable to evaluation scenarios that require function call decision-making and parameter structure validation. Supports:
- Preset dataset: evalscope/GeneralFunctionCall-Test
- Custom dataset: Pass in JSONL files via local path

The data is synthesized from official MoonshotAI [public samples](https://github.com/MoonshotAI/K2-Vendor-Verifier), including labels for "whether tool should be triggered" (K2-Thinking model as label source), facilitating experiment reproduction and automated regression testing.

### Data Format

Data is provided in JSONL format, with one sample per line, containing conversation context, tool definitions, and labels indicating whether tools should be triggered.

Example structure (one JSON object per line):
```json
{
    "messages": [
        {
            "role": "system",
            "content": "You are an assistant"
        },
        {
            "role": "user",
            "content": "Please add 2 and 3"
        }
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two numbers together",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {
                            "type": "number",
                            "description": "The first number"
                        },
                        "b": {
                            "type": "number",
                            "description": "The second number"
                        }
                    },
                    "required": [
                        "a",
                        "b"
                    ],
                    "additionalProperties": false
                }
            }
        }
    ],
    "should_call_tool": true
}
```

Field descriptions:
- messages: Evaluation input context (OpenAI-style role/content)
- tools: Available function list (OpenAI tool definition, `type=function, function={name, parameters(JSON Schema)}`)
- should_call_tool: Label indicating whether tool should be triggered (derived from official K2-thinking model results where `finish_reason == "tool_calls"`)

```{note}
- During evaluation, if the model returns `finish_reason == "tool_calls"`, it is considered as "model predicts tool call".
- Parameter validity is validated using JSON Schema (name must match, parameters must be parseable as JSON and satisfy the schema).
```

### Usage

```{note}
**Using model API services** is recommended for function calling evaluation, and ensure the model supports function calling. Local model inference does not currently support function calling.
```

Method 1: Using preset dataset ([evalscope/GeneralFunctionCall-Test](https://modelscope.cn/datasets/evalscope/GeneralFunctionCall-Test/dataPeview))

```python
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='qwen-plus',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=env.get('DASHSCOPE_API_KEY'),
    datasets=['general_fc'],  # Function calling format is fixed as 'general_fc'
    # To explicitly specify, add dataset_args={"general_fc": {"dataset_id": "evalscope/GeneralFunctionCall-Test"}}
)
run_task(task_cfg=task_cfg)
```

Method 2: Using custom local dataset

Example directory structure:
```text
fc/
└── example.jsonl  # One sample per line, structure as described above
```

Simple example (example.jsonl, 3 lines):
```json
{"messages":[{"role":"system","content":"You are an assistant"},{"role":"user","content":"Please add 2 and 3"}],"tools":[{"type":"function","function":{"name":"add","description":"Add two numbers together","parameters":{"type":"object","properties":{"a":{"type":"number","description":"The first number"},"b":{"type":"number","description":"The second number"}},"required":["a","b"],"additionalProperties":false}}}],"should_call_tool":true}
{"messages":[{"role":"system","content":"You are an assistant"},{"role":"user","content":"The weather is nice today, let's chat"}],"tools":[{"type":"function","function":{"name":"add","description":"Add two numbers together","parameters":{"type":"object","properties":{"a":{"type":"number","description":"The first number"},"b":{"type":"number","description":"The second number"}},"required":["a","b"],"additionalProperties":false}}}],"should_call_tool":false}
{"messages":[{"role":"system","content":"You are an assistant"},{"role":"user","content":"Convert 37 degrees Celsius to Fahrenheit"}],"tools":[{"type":"function","function":{"name":"convert_temperature","description":"Convert Celsius to Fahrenheit","parameters":{"type":"object","properties":{"celsius":{"type":"number","description":"Temperature value in Celsius"}},"required":["celsius"],"additionalProperties":false}}}],"should_call_tool":true}
```


Execution example:
```python
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='qwen-plus',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=env.get('DASHSCOPE_API_KEY'),
    datasets=['general_fc'],
    dataset_args={
        'general_fc': {
            "local_path": "custom_eval/text/fc",  # Custom dataset root directory
            "subset_list": ["example"]       # Corresponds to example.jsonl
        }
    },
)
run_task(task_cfg=task_cfg)
```

### Evaluation Metrics

general_fc outputs the following metrics:
- `count_finish_reason_tool_call`: Number of samples where model predicts tool call (`finish_reason == "tool_calls"`)
- `count_successful_tool_call`: Among samples attempting tool calls, number of samples with matching tool name and parameters passing JSON Schema validation
- `schema_accuracy`: Among samples attempting tool calls, proportion of parameters passing validation
- `tool_call_f1`: Binary classification F1 based on label `should_call_tool` and whether model calls tool

Example output:
```text
+-----------+------------+-------------------------------+----------+-------+---------+---------+
| Model     | Dataset    | Metric                        | Subset   |   Num |   Score | Cat.0   |
+===========+============+===============================+==========+=======+=========+=========+
| qwen-plus | general_fc | count_finish_reason_tool_call | default  |    10 |  3      | default |
+-----------+------------+-------------------------------+----------+-------+---------+---------+
| qwen-plus | general_fc | count_successful_tool_call    | default  |    10 |  2      | default |
+-----------+------------+-------------------------------+----------+-------+---------+---------+
| qwen-plus | general_fc | schema_accuracy               | default  |    10 |  0.6667 | default |
+-----------+------------+-------------------------------+----------+-------+---------+---------+
| qwen-plus | general_fc | tool_call_f1                  | default  |    10 |  0.5    | default |
+-----------+------------+-------------------------------+----------+-------+---------+---------+
```