# Evaluating the Inference Capability of R1 Models

With the widespread application of the DeepSeek-R1 model, an increasing number of developers are attempting to replicate similar models to enhance their inference capabilities. Many impressive results have emerged; however, do these new models actually demonstrate improved inference capabilities? The EvalScope framework, an open-source evaluation tool available in the Modao community, provides an assessment of the inference performance of R1 models.

In this best practice guide, we will demonstrate the evaluation process using 728 inference questions (consistent with the R1 technical report). The evaluation data includes:

- [MATH-500](https://www.modelscope.cn/datasets/HuggingFaceH4/aime_2024): A set of challenging high school mathematics competition problems across seven subjects (such as elementary algebra, algebra, number theory), comprising a total of 500 questions.
- [GPQA-Diamond](https://modelscope.cn/datasets/AI-ModelScope/gpqa_diamond/summary): This dataset contains master's level multiple-choice questions in the subfields of physics, chemistry, and biology, totaling 198 questions.
- [AIME-2024](https://modelscope.cn/datasets/AI-ModelScope/AIME_2024): A dataset from the American Invitational Mathematics Examination, containing 30 math problems.

The process outlined in this best practice includes installing the necessary dependencies, preparing the model, evaluating the model, and visualizing the evaluation results. Let’s get started.

## Installing Dependencies

First, install the [EvalScope](https://github.com/modelscope/evalscope) model evaluation framework:

```bash
pip install 'evalscope[app,perf]' -U
```

## Model Preparation

Next, we will introduce the evaluation process using the DeepSeek-R1-Distill-Qwen-1.5B model as an example. The model's capabilities will be accessed via an OpenAI API-compatible inference service for evaluation purposes. EvalScope also supports model evaluation via transformers inference; for details, please refer to the EvalScope documentation.

In addition to deploying the model on a cloud service that supports the OpenAI API, it can also be run locally using frameworks such as vLLM or ollama. Here, we will introduce the usage of the [vLLM](https://github.com/vllm-project/vllm) and [lmdeploy](https://github.com/InternLM/lmdeploy) inference frameworks, as these can effectively handle multiple concurrent requests to speed up the evaluation process. Since R1 models often produce lengthy reasoning chains with output token counts frequently exceeding 10,000, using efficient inference frameworks can enhance inference speed.

**Using vLLM**:

```bash
VLLM_USE_MODELSCOPE=True CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B  --served-model-name DeepSeek-R1-Distill-Qwen-1.5B --trust_remote_code --port 8801
```

or **Using lmdeploy**:

```bash
LMDEPLOY_USE_MODELSCOPE=True CUDA_VISIBLE_DEVICES=0 lmdeploy serve api_server deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --model-name DeepSeek-R1-Distill-Qwen-1.5B --server-port 8801
```

**(Optional) Test Inference Service Performance**

Before officially evaluating the model, you can test the performance of the inference service to select a better-performing inference engine using the `evalscope` `perf` subcommand:

```bash
evalscope perf \
 --parallel 10 \
 --url http://127.0.0.1:8801/v1/chat/completions \
 --model DeepSeek-R1-Distill-Qwen-1.5B \
 --log-every-n-query 5 \
 --connect-timeout 6000 \
 --read-timeout 6000 \
 --api openai \
 --prompt 'Write a science fiction novel, no less than 2000 words, please start your performance' \
 -n 100
```

For parameter explanations, please refer to the [Performance Evaluation Quick Start](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/quick_start.html).

<details><summary>Inference Service Performance Test Results</summary>

```text
Benchmarking summary:
+-----------------------------------+-------------------------------------------------------------------------+
| Key                               | Value                                                                   |
+===================================+=========================================================================+
| Time taken for tests (s)          | 92.66                                                                   |
+-----------------------------------+-------------------------------------------------------------------------+
| Number of concurrency             | 10                                                                      |
+-----------------------------------+-------------------------------------------------------------------------+
| Total requests                    | 100                                                                     |
+-----------------------------------+-------------------------------------------------------------------------+
| Succeed requests                  | 100                                                                     |
+-----------------------------------+-------------------------------------------------------------------------+
| Failed requests                   | 0                                                                       |
+-----------------------------------+-------------------------------------------------------------------------+
| Throughput(average tokens/s)      | 1727.453                                                                |
+-----------------------------------+-------------------------------------------------------------------------+
| Average QPS                       | 1.079                                                                   |
+-----------------------------------+-------------------------------------------------------------------------+
| Average latency (s)               | 8.636                                                                   |
+-----------------------------------+-------------------------------------------------------------------------+
| Average time to first token (s)   | 8.636                                                                   |
+-----------------------------------+-------------------------------------------------------------------------+
| Average time per output token (s) | 0.00058                                                                 |
+-----------------------------------+-------------------------------------------------------------------------+
| Average input tokens per request  | 20.0                                                                    |
+-----------------------------------+-------------------------------------------------------------------------+
| Average output tokens per request | 1600.66                                                                 |
+-----------------------------------+-------------------------------------------------------------------------+
| Average package latency (s)       | 8.636                                                                   |
+-----------------------------------+-------------------------------------------------------------------------+
| Average package per request       | 1.0                                                                     |
+-----------------------------------+-------------------------------------------------------------------------+
| Expected number of requests       | 100                                                                     |
+-----------------------------------+-------------------------------------------------------------------------+
| Result DB path                    | outputs/20250213_103632/DeepSeek-R1-Distill-Qwen-1.5B/benchmark_data.db |
+-----------------------------------+-------------------------------------------------------------------------+

Percentile results:
+------------+----------+----------+-------------+--------------+---------------+----------------------+
| Percentile | TTFT (s) | TPOT (s) | Latency (s) | Input tokens | Output tokens | Throughput(tokens/s) |
+------------+----------+----------+-------------+--------------+---------------+----------------------+
|    10%     |  5.4506  |   nan    |   5.4506    |      20      |     1011      |       183.7254       |
|    25%     |  6.1689  |   nan    |   6.1689    |      20      |     1145      |       184.9222       |
|    50%     |  9.385   |   nan    |    9.385    |      20      |     1741      |       185.5081       |
|    66%     | 11.0023  |   nan    |   11.0023   |      20      |     2048      |       185.8063       |
|    75%     | 11.0374  |   nan    |   11.0374   |      20      |     2048      |       186.1429       |
|    80%     |  11.047  |   nan    |   11.047    |      20      |     2048      |       186.3683       |
|    90%     |  11.075  |   nan    |   11.075    |      20      |     2048      |       186.5962       |
|    95%     |  11.147  |   nan    |   11.147    |      20      |     2048      |       186.7836       |
|    98%     | 11.1574  |   nan    |   11.1574   |      20      |     2048      |       187.4917       |
|    99%     | 11.1688  |   nan    |   11.1688   |      20      |     2048      |       197.4991       |
+------------+----------+----------+-------------+--------------+---------------+----------------------+
```

</details>

## Evaluating the Model

We will integrate the MATH-500, GPQA-Diamond, and AIME-2024 datasets into a single dataset, located in [evalscope/R1-Distill-Math-Test-v2](https://modelscope.cn/datasets/evalscope/R1-Distill-Math-Test-v2). Readers can directly use the dataset ID for evaluation operations.

If you want to learn about the dataset generation process or customize a dataset, please refer to the [Usage Tutorial](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/collection/index.html).

**Configuring the Evaluation Task**

You can evaluate the performance of the DeepSeek-R1-Distill-Qwen-1.5B model on the inference dataset using the following Python code:

```python
from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType

task_cfg = TaskConfig(
    model='DeepSeek-R1-Distill-Qwen-1.5B',   # Model name (must match the name used during deployment)
    api_url='http://127.0.0.1:8801/v1',  # Inference service address
    api_key='EMPTY',
    eval_type=EvalType.SERVICE,   # Evaluation type, SERVICE indicates evaluation of inference service
    datasets=[
        'data_collection',  # Dataset name (fixed as data_collection indicates mixed dataset usage)
    ],
    dataset_args={
        'data_collection': {
            'dataset_id': 'evalscope/R1-Distill-Math-Test-v2'  # Dataset ID or local path
        }
    },
    eval_batch_size=32,       # Number of concurrent requests
    generation_config={       # Model inference configuration
        'max_tokens': 20000,  # Maximum number of generated tokens, recommended to set a high value to avoid truncation
        'temperature': 0.6,   # Sampling temperature (recommended value from DeepSeek report)
        'top_p': 0.95,        # Top-p sampling (recommended value from DeepSeek report)
        'n': 5                # Number of replies generated per request (note that lmdeploy currently only supports n=1)
    },
    stream=True               # Whether to use streaming requests, recommended to set to True to prevent request timeouts
)

run_task(task_cfg=task_cfg)
```

**Output Results**:

**The computed metric here is `AveragePass@1`, and each sample was generated five times, with the final evaluation result being the average of these five attempts.** Due to the sampling during model generation, the output may exhibit some fluctuations.

```text
+-----------+--------------+---------------+-------+
| task_type | dataset_name | average_score | count |
+-----------+--------------+---------------+-------+
|   math    |   math_500   |    0.7832     |  500  |
|   math    |     gpqa     |    0.3434     |  198  |
|   math    |    aime24    |      0.2      |   30  |
+-----------+--------------+---------------+-------+
```

----

If you only wish to run specific datasets, you can modify the `datasets` and `dataset_args` parameters in the above configuration, for example:

```python
datasets=[
    # 'math_500',  # Dataset name
    'gpqa_diamond',
    'aime24'
],
dataset_args={ # Built-in support in EvalScope, no need to specify dataset ID
    'math_500': {'few_shot_num': 0 } ,
    'gpqa_diamond': {'few_shot_num': 0},
    'aime24': {'few_shot_num': 0}
},
```

Other available datasets include `gsm8k`, `aime25`, etc. For details, refer to the [Supported Datasets](https://evalscope.readthedocs.io/zh-cn/latest/get_started/supported_dataset/index.html).

## Visualizing Evaluation Results

EvalScope supports result visualization, allowing you to view the model's specific outputs.

Run the following command to launch the visualization interface:
```bash
evalscope app
```

The terminal will output the following link content:
```text
* Running on local URL:  http://0.0.0.0:7860
```

Click the link to see the visualization interface. You need to select the evaluation report and then click load:

<p align="center">
  <img src="https://sail-moe.oss-cn-hangzhou.aliyuncs.com/yunlin/images/distill/score.png" alt="alt text" width="100%">
</p>

Additionally, by selecting the corresponding sub-dataset, you can also view the model's output and check whether the output is correct (or if there are issues with answer matching):

<p align="center">
  <img src="https://sail-moe.oss-cn-hangzhou.aliyuncs.com/yunlin/images/distill/detail.png" alt="alt text" width="100%">
</p>

## Tips

:::{card}

💡 Here are some "pitfalls" that may be encountered during evaluation:

1. **Model Generation Configuration**:
    - Setting max_tokens: Ensure that max_tokens is set to a large value (usually above 8000). If set too low, the model may be truncated before outputting a complete answer.
    - Setting the number of replies n: In this evaluation, the number of replies generated per request n is set to 5, while in the R1 report, it was 64. Readers can adjust this parameter according to their needs to balance evaluation speed and result diversity.
    - Setting stream: The stream parameter should be set to True to prevent the model from timing out while generating long answers.
2. **Dataset Prompt Template Settings**:
    - The prompt template used in this article follows the recommended settings from the R1 report: "Please reason step by step, and put your final answer within \boxed{}."; simultaneously, no system prompt was set. Ensuring the correctness of the prompt template is crucial for generating expected results.
    - When evaluating reasoning models, it is advisable to set a 0-shot configuration; overly complex prompts or few-shot examples may degrade model performance.
3. **Parsing and Matching Generated Answers**:
    - We reused the parsing method from the Qwen-Math project, which is based on rules for answer parsing. However, this rule-based parsing might lead to matching errors, slightly affecting the reported metrics. It is recommended to utilize the evaluation result visualization feature to check for any discrepancies in parsing results.
:::

## Conclusion

Through this process, developers can effectively evaluate the performance of R1 models across multiple mathematical and scientific reasoning datasets, allowing for an objective assessment of specific model performances and collectively advancing the development and application of R1 models.
