(ragas)=
# RAGAS

This framework supports [RAGAS](https://github.com/explodinggradients/ragas) (Retrieval Augmented Generation Assessment), a dedicated framework for evaluating the performance of Retrieval Augmented Generation (RAG) processes. Some of its core evaluation metrics include:

- Faithfulness: Measures the factual accuracy of generated answers, ensuring the truthfulness and reliability of the generated content.
- Answer Relevance: Assesses the relevance of answers to the given questions, verifying whether the response directly addresses the user's query.
- Context Precision: Evaluates the precision of the context used to generate answers, ensuring that relevant information is selected from the context.
- Context Relevancy: Measures the relevance of the chosen context to the question, helping to improve context selection to enhance answer accuracy.
- Context Recall: Evaluates the completeness of retrieved relevant contextual information, ensuring that no key context is omitted.

In addition, RAGAS also provides tools for automatic test data generation for user convenience.

## Environment Setup
Install dependencies

```shell
pip install ragas
```

## Dataset Preparation

Example of evaluation dataset is as follows:
```json
{
    "question": [
        "When was the first super bowl?",
        "Who won the most super bowls?"
    ],
    "answer": [
        "The first superbowl was held on Jan 15, 1967",
        "The most super bowls have been won by The New England Patriots"
    ],
    "contexts": [
        [
            "The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles,"
        ],
        [
            "The Green Bay Packers...Green Bay, Wisconsin.",
            "The Packers compete...Football Conference"
        ]
    ],
    "ground_truth": [
        "The first superbowl was held on January 15, 1967",
        "The New England Patriots have won the Super Bowl a record six times"
    ]
}
```
Required fields include:
- question: Question
- answer: Answer
- contexts: List of contexts
- ground_truth: Standard answer
### Automatic Data Generation

RAGAS offers the capability to automatically generate test data, allowing users to specify parameters such as test set size, data distribution, generator, and LLM, to automatically produce test datasets. The specific steps are as follows:

```{figure} images/generation_process.png

Ragas adopts a novel approach to generating evaluation data. An ideal evaluation dataset should encompass various types of questions encountered in real-world applications, including questions of different difficulty levels. By default, large language models (LLMs) struggle to create diverse samples since they often follow common paths. Inspired by works like [Evol-Instruct](https://arxiv.org/abs/2304.12244), Ragas achieves this goal by adopting an evolutionary generation paradigm, where questions with different characteristics (such as reasoning, conditions, multiple contexts, etc.) are systematically constructed based on the provided document set. This method ensures comprehensive coverage of the various components' performance in your pipeline, making the evaluation process more robust.
```

#### Configuration Task
```python
generate_testset_task_cfg = {
    "eval_backend": "RAGEval",
    "eval_config": {
        "tool": "RAGAS",
        "testset_generation": {
            "docs": ["README.md"],
            "test_size": 5,
            "output_file": "outputs/testset.json",
            "distribution": {"simple": 0.5, "multi_context": 0.4, "reasoning": 0.1},
            "generator_llm": {
                "model_name_or_path": "qwen/Qwen2-7B-Instruct",
                "template_type": "qwen",
            },
            "critic_llm": {
                "model_name_or_path": "QwenCollection/Ragas-critic-llm-Qwen1.5-GPTQ",
                "template_type": "qwen",
            },
            "embeddings": {
                "model_name_or_path": "AI-ModelScope/m3e-base",
            },
        }
    },
}
```
Configuration file description:
- `eval_backend`: string, description: The name of the evaluation backend, e.g., "RAGEval".
- `eval_config`: dictionary, description: Contains detailed information about the evaluation configuration.
  - `tool`: string, description: The name of the evaluation tool, e.g., "RAGAS".
  - `testset_generation`: dictionary, description: Configuration for test set generation:
    - `docs`: list, description: A list of documents needed for test set generation, e.g., ["README.md"].
    - `test_size`: integer, description: The size of the generated test set, e.g., 5.
    - `output_file`: string, description: The path for the generated output file, e.g., "outputs/testset.json".
    - `distribution`: dictionary, description: Configuration for the distribution of test set content.
      - `simple`: float, description: The distribution proportion of simple content, e.g., 0.5.
      - `multi_context`: float, description: The distribution proportion of multi-context content, e.g., 0.4.
      - `reasoning`: float, description: The distribution proportion of reasoning content, e.g., 0.1.

    - `generator_llm`: dictionary, description: Configuration for the generator LLM (Large Language Model):
        - If using a local model, the following parameters are supported:
            - `model_name_or_path`: string, description: The name or path of the generator model. Enter a name, e.g., "qwen/Qwen2-7B-Instruct" to automatically download the model from ModelScope; enter a path to load the model locally.
            - `template_type`: string, description: Template type, e.g., "qwen".
            - `generation_config`: dictionary, description: Generation configuration, e.g., `{"temperature": 0.7}`.
        - If using an API model, the following parameters are supported:
            - `model_name`: string, custom name for the model.
            - `api_base`: string, custom base URL.
            - `api_key`: optional, your API key.
    - `critic_llm`: dictionary, description: Configuration for the evaluator LLM, same as above.
    - `embeddings`: dictionary, description: Configuration for the embedding model.
      - `model_name_or_path`: string, description: The name or path of the embedding model, e.g., "AI-ModelScope/m3e-base".

#### Execute Task
```python
from evalscope.run import run_task
from evalscope.utils.logger import get_logger
logger = get_logger()

# Run task
run_task(task_cfg=generate_testset_task_cfg) 
```

Using examples generated from the project's README documentation as follows:

<details><summary>Click to view the automatically generated dataset</summary>

``` json
[
    {
        "question": "What is the purpose of specifying the 'template type' when conducting a more customized evaluation using the evalscope/run.py command?",
        "contexts": [
            " to run the custom code? [y\/N]`, please type `y`. #### Basic Parameter Descriptions - `--model`: Specifies the `model_id` of the model on [ModelScope](https:\/\/modelscope.cn\/), allowing automatic download. For example, see the [Qwen2-0.5B-Instruct model link](https:\/\/modelscope.cn\/models\/qwen\/Qwen2-0.5B-Instruct\/summary); you can also use a local path, such as `\/path\/to\/model`. - `--template-type`: Specifies the template type corresponding to the model. Refer to the `Default Template` field in the [template table](https:\/\/swift.readthedocs.io\/en\/latest\/Instruction\/Supported-models-datasets.html#llm) for filling in this field. - `--datasets`: The dataset name, allowing multiple datasets to be specified, separated by spaces; these datasets will be automatically downloaded. Refer to the [supported datasets list](https:\/\/evalscope.readthedocs.io\/en\/latest\/get_started\/supported_dataset.html) for available options. ### 2. Parameterized Evaluation If you wish to conduct a more customized evaluation, such as modifying model parameters or dataset parameters, you can use the following commands: **Example 1:** ```shell python evalscope\/run.py \\ --model qwen\/Qwen2-0.5B-Instruct \\ --template-type qwen \\ --model-args revision=master,precision=torch.float16,device_map=auto \\ --datasets gsm8k ceval \\ --use-cache true \\ --limit 10 ``` **Example 2:** ```shell python evalscope\/run.py \\ --model qwen\/Qwen2-0.5B-Instruct \\ --template-type qwen \\ --generation-config do_sample=false,temperature=0.0 \\ --datasets ceval \\ --dataset-args '{\"ceval\": {\"few_shot_num\": 0, \"few_shot_random\": false}}' \\ --limit 10 ``` #### Parameter Descriptions In addition to the three [basic parameters](#basic-parameter-descriptions), the other parameters are as follows: - `--model-args`: Model loading parameters, separated by commas, in `key=value` format. - `--generation-config`: Generation parameters, separated by commas, in `key=value` format. - `do_sample`: Whether to use sampling, default is `false`. - `max_new_tokens`: Maximum generation length, default is 1024. - `temperature`: Sampling temperature. - `top_p`: Sampling threshold. - `top_k`: Sampling threshold. - `--use-cache`: Whether to use local cache, default is `false`. If set to `true`, previously evaluated model and dataset combinations will not be evaluated again, and will be read directly from the local cache. - `--dataset-args`: Evaluation dataset configuration parameters, provided in JSON format, where the key is the dataset name and the value is the parameter; note that these must correspond one-to-one with the values in `--datasets`. - `--few_shot_num`: Number of few-shot examples. - `--few_shot_random`: Whether to randomly sample few-shot data; if not specified, defaults to `true`. - `--limit`: Maximum number of evaluation samples per dataset; if not specified, all will be evaluated, which is useful for quick validation. ### 3. Use the run_task Function to Submit an Evaluation Task Using the `run_task` function to submit an evaluation task requires the same parameters as the command line. You need to pass a dictionary as the parameter, which includes the following fields: #### 1. Configuration Task Dictionary Parameters ```python import torch from evalscope.constants import DEFAULT_ROOT_CACHE_DIR # Example your_task_cfg = { 'model_args': {'revision': None, 'precision': torch.float16, 'device_map': 'auto'}, 'generation_config': {'do_sample': False, 'repetition_penalty': 1.0, 'max_new_tokens': 512}, 'dataset_args': {}, 'dry_run': False, 'model': 'qwen\/Qwen2-0.5B-Instruct', 'template_type': 'qwen', 'datasets': ['arc', 'hellasw"
        ],
        "ground_truth": "The purpose of specifying the 'template type' when conducting a more customized evaluation using the evalscope/run.py command is to associate the correct evaluation template with the chosen model. This ensures that the evaluation process uses the appropriate instructions, prompts, and settings tailored for that specific model, enhancing the relevance and effectiveness of the evaluation.",
        "answer": "Specifying the 'template type' when conducting a more customized evaluation using the evalscope/run.py command is crucial because it corresponds to the specific structure and requirements of the model being used. This ensures that the evaluation process aligns correctly with the model's capabilities and expected input format, leading to accurate and meaningful results."
    },
    {
        "question": "What automated outputs does the EvalScope framework produce following a model evaluation?",
        "contexts": [
            "\ud83d\udcd6 Documents &nbsp | &nbsp \ud83d\udcd6 \u4e2d\u6587\u6587\u6863",
            "## \ud83d\udccb Table of Contents - [Introduction](#introduction) - [News](#News) - [Installation](#installation) - [Quick Start](#quick-start) - [Evaluation Backend](#evaluation-backend) - [Custom Dataset Evaluation](#custom-dataset-evaluation) - [Offline Evaluation](#offline-evaluation) - [Arena Mode](#arena-mode) - [Model Serving Performance Evaluation](#Model-Serving-Performance-Evaluation) - [Leaderboard](#leaderboard) ## \ud83d\udcdd Introduction Large Model (including Large Language Models, Multi-modal Large Language Models) evaluation has become a critical process for assessing and improving LLMs. To better support the evaluation of large models, we propose the EvalScope framework. ### Framework Features - **Benchmark Datasets**: Preloaded with several commonly used test benchmarks, including MMLU, CMMLU, C-Eval, GSM8K, ARC, HellaSwag, TruthfulQA, MATH, HumanEval, etc. - **Evaluation Metrics**: Implements various commonly used evaluation metrics. - **Model Access**: A unified model access mechanism that is compatible with the Generate and Chat interfaces of multiple model families. - **Automated Evaluation**: Includes automatic evaluation of objective questions and complex task evaluation using expert models. - **Evaluation Reports**: Automatically generates evaluation reports. - **Arena Mode**: Used for comparisons between models and objective evaluation of models, supporting various evaluation modes, including: - **Single mode**: Scoring a single model. - **Pairwise-baseline mode**: Comparing against a baseline model. - **Pairwise (all) mode**: Pairwise comparison among all models. - **Visualization Tools**: Provides intuitive displays of evaluation results. - **Model Performance Evaluation**: Offers a performance testing tool for model inference services and detailed statistics, see [Model Performance Evaluation Documentation](https:\/\/evalscope.readthedocs.io\/en\/latest\/user_guides\/stress_test.html). - **OpenCompass Integration**: Supports OpenCompass as the evaluation backend, providing advanced encapsulation and task simplification, allowing for easier task submission for evaluation. - **VLMEvalKit Integration**: Supports VLMEvalKit as the evaluation backend, facilitating the initiation of multi-modal evaluation tasks, supporting various multi-modal models and datasets. - **Full-Link Support**: Through seamless integration with the [ms-swift](https:\/\/github.com\/modelscope\/ms-swift) training framework, provides a one-stop development process for model training, model deployment, model evaluation, and report viewing, enhancing user development efficiency."
        ],
        "ground_truth": "The EvalScope framework automatically generates evaluation reports following a model evaluation.",
        "answer": "The EvalScope framework produces automated evaluation reports following a model evaluation. These reports include results from automatic evaluations of objective questions and complex task evaluations using expert models."
    },
    {
        "question": "How does EvalScope framework create detailed reports for big models using benchmark datasets, metrics, and backends like OpenCompass or VLMEvalKit?",
        "contexts": [
            "## \ud83d\udccb Table of Contents - [Introduction](#introduction) - [News](#News) - [Installation](#installation) - [Quick Start](#quick-start) - [Evaluation Backend](#evaluation-backend) - [Custom Dataset Evaluation](#custom-dataset-evaluation) - [Offline Evaluation](#offline-evaluation) - [Arena Mode](#arena-mode) - [Model Serving Performance Evaluation](#Model-Serving-Performance-Evaluation) - [Leaderboard](#leaderboard) ## \ud83d\udcdd Introduction Large Model (including Large Language Models, Multi-modal Large Language Models) evaluation has become a critical process for assessing and improving LLMs. To better support the evaluation of large models, we propose the EvalScope framework. ### Framework Features - **Benchmark Datasets**: Preloaded with several commonly used test benchmarks, including MMLU, CMMLU, C-Eval, GSM8K, ARC, HellaSwag, TruthfulQA, MATH, HumanEval, etc. - **Evaluation Metrics**: Implements various commonly used evaluation metrics. - **Model Access**: A unified model access mechanism that is compatible with the Generate and Chat interfaces of multiple model families. - **Automated Evaluation**: Includes automatic evaluation of objective questions and complex task evaluation using expert models. - **Evaluation Reports**: Automatically generates evaluation reports. - **Arena Mode**: Used for comparisons between models and objective evaluation of models, supporting various evaluation modes, including: - **Single mode**: Scoring a single model. - **Pairwise-baseline mode**: Comparing against a baseline model. - **Pairwise (all) mode**: Pairwise comparison among all models. - **Visualization Tools**: Provides intuitive displays of evaluation results. - **Model Performance Evaluation**: Offers a performance testing tool for model inference services and detailed statistics, see [Model Performance Evaluation Documentation](https:\/\/evalscope.readthedocs.io\/en\/latest\/user_guides\/stress_test.html). - **OpenCompass Integration**: Supports OpenCompass as the evaluation backend, providing advanced encapsulation and task simplification, allowing for easier task submission for evaluation. - **VLMEvalKit Integration**: Supports VLMEvalKit as the evaluation backend, facilitating the initiation of multi-modal evaluation tasks, supporting various multi-modal models and datasets. - **Full-Link Support**: Through seamless integration with the [ms-swift](https:\/\/github.com\/modelscope\/ms-swift) training framework, provides a one-stop development process for model training, model deployment, model evaluation, and report viewing, enhancing user development efficiency."
        ],
        "ground_truth": "EvalScope framework creates detailed reports for big models by utilizing benchmark datasets, implementing evaluation metrics, and integrating with backends such as OpenCompass or VLMEvalKit. It supports automated evaluation of objective questions and complex tasks, offers a unified model access mechanism compatible with various model families, and includes features like Arena Mode for model comparisons. Additionally, it generates evaluation reports and provides visualization tools for intuitive result displays. Seamless integration with frameworks like ms-swift enhances the development process, offering a comprehensive solution for model training, deployment, evaluation, and reporting.",
        "answer": "EvalScope framework creates detailed reports for big models by utilizing benchmark datasets, metrics, and backends such as OpenCompass or VLMEvalKit. It automates the evaluation process, generating comprehensive reports that reflect the model's performance across various tasks and metrics. This is achieved through its integrated backend support, which simplifies task submission and enables advanced encapsulation, making it easier to assess model capabilities in both single and comparative evaluations."
    },
    {
        "question": "How can one tailor model assessments in `evalscope`, and which parameters related to model and dataset configurations can be modified for customization, given the tool's capabilities for model-specific loading, generation, and evaluation dataset setup?",
        "contexts": [
            " to run the custom code? [y\/N]`, please type `y`. #### Basic Parameter Descriptions - `--model`: Specifies the `model_id` of the model on [ModelScope](https:\/\/modelscope.cn\/), allowing automatic download. For example, see the [Qwen2-0.5B-Instruct model link](https:\/\/modelscope.cn\/models\/qwen\/Qwen2-0.5B-Instruct\/summary); you can also use a local path, such as `\/path\/to\/model`. - `--template-type`: Specifies the template type corresponding to the model. Refer to the `Default Template` field in the [template table](https:\/\/swift.readthedocs.io\/en\/latest\/Instruction\/Supported-models-datasets.html#llm) for filling in this field. - `--datasets`: The dataset name, allowing multiple datasets to be specified, separated by spaces; these datasets will be automatically downloaded. Refer to the [supported datasets list](https:\/\/evalscope.readthedocs.io\/en\/latest\/get_started\/supported_dataset.html) for available options. ### 2. Parameterized Evaluation If you wish to conduct a more customized evaluation, such as modifying model parameters or dataset parameters, you can use the following commands: **Example 1:** ```shell python evalscope\/run.py \\ --model qwen\/Qwen2-0.5B-Instruct \\ --template-type qwen \\ --model-args revision=master,precision=torch.float16,device_map=auto \\ --datasets gsm8k ceval \\ --use-cache true \\ --limit 10 ``` **Example 2:** ```shell python evalscope\/run.py \\ --model qwen\/Qwen2-0.5B-Instruct \\ --template-type qwen \\ --generation-config do_sample=false,temperature=0.0 \\ --datasets ceval \\ --dataset-args '{\"ceval\": {\"few_shot_num\": 0, \"few_shot_random\": false}}' \\ --limit 10 ``` #### Parameter Descriptions In addition to the three [basic parameters](#basic-parameter-descriptions), the other parameters are as follows: - `--model-args`: Model loading parameters, separated by commas, in `key=value` format. - `--generation-config`: Generation parameters, separated by commas, in `key=value` format. - `do_sample`: Whether to use sampling, default is `false`. - `max_new_tokens`: Maximum generation length, default is 1024. - `temperature`: Sampling temperature. - `top_p`: Sampling threshold. - `top_k`: Sampling threshold. - `--use-cache`: Whether to use local cache, default is `false`. If set to `true`, previously evaluated model and dataset combinations will not be evaluated again, and will be read directly from the local cache. - `--dataset-args`: Evaluation dataset configuration parameters, provided in JSON format, where the key is the dataset name and the value is the parameter; note that these must correspond one-to-one with the values in `--datasets`. - `--few_shot_num`: Number of few-shot examples. - `--few_shot_random`: Whether to randomly sample few-shot data; if not specified, defaults to `true`. - `--limit`: Maximum number of evaluation samples per dataset; if not specified, all will be evaluated, which is useful for quick validation. ### 3. Use the run_task Function to Submit an Evaluation Task Using the `run_task` function to submit an evaluation task requires the same parameters as the command line. You need to pass a dictionary as the parameter, which includes the following fields: #### 1. Configuration Task Dictionary Parameters ```python import torch from evalscope.constants import DEFAULT_ROOT_CACHE_DIR # Example your_task_cfg = { 'model_args': {'revision': None, 'precision': torch.float16, 'device_map': 'auto'}, 'generation_config': {'do_sample': False, 'repetition_penalty': 1.0, 'max_new_tokens': 512}, 'dataset_args': {}, 'dry_run': False, 'model': 'qwen\/Qwen2-0.5B-Instruct', 'template_type': 'qwen', 'datasets': ['arc', 'hellasw",
            "ag'], 'work_dir': DEFAULT_ROOT_CACHE_DIR, 'outputs': DEFAULT_ROOT_CACHE_DIR, 'mem_cache': False, 'dataset_hub': 'ModelScope', 'dataset_dir': DEFAULT_ROOT_CACHE_DIR, 'limit': 10, 'debug': False } ``` Here, `DEFAULT_ROOT_CACHE_DIR` is set to `'~\/.cache\/evalscope'`. #### 2. Execute Task with run_task ```python from evalscope.run import run_task run_task(task_cfg=your_task_cfg) ``` ## Evaluation Backend EvalScope supports using third-party evaluation frameworks to initiate evaluation tasks, which we call Evaluation Backend. Currently supported Evaluation Backend includes: - **Native**: EvalScope's own **default evaluation framework**, supporting various evaluation modes including single model evaluation, arena mode, and baseline model comparison mode. - [OpenCompass](https:\/\/github.com\/open-compass\/opencompass): Initiate OpenCompass evaluation tasks through EvalScope. Lightweight, easy to customize, supports seamless integration with the LLM fine-tuning framework ms-swift. [\ud83d\udcd6 User Guide](https:\/\/evalscope.readthedocs.io\/en\/latest\/user_guides\/opencompass_backend.html) - [VLMEvalKit](https:\/\/github.com\/open-compass\/VLMEvalKit): Initiate VLMEvalKit multimodal evaluation tasks through EvalScope. Supports various multimodal models and datasets, and offers seamless integration with the LLM fine-tuning framework ms-swift. [\ud83d\udcd6 User Guide](https:\/\/evalscope.readthedocs.io\/en\/latest\/user_guides\/vlmevalkit_backend.html) - **ThirdParty**: The third-party task, e.g. [ToolBench](https:\/\/evalscope.readthedocs.io\/en\/latest\/third_party\/toolbench.html), you can contribute your own evaluation task to EvalScope as third-party backend. ## Custom Dataset Evaluation EvalScope supports custom dataset evaluation. For detailed information, please refer to the Custom Dataset Evaluation [\ud83d\udcd6User Guide](https://evalscope.readthedocs.io/en/latest/advanced_guides/custom_dataset.html) ## Offline Evaluation You can use local dataset to evaluate the model without internet connection. Refer to: Offline Evaluation [\ud83d\udcd6 User Guide](https://evalscope.readthedocs.io/en/latest/user_guides/offline_evaluation.html) ## Arena Mode The Arena mode allows multiple candidate models to be evaluated through pairwise battles, and can choose to use the AI Enhanced Auto-Reviewer (AAR) automatic evaluation process or manual evaluation to obtain the evaluation report. Refer to: Arena Mode [\ud83d\udcd6 User Guide](https://evalscope.readthedocs.io/en/latest/user_guides/arena.html) ## Model Serving Performance Evaluation A stress testing tool that focuses on large language models and can be customized to support various data set formats and different API protocol formats. Refer to: Model Serving Performance Evaluation [\ud83d\udcd6 User Guide](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test.html) ## Leaderboard The LLM Leaderboard aims to provide an objective and comprehensive evaluation standard and platform to help researchers and developers understand and compare the performance of models on various tasks on ModelScope. Refer to: [Leaderboard](https://modelscope.cn/leaderboard/58/ranking?type=free) ## TO-DO List - [x] Agents evaluation - [x] vLLM - [ ] Distributed evaluating - [x] Multi-modal evaluation - [ ] Benchmarks - [ ] GAIA - [ ] GPQA - [x] MBPP - [ ] Auto-reviewer - [ ] Qwen-max"
        ],
        "ground_truth": "To tailor model assessments in `evalscope`, one can modify several parameters related to model and dataset configurations. For model-specific loading, `model-args` allows setting parameters like `revision`, `precision`, and `device_map`. For generation, `generation-config` lets users control aspects such as `do_sample`, `temperature`, `max_new_tokens`, and `top_p`. To customize evaluation datasets, `dataset-args` provides configuration options for each specified dataset. Additionally, parameters like `few_shot_num`, `few_shot_random`, and `limit` enable adjustments for few-shot learning and limiting the number of evaluation samples per dataset.",
        "answer": "To tailor model assessments in `evalscope`, you can modify parameters related to model and dataset configurations. Key parameters include:\n\n- `--model`: Specifies the model ID or local path.\n- `--template-type`: Corresponds to the model's template type.\n- `--datasets`: Allows specifying multiple datasets to evaluate, which are automatically downloaded.\n- `--model-args`: Model loading parameters, like `revision`, `precision`, and `device_map`.\n- `--generation-config`: Generation parameters, including `do_sample`, `temperature`, and `max_new_tokens`.\n- `--use-cache`: Determines if local cache should be used.\n- `--dataset-args`: Evaluation dataset configuration parameters, provided in JSON format.\n- `--few_shot_num`: Number of few-shot examples.\n- `--limit`: Maximum number of evaluation samples per dataset.\n\nThese parameters enable customization for model-specific loading, generation, and evaluation dataset setup."
    },
    {
        "question": "How does the OpenCompass Integration streamline task submission for evaluation compared to EvalScope and VLMEvalKit?",
        "contexts": [
            "## \ud83d\udccb Table of Contents - [Introduction](#introduction) - [News](#News) - [Installation](#installation) - [Quick Start](#quick-start) - [Evaluation Backend](#evaluation-backend) - [Custom Dataset Evaluation](#custom-dataset-evaluation) - [Offline Evaluation](#offline-evaluation) - [Arena Mode](#arena-mode) - [Model Serving Performance Evaluation](#Model-Serving-Performance-Evaluation) - [Leaderboard](#leaderboard) ## \ud83d\udcdd Introduction Large Model (including Large Language Models, Multi-modal Large Language Models) evaluation has become a critical process for assessing and improving LLMs. To better support the evaluation of large models, we propose the EvalScope framework. ### Framework Features - **Benchmark Datasets**: Preloaded with several commonly used test benchmarks, including MMLU, CMMLU, C-Eval, GSM8K, ARC, HellaSwag, TruthfulQA, MATH, HumanEval, etc. - **Evaluation Metrics**: Implements various commonly used evaluation metrics. - **Model Access**: A unified model access mechanism that is compatible with the Generate and Chat interfaces of multiple model families. - **Automated Evaluation**: Includes automatic evaluation of objective questions and complex task evaluation using expert models. - **Evaluation Reports**: Automatically generates evaluation reports. - **Arena Mode**: Used for comparisons between models and objective evaluation of models, supporting various evaluation modes, including: - **Single mode**: Scoring a single model. - **Pairwise-baseline mode**: Comparing against a baseline model. - **Pairwise (all) mode**: Pairwise comparison among all models. - **Visualization Tools**: Provides intuitive displays of evaluation results. - **Model Performance Evaluation**: Offers a performance testing tool for model inference services and detailed statistics, see [Model Performance Evaluation Documentation](https:\/\/evalscope.readthedocs.io\/en\/latest\/user_guides\/stress_test.html). - **OpenCompass Integration**: Supports OpenCompass as the evaluation backend, providing advanced encapsulation and task simplification, allowing for easier task submission for evaluation. - **VLMEvalKit Integration**: Supports VLMEvalKit as the evaluation backend, facilitating the initiation of multi-modal evaluation tasks, supporting various multi-modal models and datasets. - **Full-Link Support**: Through seamless integration with the [ms-swift](https:\/\/github.com\/modelscope\/ms-swift) training framework, provides a one-stop development process for model training, model deployment, model evaluation, and report viewing, enhancing user development efficiency."
        ],
        "ground_truth": "The OpenCompass Integration streamlines task submission for evaluation by providing advanced encapsulation and task simplification. This allows for easier task submission for evaluation, making the process more efficient and user-friendly.",
        "answer": "The OpenCompass Integration in EvalScope streamlines task submission for evaluation by offering advanced encapsulation and simplifying tasks, making it easier to submit tasks for assessment compared to EvalScope's direct use of the Evaluation Backend or VLMEvalKit."
    }
]
```
</details>

## Configuration for Evaluation Task
```python
eval_task_cfg = {
    "eval_backend": "RAGEval",
    "eval_config": {
        "tool": "RAGAS",
        "eval": {
            "testset_file": "outputs/testset.json",
            "critic_llm": {
                "model_name_or_path": "qwen/Qwen2-7B-Instruct",
                "template_type": "qwen",
            },
            "embeddings": {
                "model_name_or_path": "AI-ModelScope/m3e-base",
            },
            "metrics": [
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "answer_correctness",
            ],
        },
    },
}
```
The basic parameters are consistent with those of the automatic evaluation task, with the following differences:
- `testset_file`: Specifies the path to the evaluation dataset file, defaulting to `outputs/testset.json`.
- `metrics`: Specifies the evaluation metrics, refer to the [metrics list](https://docs.ragas.io/en/latest/concepts/metrics/index.html).

## Execute Evaluation
```python
from evalscope.run import run_task
from evalscope.utils.logger import get_logger
logger = get_logger()

# Run task
run_task(task_cfg=eval_task_cfg)
```

The output evaluation results are as follows:

```{figure} images/eval_result.png

Evaluation Results
```