(ragas)=

# RAGAS

本框架支持[RAGAS](https://github.com/explodinggradients/ragas)（Retrieval Augmented Generation Assessment），一个专门用于评估检索增强生成（RAG）流程性能的框架。其具有的一些核心评估指标包括：

- 忠实性（Faithfulness）：衡量生成答案的事实准确性，确保生成内容的真实性和可靠性。
- 答案相关性（Answer Relevance）：评估答案与给定问题的相关性，验证响应是否直接解决了用户的查询。
- 上下文精准度（Context Precision）：评估用于生成答案的上下文的精确度，确保从上下文中选择了相关信息。
- 上下文相关性（Context Relevancy）：衡量所选上下文与问题的相关性，帮助改进上下文选择以提高答案的准确性。
- 上下文召回率（Context Recall）：评估检索到的相关上下文信息的完整性，确保没有遗漏关键上下文。

此外，RAGAS还提供了自动测试数据生成的工具，方便用户使用。

## 环境准备
安装依赖
```shell
pip install ragas
```

## 数据集准备
评测数据集示例如下：
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
需要的字段包括：
- question：问题
- answer：答案
- contexts：上下文列表
- ground_truth：标准答案

### 自动生成数据集
RAGAS提供了自动生成测试数据的功能，用户可以指定测试集大小、数据分布、生成器和 LLM 等参数，自动生成测试数据集。具体步骤如下：

```{figure} images/generation_process.png

Ragas采用了一种新颖的评估数据生成方法。理想的评估数据集应该涵盖在实际应用中遇到的各种类型的问题，包括不同难度等级的问题。默认情况下，大语言模型（LLMs）不擅长创建多样化的样本，因为它们往往遵循常见的路径。受到[Evol-Instruct](https://arxiv.org/abs/2304.12244)等工作的启发，Ragas通过采用进化生成范式来实现这一目标，在这一过程中，具有不同特征的问题（如推理、条件、多个上下文等）会根据提供的文档集被系统地构建。这种方法确保了对您管道中各个组件性能的全面覆盖，从而使评估过程更加稳健。
```

#### 配置任务
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
配置文件说明：
- `eval_backend`: 字符串, 描述: 评估后端的名称，例如 "RAGEval"。
- `eval_config`: 字典, 描述: 包含评估配置的详细信息。
  - `tool`: 字符串, 描述: 评估工具的名称，例如 "RAGAS"。
  - `testset_generation`: 字典, 描述: 测试集生成的配置。
    - `docs`: 列表, 描述: 测试集生成所需的文档列表，例如 ["README.md"]。
    - `test_size`: 整数, 描述: 生成测试集的大小，例如 5。
    - `output_file`: 字符串, 描述: 生成的输出文件路径，例如 "outputs/testset.json"。
    - `distribution`: 字典, 描述: 测试集内容的分布配置。
      - `simple`: 浮点数, 描述: 简单内容的分布比例，例如 0.5。
      - `multi_context`: 浮点数, 描述: 多上下文内容的分布比例，例如 0.4。
      - `reasoning`: 浮点数, 描述: 推理内容的分布比例，例如 0.1。
    - `generator_llm`: 字典, 描述: 生成器 LLM（大语言模型）的配置。
      - `model_name_or_path`: 字符串, 描述: 生成器模型的名称或路径，例如 "qwen/Qwen2-7B-Instruct"。
      - `template_type`: 字符串, 描述: 模板类型，例如 "qwen"。
      - `generation_config`: 字典, 描述: 生成配置，例如 `{"temperature": 0.7}`。
    - `critic_llm`: 字典, 描述: 评价器 LLM 的配置。
      - `model_name_or_path`: 字符串, 描述: 评价器模型的名称或路径，例如 "QwenCollection/Ragas-critic-llm-Qwen1.5-GPTQ"。
      - `template_type`: 字符串, 描述: 模板类型，例如 "qwen"。
      - `generation_config`: 字典, 描述: 生成配置，例如 `{"temperature": 0.7}`。
    - `embeddings`: 字典, 描述: 嵌入模型的配置。
      - `model_name_or_path`: 字符串, 描述: 嵌入模型的名称或路径，例如 "AI-ModelScope/m3e-base"。


#### 执行任务
```python
from evalscope.run import run_task
from evalscope.utils.logger import get_logger

logger = get_logger()

# Run task
run_task(task_cfg=generate_testset_task_cfg) 
```


使用本项目README文档生成的示例如下：

<details><summary>点击查看自动生成的数据集</summary>

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

## 配置评测任务

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
基本参数与自动评测任务配置一致，不同点在于：
- `testset_file` ：指定评测数据集文件路径，默认为 `outputs/testset.json`
- `metrics`: 指定评测指标，参考[metrics 列表](https://docs.ragas.io/en/latest/concepts/metrics/index.html)


## 执行评测

```python
from evalscope.run import run_task
from evalscope.utils.logger import get_logger

logger = get_logger()

# Run task
run_task(task_cfg=eval_task_cfg)
```
输出评测结果如下：

```{figure} images/eval_result.png

评测结果
```
