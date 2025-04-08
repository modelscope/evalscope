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

## RAG Evaluation

### Dataset Preparation
An example of the evaluation dataset is as follows:
```json
[
    {
        "user_input": "When was the first Olympic Games held?",
        "retrieved_contexts": [
            "The first modern Olympic Games were held from April 6 to April 15, 1896, in Athens, Greece."
        ],
        "response": "The first modern Olympic Games were held on April 6, 1896.",
        "reference": "The first modern Olympic Games opened on April 6, 1896, in Athens, Greece."
    },
    {
        "user_input": "Which athlete has won the most Olympic gold medals?",
        "retrieved_contexts": [
            "Michael Phelps is the athlete with the most Olympic gold medals in history, having won a total of 23 gold medals."
        ],
        "response": "Michael Phelps has won the most Olympic gold medals.",
        "reference": "Michael Phelps is the athlete with the most Olympic gold medals, having won a total of 23 gold medals."
    }
]
```
The required fields include:
- user_input: User input
- response: Model generated answer
- retrieved_contexts: List of retrieved contexts
- reference: Standard answer

#### Automatically Generating Datasets
RAGAS provides the functionality to automatically generate test datasets. Users can specify parameters such as test set size, data distribution, LLM generators, and so on to automatically generate a test dataset. The specific steps are as follows:

```{figure} images/generation_process.png

Ragas adopts a novel approach to generating evaluation data. An ideal evaluation dataset should encompass various types of questions encountered in real-world applications, including questions of different difficulty levels. By default, large language models (LLMs) struggle to create diverse samples since they often follow common paths. Inspired by works like [Evol-Instruct](https://arxiv.org/abs/2304.12244), Ragas achieves this goal by adopting an evolutionary generation paradigm, where questions with different characteristics (such as reasoning, conditions, multiple contexts, etc.) are systematically constructed based on the provided document set. This method ensures comprehensive coverage of the various components' performance in your pipeline, making the evaluation process more robust.
```

**Configuration Task**
```python
generate_testset_task_cfg = {
    "eval_backend": "RAGEval",
    "eval_config": {
        "tool": "RAGAS",
        "testset_generation": {
            "docs": ["README.md"],
            "test_size": 10,
            "output_file": "outputs/testset.json",
            "knowledge_graph", "outputs/knowledge_graph.json",
            "generator_llm": {
                "model_name_or_path": "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4",
            },
            "embeddings": {
                "model_name_or_path": "AI-ModelScope/m3e-base",
            },
            "language": "english"
        }
    },
}
```
Configuration file description:

- `eval_backend`: `str`: The name of the evaluation backend, "RAGEval".
- `eval_config`: `dict`: Detailed information of the evaluation configuration.
  - `tool`: `str`: The name of the evaluation tool, "RAGAS".
  - `testset_generation`: `dict`: Configuration for test set generation.
    - `docs`: `list`: List of documents required for test set generation, e.g., ["README.md"].
    - `test_size`: `int`: Size of the generated test set, e.g., 5.
    - `output_file`: `str`: Path of the generated dataset output file, e.g., "outputs/testset.json".
    - `knowledge_graph`: `str`: The file path of the knowledge graph, e.g., "outputs/knowledge_graph.json". The knowledge graph generated during the document processing will be saved in this path. If a knowledge graph already exists at this path, it will be loaded directly, skipping the generation step of the knowledge graph.
    - `generator_llm`: `dict`: Configuration for the generator LLM:
        - If using a **local model**, the following parameters are supported:
            - `model_name_or_path`: `str`: The name or path of the generator model, for example, "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4". The model can be automatically downloaded from ModelScope. If a path is provided, the model will be loaded locally.
            - `generation_config`: `dict`: Generation configuration, for example, `{"temperature": 0.7}`.
        - If using an **API model**, the following parameters are supported:
            - `model_name`: `str`: The name of the custom model.
            - `api_base`: `str`: The base URL of the custom API, for example, "http://127.0.0.1:8000".
            - `api_key`: `Optional[str]`: Your API key, default is "EMPTY".

    - `embeddings`: `dict`: Configuration for the embedding model.
        - If using a **local model**, the following parameters are supported:
            - `model_name_or_path`: `str`: The name or path of the embedding model, for example, "AI-ModelScope/m3e-base".
        - If using an **API model**, the following parameters are supported:
            - `model_name`: `str`: Model name.
            - `api_base`: `str`: Model API service address.
            - `api_key`: `str`: Model API key.
            - `dimension`: `int`: Model output dimension.
    - `language`: `str`: Language, default is `english`, can be set to another language such as "chinese". The `generator_llm` will automatically translate prompts to the target language. The framework has pre-translated some prompts using `Qwen2.5-72B-Instruct`.

````{note}
`generator_llm` requires a model with strong instruction-following capabilities. Models with 7B parameters or fewer may encounter the following error during execution:
```
ragas.testset.transforms.engine - ERROR - unable to apply transformation: 'Generation' object has no attribute 'message'
```
This is because the model output format is incorrect, leading to parsing errors. In this case, please try using a larger model, such as `Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4`, or proprietary models like `GPT-4o`.
````

````{tip}
If you encounter the following error, or if the dataset generation is not effective, it indicates that there may be issues with the `unstructured` library processing the document. You can manually preprocess the input document into txt format.
```
ValueError: Documents appear to be too short (i.e., 100 tokens or less). Please provide longer documents.
```
````

**Execute Task**
```python
from evalscope.run import run_task
from evalscope.utils.logger import get_logger
logger = get_logger()

# Run task
run_task(task_cfg=generate_testset_task_cfg)
```

Using examples generated from the project's `README.md` documentation as follows:

<details><summary>Click to view the automatically generated dataset</summary>

``` json
[
    {
        "user_input":"What are the key sectoins covered in the documet's table of contens?",
        "retrieved_contexts":[
            "## \ud83d\udccb Table of Contents - [Introduction](#introduction) - [News](#News) - [Installation](#installation) - [Quick Start](#quick-start) - [Evaluation Backend](#evaluation-backend) - [Custom Dataset Evaluation](#custom-dataset-evaluation) - [Offline Evaluation](#offline-evaluation) - [Arena Mode](#arena-mode) - [Model Serving Performance Evaluation](#Model-Serving-Performance-Evaluation) - [Leaderboard](#leaderboard) "
        ],
        "response":"The document's table of contents covers sections such as Introduction, News, Installation, Quick Start, Evaluation Backend, Custom Dataset Evaluation, Offline Evaluation, Arena Mode, Model Serving Performance Evaluation, and Leaderboard.",
        "reference":"The key sections covered in the document's table of contents include Introduction, News, Installation, Quick Start, Evaluation Backend, Custom Dataset Evaluation, Offline Evaluation, Arena Mode, Model Serving Performance Evaluation, and Leaderboard."
    },
    {
        "user_input":"What EvalScope mean for model eval?",
        "retrieved_contexts":[
            "## \ud83d\udcdd Introduction EvalScope is the official model evaluation and performance benchmarking framework launched by the [ModelScope](https:\/\/modelscope.cn\/) community. It comes with built-in common benchmarks and evaluation metrics, such as MMLU, CMMLU, C-Eval, GSM8K, ARC, HellaSwag, TruthfulQA, MATH, and HumanEval. EvalScope supports various types of model evaluations, including LLMs, multimodal LLMs, embedding models, and reranker models. It is also applicable to multiple evaluation scenarios, such as end-to-end RAG evaluation, arena mode, and model inference performance stress testing. Moreover, with the seamless integration of the ms-swift training framework, evaluations can be initiated with a single click, providing full end-to-end support from model training to evaluation \ud83d\ude80"
        ],
        "response":"EvalScope is a framework by ModelScope for evaluating and benchmarking model performance, supporting a wide range of models and evaluation scenarios. It integrates common benchmarks and metrics, and offers seamless evaluation processes from training to performance testing.",
        "reference":"EvalScope is a framework for model evaluation and performance benchmarking, supporting various types of models and evaluation scenarios, and providing seamless integration with the ms-swift training framework for easy and comprehensive model evaluation."
    },
    {
        "user_input":"key components architecture contribution evaluation performance measurement models",
        "retrieved_contexts":[
            "The architecture includes the following modules: 1. **Model Adapter**: The model adapter is used to convert the outputs of specific models into the format required by the framework, supporting both API call models and locally run models. 2. **Data Adapter**: The data adapter is responsible for converting and processing input data to meet various evaluation needs and formats. 3. **Evaluation Backend**: - **Native**: EvalScope\u2019s own **default evaluation framework**, supporting various evaluation modes, including single model evaluation, arena mode, baseline model comparison mode, etc. - **OpenCompass**: Supports [OpenCompass](https:\/\/github.com\/open-compass\/opencompass) as the evaluation backend, providing advanced encapsulation and task simplification, allowing you to submit tasks for evaluation more easily. - **VLMEvalKit**: Supports [VLMEvalKit](https:\/\/github.com\/open-compass\/VLMEvalKit) as the evaluation backend, enabling easy initiation of multi-modal evaluation tasks, supporting various multi-modal models and datasets. - **RAGEval**: Supports RAG evaluation, supporting independent evaluation of embedding models and rerankers using [MTEB\/CMTEB](https:\/\/evalscope.readthedocs.io\/en\/latest\/user_guides\/backend\/rageval_backend\/mteb.html), as well as end-to-end evaluation using [RAGAS](https:\/\/evalscope.readthedocs.io\/en\/latest\/user_guides\/backend\/rageval_backend\/ragas.html). - **ThirdParty**: Other third-party evaluation tasks, such as ToolBench. 4. **Performance Evaluator**: Model performance evaluation, responsible for measuring model inference service performance, including performance testing, stress testing, performance report generation, and visualization. 5. **Evaluation Report**: The final generated evaluation report summarizes the model's performance, which can be used for decision-making and further model optimization. 6. **Visualization**: Visualization results help users intuitively understand evaluation results, facilitating analysis and comparison of different model performances. ## "
        ],
        "response":"The key components of the architecture include the Model Adapter, Data Adapter, Evaluation Backend (with options like Native, OpenCompass, VLMEvalKit, RAGEval, and ThirdParty), Performance Evaluator, Evaluation Report, and Visualization. These components collectively support the conversion of model outputs, data processing, performance measurement, and result visualization.",
        "reference":"The key components of the architecture include the Model Adapter, Data Adapter, Evaluation Backend, Performance Evaluator, Evaluation Report, and Visualization. The Model Adapter converts model outputs to the required format, the Data Adapter processes input data, the Evaluation Backend supports various evaluation frameworks, the Performance Evaluator measures model performance, the Evaluation Report summarizes the results, and Visualization helps in understanding and comparing model performances."
    },
    {
        "user_input":"How do the features and capabilities of the EvalScope Framework compare across different reports, particularly in terms of model evaluation, performance benchmarking, and support for various evaluation scenarios such as RAG evaluation and custom dataset evaluations?",
        "retrieved_contexts":[
            "EvalScope Framework is a tool designed to evaluate and improve the scope and effectiveness of AI models and systems.",
            "The provided text appears to be a navigation or menu option, offering documents in both English and Chinese.",
            "EvalScope, launched by the ModelScope community, is a framework for model evaluation and performance benchmarking. It includes common benchmarks and metrics for various models like LLMs, multimodal LLMs, and embedding models. EvalScope supports multiple evaluation scenarios, including RAG evaluation, arena mode, and model inference performance testing. The framework integrates with the ms-swift training framework, allowing for easy one-click evaluations from training to performance testing.",
            "The text provides language options: English and Simplified Chinese.",
            "EvalScope is a comprehensive framework for evaluating AI models, featuring modules like Model Adapter, Data Adapter, and Evaluation Backend. It supports various backends such as Native, OpenCompass, VLMEvalKit, RAGEval, and ThirdParty for different evaluation needs. The framework also includes a Performance Evaluator, Evaluation Report, and Visualization tools. Recent updates include support for RAG evaluation, custom dataset evaluations, and long output generation. Installation can be done via pip or from source, and quick start guides are available for simple and parameterized evaluations."
        ],
        "response":"EvalScope Framework supports a wide range of evaluation scenarios, including RAG evaluation and custom dataset evaluations, with comprehensive features for model evaluation and performance benchmarking. It offers various backends and tools for different needs, ensuring flexibility and ease of use.",
        "reference":"EvalScope Framework is a comprehensive tool for model evaluation and performance benchmarking, supporting various models and multiple evaluation scenarios. It includes features like RAG evaluation, custom dataset evaluations, and performance testing. The framework is integrated with the ms-swift training framework and supports different backends for diverse evaluation needs, including Native, OpenCompass, VLMEvalKit, RAGEval, and ThirdParty. It also provides Performance Evaluator, Evaluation Report, and Visualization tools, making it a versatile solution for evaluating AI models."
    },
    {
        "user_input":"How do the features and functionalities of EvalScope Framework, as described in the provided summaries, compare in terms of model evaluation, performance benchmarking, and support for different languages and evaluation scenarios?",
        "retrieved_contexts":[
            "EvalScope Framework is a tool designed to evaluate and improve the scope and effectiveness of AI models and systems.",
            "The provided text appears to be a navigation or menu option, offering documents in both English and Chinese.",
            "EvalScope, launched by the ModelScope community, is a framework for model evaluation and performance benchmarking. It includes common benchmarks and metrics for various models like LLMs, multimodal LLMs, and embedding models. EvalScope supports multiple evaluation scenarios, including RAG evaluation, arena mode, and model inference performance testing. The framework integrates with the ms-swift training framework, allowing for easy one-click evaluations from training to performance testing.",
            "The text provides language options: English and Simplified Chinese.",
            "EvalScope is a comprehensive framework for evaluating AI models, featuring modules like Model Adapter, Data Adapter, and Evaluation Backend. It supports various backends such as Native, OpenCompass, VLMEvalKit, RAGEval, and ThirdParty for different evaluation needs. The framework also includes a Performance Evaluator, Evaluation Report, and Visualization tools. Recent updates include support for RAG evaluation, custom dataset evaluations, and long output generation. Installation can be done via pip or from source, and quick start guides are available for simple and parameterized evaluations."
        ],
        "response":"EvalScope Framework offers comprehensive model evaluation and performance benchmarking, supporting a wide range of models and evaluation scenarios, including RAG, arena mode, and performance testing, with multi-language support. It integrates multiple backends and tools for diverse evaluation needs, and recent updates enhance its functionality with RAG evaluation, custom datasets, and long output generation.",
        "reference":"EvalScope Framework is a comprehensive tool for model evaluation and performance benchmarking, supporting a wide range of models including LLMs, multimodal LLMs, and embedding models. It offers multiple evaluation scenarios such as RAG evaluation, arena mode, and model inference performance testing. The framework supports various backends and includes features like Model Adapter, Data Adapter, and Evaluation Backend. It also provides Performance Evaluator, Evaluation Report, and Visualization tools, with recent updates enhancing support for RAG evaluation, custom dataset evaluations, and long output generation. EvalScope is available in both English and Simplified Chinese, making it accessible for different language users."
    },
    {
        "user_input":"How do the documentation and community support for EvalScope Framework compare in terms of language options, evaluation features, and community contributions, as described in the provided summaries?",
        "retrieved_contexts":[
            "EvalScope Framework is a tool designed to evaluate and improve the scope and effectiveness of AI models and systems.",
            "The provided text appears to be a navigation or menu option, offering documents in both English and Chinese.",
            "EvalScope, launched by the ModelScope community, is a framework for model evaluation and performance benchmarking. It includes common benchmarks and metrics for various models like LLMs, multimodal LLMs, and embedding models. EvalScope supports multiple evaluation scenarios, including RAG evaluation, arena mode, and model inference performance testing. The framework integrates with the ms-swift training framework, allowing for easy one-click evaluations from training to performance testing.",
            "The text provides language options: English and Simplified Chinese.",
            "EvalScope is a comprehensive framework for evaluating AI models, featuring modules like Model Adapter, Data Adapter, and Evaluation Backend. It supports various backends such as Native, OpenCompass, VLMEvalKit, RAGEval, and ThirdParty for different evaluation needs. The framework also includes a Performance Evaluator, Evaluation Report, and Visualization tools. Recent updates include support for RAG evaluation, custom dataset evaluations, and long output generation. Installation can be done via pip or from source, and quick start guides are available for simple and parameterized evaluations."
        ],
        "response":"EvalScope Framework offers documentation in English and Simplified Chinese, supports a wide range of evaluation features including RAG evaluation and performance testing, and has community contributions through the ModelScope community, including integrations and updates.",
        "reference":"EvalScope Framework offers documentation in both English and Chinese, supporting a wide range of evaluation features including common benchmarks, metrics for various models, and multiple evaluation scenarios. The community contributions are evident through the integration with the ms-swift training framework, recent updates, and the availability of quick start guides. This comprehensive framework supports various backends and includes tools for performance evaluation, reporting, and visualization."
    },
    {
        "user_input":"What is the purpose of offline evaluation in the context of using a local dataset?",
        "retrieved_contexts":[
            "Offline Evaluation You can use local dataset to evaluate the model without internet connection. Refer to: Offline Evaluation [\ud83d\udcd6 User Guide](https:\/\/evalscope.readthedocs.io\/en\/latest\/user_guides\/offline_evaluation.html) ## "
        ],
        "response":"The purpose of offline evaluation with a local dataset is to assess model performance without needing internet access. This method allows for testing in environments where connectivity is limited or restricted.",
        "reference":"The purpose of offline evaluation in the context of using a local dataset is to evaluate the model without an internet connection."
    },
    {
        "user_input":"What is the purpose of Arena mode in evaluating candidate models?",
        "retrieved_contexts":[
            "Arena Mode The Arena mode allows multiple candidate models to be evaluated through pairwise battles, and can choose to use the AI Enhanced Auto-Reviewer (AAR) automatic evaluation process or manual evaluation to obtain the evaluation report. Refer to: Arena Mode [\ud83d\udcd6 User Guide](https:\/\/evalscope.readthedocs.io\/en\/latest\/user_guides\/arena.html) ## "
        ],
        "response":"The purpose of Arena mode is to evaluate multiple candidate models by setting them against each other in pairwise comparisons, using either automatic or manual evaluation methods to generate reports.",
        "reference":"The purpose of Arena mode is to evaluate multiple candidate models through pairwise battles, using either the AI Enhanced Auto-Reviewer (AAR) automatic evaluation process or manual evaluation to obtain the evaluation report."
    },
    {
        "user_input":"What does the LLM Leaderboard aim to evaluate and compare in terms of the performance of models?",
        "retrieved_contexts":[
            "Leaderboard The LLM Leaderboard aims to provide an objective and comprehensive evaluation standard and platform to help researchers and developers understand and compare the performance of models on various tasks on ModelScope. Refer to : [Leaderboard](https:\/\/modelscope.cn\/leaderboard\/58\/ranking?type=free) ## "
        ],
        "response":"The LLM Leaderboard aims to evaluate and compare the performance of language models on various tasks, providing an objective and comprehensive platform for researchers and developers.",
        "reference":"The LLM Leaderboard aims to evaluate and compare the performance of models on various tasks on ModelScope."
    },
    {
        "user_input":"What are the different API protocol formats supported by the stress testing tool for model serving performance evaluation?",
        "retrieved_contexts":[
            "Model Serving Performance Evaluation A stress testing tool that focuses on large language models and can be customized to support various data set formats and different API protocol formats. Refer to : Model Serving Performance Evaluation [\ud83d\udcd6 User Guide](https:\/\/evalscope.readthedocs.io\/en\/latest\/user_guides\/stress_test.html) ## "
        ],
        "response":"The stress testing tool for model serving performance evaluation supports various API protocol formats, but specific formats are not detailed in the provided context. Refer to the user guide for more detailed information.",
        "reference":"The stress testing tool for model serving performance evaluation supports various data set formats and different API protocol formats."
    }
]
```
</details>

### Configuration for Evaluation Task
```python
eval_task_cfg = {
    "eval_backend": "RAGEval",
    "eval_config": {
        "tool": "RAGAS",
        "eval": {
            "testset_file": "outputs/testset_with_answer.json",
            "critic_llm": {
                "model_name_or_path": "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4",
            },
            "embeddings": {
                "model_name_or_path": "AI-ModelScope/m3e-base",
            },
            "metrics": [
                "Faithfulness",
                "AnswerRelevancy",
                "ContextPrecision",
                "AnswerCorrectness",
            ],
            "language": "english"
        },
    },
}
```
The basic parameters are consistent with those of the automatic evaluation task, with the following differences:
- `critic_llm`: Configuration for the evaluation model, parameters are the same as `generator_llm`.
- `testset_file`: Specifies the path to the evaluation dataset file, default is `outputs/testset_with_answer.json`.
- `metrics`: Specifies the evaluation metrics, refer to the [metrics list](https://docs.ragas.io/en/latest/concepts/metrics/index.html).

### Execute Evaluation
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


## Multi-Modal Text-Image RAG Evaluation

This framework extends RAG evaluation to support multi-modal text-image evaluation, accommodating text-image contexts.

### Dataset Preparation

```json
[
    {
        "user_input": "What is the brand of the car in the image?",
        "retrieved_contexts": [
            "custom_eval/multimodal/images/tesla.jpg"
        ],
        "response": "Tesla is a car brand.",
        "reference": "The car brand in the image is Tesla."
    },
    {
        "user_input": "What about Tesla Model X?",
        "retrieved_contexts": [
            "custom_eval/multimodal/images/tesla.jpg"
        ],
        "response": "Cats are cute.",
        "reference": "The Tesla Model X is an electric SUV manufactured by Tesla."
    }
]
```

The required fields include:
- user_input: User input
- response: Model generated answer
- retrieved_contexts: List of retrieved contexts, which can be text or image paths (local or URL)
- reference: Standard answer

### Configuring the Evaluation Task
```python
multi_modal_task_cfg = {
    "eval_backend": "RAGEval",
    "eval_config": {
        "tool": "RAGAS",
        "eval": {
            "testset_file": "outputs/testset_multi_modal.json",
            "critic_llm": {
                "model_name": "gpt-4o",
                "api_base": "http://127.0.0.1:8088/v1",
                "api_key": "EMPTY",
            },
            "embeddings": {
                "model_name_or_path": "AI-ModelScope/bge-large-zh",
            },
            "metrics": [
                "MultiModalFaithfulness",
                "MultiModalRelevance",
            ],
        },
    },
}
```
The basic parameters are consistent with the RAG evaluation task configuration. The differences are as follows:
- `critic_llm`: Configuration for the evaluation model, parameters are the same as `generator_llm`. The model needs to support multi-modal interleaved text-image input.
- `metrics`: Specifies the evaluation metrics. Currently supports `MultiModalFaithfulness` and `MultiModalRelevance` for multi-modal text-image evaluation. For other metrics that do not involve multi-modal data, such as `AnswerCorrectness`, please refer to the [metrics list](https://docs.ragas.io/en/latest/concepts/metrics/index.html).
- `embeddings` is an optional parameter and may be omitted.

### Running the Evaluation
```python
from evalscope.run import run_task

run_task(task_cfg=multi_modal_task_cfg)
```

Output results:

```json
[
    {
        "user_input":"What is the brand of the car in the image?",
        "retrieved_contexts":[
            "custom_eval/multimodal/images/tesla.jpg"
        ],
        "response":"Tesla is a car brand.",
        "reference":"The car brand in the image is Tesla.",
        "faithful_rate":true,
        "relevance_rate":true
    },
    {
        "user_input":"What about Tesla Model X?",
        "retrieved_contexts":[
            "custom_eval/multimodal/images/tesla.jpg"
        ],
        "response":"Cats are cute.",
        "reference":"The Tesla Model X is an electric SUV manufactured by Tesla.",
        "faithful_rate":false,
        "relevance_rate":false
    }
]
```
