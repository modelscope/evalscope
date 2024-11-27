# Introduction
EvalScope is the official model evaluation and performance benchmarking framework launched by the [ModelScope](https://modelscope.cn/) community. It comes with built-in common benchmarks and evaluation metrics, such as MMLU, CMMLU, C-Eval, GSM8K, ARC, HellaSwag, TruthfulQA, MATH, and HumanEval. EvalScope supports various types of model evaluations, including LLMs, multimodal LLMs, embedding models, and reranker models. It is also applicable to multiple evaluation scenarios, such as end-to-end RAG evaluation, arena mode, and model inference performance stress testing. Moreover, with the seamless integration of the ms-swift training framework, evaluations can be initiated with a single click, providing full end-to-end support from model training to evaluation.

## Overall Architecture
![EvalScope Architecture Diagram](../_static/images/evalscope_framework.png)
*EvalScope Architecture Diagram.*

The architecture includes the following modules:
1. **Model Adapter**: The model adapter is used to convert the outputs of specific models into the format required by the framework, supporting both API call models and locally run models.
2. **Data Adapter**: The data adapter is responsible for converting and processing input data to meet various evaluation needs and formats.
3. **Evaluation Backend**: 
    - **Native**: EvalScope’s own **default evaluation framework**, supporting various evaluation modes, including single model evaluation, arena mode, baseline model comparison mode, etc.
    - **OpenCompass**: Supports [OpenCompass](https://github.com/open-compass/opencompass) as the evaluation backend, providing advanced encapsulation and task simplification, allowing you to submit tasks for evaluation more easily.
    - **VLMEvalKit**: Supports [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) as the evaluation backend, enabling easy initiation of multi-modal evaluation tasks, supporting various multi-modal models and datasets.
    - **ThirdParty**: Other third-party evaluation tasks, such as [ToolBench](../third_party/toolbench.md).
    - **RAGEval**: Supports RAG evaluation, supporting independent evaluation of embedding models and rerankers using [MTEB/CMTEB](../user_guides/backend/rageval_backend/mteb.md), as well as end-to-end evaluation using [RAGAS](../user_guides/backend/rageval_backend/ragas.md).
4. **Performance Evaluator**: Model performance evaluation, responsible for measuring model inference service performance, including performance testing, stress testing, performance report generation, and visualization.
5. **Evaluation Report**: The final generated evaluation report summarizes the model's performance, which can be used for decision-making and further model optimization.
6. **Visualization**: Visualization results help users intuitively understand evaluation results, facilitating analysis and comparison of different model performances.

## Framework Features
- **Benchmark Datasets**: Preloaded with several commonly used test benchmarks, including MMLU, CMMLU, C-Eval, GSM8K, ARC, HellaSwag, TruthfulQA, MATH, HumanEval, etc.
- **Evaluation Metrics**: Implements various commonly used evaluation metrics.
- **Model Access**: A unified model access mechanism that is compatible with the Generate and Chat interfaces of multiple model families.
- **Automated Evaluation**: Includes automatic evaluation of objective questions and complex task evaluation using expert models.
- **Evaluation Reports**: Automatically generates evaluation reports.
- **Arena Mode**: Used for comparisons between models and objective evaluation of models, supporting various evaluation modes, including:
  - **Single mode**: Scoring a single model.
  - **Pairwise-baseline mode**: Comparing against a baseline model.
  - **Pairwise (all) mode**: Pairwise comparison among all models.
- **Visualization Tools**: Provides intuitive displays of evaluation results.
- **Model Performance Evaluation**: Offers a performance testing tool for model inference services and detailed statistics, see [Model Performance Evaluation Documentation](../user_guides/stress_test/index.md).
- **OpenCompass Integration**: Supports OpenCompass as the evaluation backend, providing advanced encapsulation and task simplification, allowing for easier task submission for evaluation.
- **VLMEvalKit Integration**: Supports VLMEvalKit as the evaluation backend, facilitating the initiation of multi-modal evaluation tasks, supporting various multi-modal models and datasets.
- **Full-Link Support**: Through seamless integration with the [ms-swift](https://github.com/modelscope/ms-swift) training framework, provides a one-stop development process for model training, model deployment, model evaluation, and report viewing, enhancing user development efficiency.