(ragas)=

# RAGAS

本框架支持[RAGAS](https://github.com/explodinggradients/ragas)（Retrieval Augmented Generation Assessment），一个专门用于评测检索增强生成（RAG）流程性能的框架。其具有的一些核心评测指标包括：

- 忠实性（Faithfulness）：衡量生成答案的事实准确性，确保生成内容的真实性和可靠性。
- 答案相关性（Answer Relevance）：评测答案与给定问题的相关性，验证响应是否直接解决了用户的查询。
- 上下文精准度（Context Precision）：评测用于生成答案的上下文的精确度，确保从上下文中选择了相关信息。
- 上下文相关性（Context Relevancy）：衡量所选上下文与问题的相关性，帮助改进上下文选择以提高答案的准确性。
- 上下文召回率（Context Recall）：评测检索到的相关上下文信息的完整性，确保没有遗漏关键上下文。

此外，RAGAS还提供了自动测试数据生成的工具，方便用户使用。

## 环境准备
安装依赖
```shell
pip install evalscope[rag] -U
```

## RAG 评测

### 数据集准备
评测数据集示例如下：
```json
[
    {
        "user_input": "第一届奥运会是什么时候举行的？",
        "retrieved_contexts": [
            "第一届现代奥运会于1896年4月6日到4月15日在希腊雅典举行。"
        ],
        "response": "第一届现代奥运会于1896年4月6日举行。",
        "reference": "第一届现代奥运会于1896年4月6日在希腊雅典开幕。"
    },
    {
        "user_input": "哪位运动员赢得了最多的奥运金牌？",
        "retrieved_contexts": [
            "迈克尔·菲尔普斯是历史上获得奥运金牌最多的运动员，他共赢得了23枚奥运金牌。"
        ],
        "response": "迈克尔·菲尔普斯赢得了最多的奥运金牌。",
        "reference": "迈克尔·菲尔普斯是获得奥运金牌最多的运动员，共赢得23枚金牌。"
    }
]
```
需要的字段包括：
- user_input：用户输入
- response：模型生成的答案
- retrieved_contexts：检索得到的上下文列表
- reference：标准答案

#### 自动生成数据集
RAGAS提供了自动生成测试数据的功能，用户可以指定测试集大小、数据分布、LLM生成器 等参数，自动生成测试数据集。具体步骤如下：

```{figure} images/generation_process.png

Ragas采用了一种新颖的评测数据生成方法。理想的评测数据集应该涵盖在实际应用中遇到的各种类型的问题，包括不同难度等级的问题。默认情况下，大语言模型（LLMs）不擅长创建多样化的样本，因为它们往往遵循常见的路径。受到[Evol-Instruct](https://arxiv.org/abs/2304.12244)等工作的启发，Ragas通过采用进化生成范式来实现这一目标，在这一过程中，具有不同特征的问题（如推理、条件、多个上下文等）会根据提供的文档集被系统地构建。这种方法确保了对您管道中各个组件性能的全面覆盖，从而使评测过程更加稳健。
```

**配置任务**
```python
generate_testset_task_cfg = {
    "eval_backend": "RAGEval",
    "eval_config": {
        "tool": "RAGAS",
        "testset_generation": {
            "docs": ["README_zh.md"],
            "test_size": 10,
            "output_file": "outputs/testset.json",
            "knowledge_graph": "outputs/knowledge_graph.json",
            "generator_llm": {
                "model_name_or_path": "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4",
            },
            "embeddings": {
                "model_name_or_path": "AI-ModelScope/m3e-base",
            },
            "language": "chinese"
        }
    },
}

```
配置文件说明：
- `eval_backend`: `str`：评测后端的名称 "RAGEval"。
- `eval_config`: `dict`：包含评测配置的详细信息。
  - `tool`: `str`：评测工具的名称"RAGAS"。
  - `testset_generation`: `dict`：测试集生成的配置。
    - `docs`: `list`：测试集生成所需的文档列表，例如 ["README_zh.md"]。
    - `test_size`: `int`：生成测试集的大小，例如 5。
    - `output_file`: `str`：生成数据集的输出文件路径，例如 "outputs/testset.json"。
    - `knowledge_graph`: `str`：知识图谱文件路径，例如 "outputs/knowledge_graph.json"，文档处理过程中生成的知识图谱会保存在该路径下；若该路径已有知识图谱，则会直接加载知识图谱，跳过生成知识图谱的步骤。
    - `generator_llm`: `dict`：生成器LLM的配置：
      - 若使用**本地模型**，支持如下参数：
        - `model_name_or_path`: `str`：生成器模型的名称或路径，例如 "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4" 可以从 ModelScope 自动下载模型；填入路径则从本地加载模型。
        - `generation_config`: `dict`：生成配置，例如 `{"temperature": 0.7}`。
      - 若使用**API 模型**，支持如下参数：
        - `model_name`: `str`：自定义模型的名称。
        - `api_base`: `str`：自定义基础 URL，例如 "http://127.0.0.1:8000"。
        - `api_key`: `Optional[str]`：你的 API 密钥，默认为"EMPTY"。
    - `embeddings`: `dict`：嵌入模型的配置。
      - 若使用**本地模型**，支持如下参数：
        - `model_name_or_path`: `str`：嵌入模型的名称或路径，例如 "AI-ModelScope/m3e-base"。
      - 若使用**API 模型**，支持如下参数：
        - `model_name`: `str` 模型名称。
        - `api_base`: `str` 模型API服务地址。
        - `api_key`: `str` 模型API密钥。
        - `dimension`: `int` 模型输出维度。
    - `language`: `str`：语言，默认为`english`，可以设置为其他语言例如 "chinese"，会使用`generator_llm`自动将prompt翻译到目标语言，框架已使用`Qwen2.5-72B-Instruct`预先翻译了部分prompt。


````{note}
`generator_llm`需要为指令遵循能力较强的模型，7B及以下规模的模型在运行时可能会有如下报错信息，例如：
```
ragas.testset.transforms.engine - ERROR - unable to apply transformation: 'Generation' object has no attribute 'message'
```
这是因为模型输出的格式不对，导致解析出错，此时请尝试使用规模更大的模型，例如 `Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4`，或是闭源模型`GPT-4o`等。
````

````{tip}
若出现如下错误，或生成数据集效果不好，说明`unstructured`库处理文档时，可能存在问题，可自行预处理输入文档为txt格式
```
ValueError: Documents appears to be too short (ie 100 tokens or less). Please provide longer documents.
```
````

**执行任务**
```python
from evalscope.run import run_task
from evalscope.utils.logger import get_logger

logger = get_logger()

# Run task
run_task(task_cfg=generate_testset_task_cfg)
```


使用本项目`README_zh.md`文档生成的示例如下：

<details><summary>点击查看自动生成的数据集</summary>

``` json
[
    {
        "user_input":"在提供的上下文中，如何有效地组织和利用目录中的各个部分来实现最佳的评测体验？",
        "retrieved_contexts":[
            "📋 目录\n\n简介\n\n新闻\n\n环境准备\n\n快速开始\n\n使用其他评测后端\n\n自定义数据集评测\n\n离线环境评测\n\n竞技场模式\n\n性能评测工具\n\nLeaderboard榜单\n\n"
        ],
        "response":"在提供的上下文中，有效地组织和利用目录中的各个部分可以通过依次进行环境准备、快速开始、自定义数据集评测和使用性能评测工具来实现最佳的评测体验。根据需要，还可以探索离线环境评测和竞技场模式以获得更全面的评测结果。",
        "reference":"为了实现最佳的评测体验，可以有效地组织和利用目录中的各个部分：首先，通过'简介'和'新闻'了解评测的背景和最新动态；接着，按照'环境准备'和'快速开始'的指导进行初步设置；然后，根据需要选择'使用其他评测后端'、'自定义数据集评测'或'离线环境评测'等特定评测方式；此外，可以利用'竞技场模式'和'性能评测工具'进行更深入的评测；最后，通过'Leaderboard榜单'查看和比较评测结果。"
    },
    {
        "user_input":"How does EvalScope evaluate model performance?",
        "retrieved_contexts":[
            "📝 简介\n\nEvalScope是魔搭社区官方推出的模型评测与性能基准测试框架，内置多个常用测试基准和评测指标，如MMLU、CMMLU、C-Eval、GSM8K、ARC、HellaSwag、TruthfulQA、MATH和HumanEval等；支持多种类型的模型评测，包括LLM、多模态LLM、embedding模型和reranker模型。EvalScope还适用于多种评测场景，如端到端RAG评测、竞技场模式和模型推理性能压测等。此外，通过ms-swift训练框架的无缝集成，可一键发起评测，实现了模型训练到评测的全链路支持🚀\n\n"
        ],
        "response":"EvalScope通过内置的多个常用测试基准和评测指标来评测模型性能，支持多种类型的模型评测和多种评测场景。它还与ms-swift训练框架无缝集成，实现了一键评测功能。",
        "reference":"EvalScope通过内置多个常用测试基准和评测指标来评测模型性能，支持多种类型的模型评测和多种评测场景。"
    },
    {
        "user_input":"How does EvalScope's archtecture support model evalution?",
        "retrieved_contexts":[
            "EvalScope 整体架构图.\n\nEvalScope包括以下模块：\n\nModel Adapter: 模型适配器，用于将特定模型的输出转换为框架所需的格式，支持API调用的模型和本地运行的模型。\n\nData Adapter: 数据适配器，负责转换和处理输入数据，以便适应不同的评测需求和格式。\n\nEvaluation Backend:\n\nNative：EvalScope自身的默认评测框架，支持多种评测模式，包括单模型评测、竞技场模式、Baseline模型对比模式等。\n\nOpenCompass：支持OpenCompass作为评测后端，对其进行了高级封装和任务简化，您可以更轻松地提交任务进行评测。\n\nVLMEvalKit：支持VLMEvalKit作为评测后端，轻松发起多模态评测任务，支持多种多模态模型和数据集。\n\nRAGEval：支持RAG评测，支持使用MTEB\/CMTEB进行embedding模型和reranker的独立评测，以及使用RAGAS进行端到端评测。\n\nThirdParty：其他第三方评测任务，如ToolBench。\n\nPerformance Evaluator: 模型性能评测，负责具体衡量模型推理服务性能，包括性能评测、压力测试、性能评测报告生成、可视化。\n\nEvaluation Report: 最终生成的评测报告，总结模型的性能表现，报告可以用于决策和进一步的模型优化。\n\nVisualization: 可视化结果，帮助用户更直观地理解评测结果，便于分析和比较不同模型的表现。\n\n"
        ],
        "response":"EvalScope的架构通过模型适配器和数据适配器支持多种模型和数据格式的评测，并通过多种评测后端提供灵活的评测模式。性能评测模块和可视化工具进一步帮助衡量和展示模型的性能。",
        "reference":"EvalScope's architecture supports model evaluation through various modules: Model Adapter for converting model outputs, Data Adapter for processing input data, Evaluation Backend with multiple evaluation modes and support for different frameworks like OpenCompass, VLMEvalKit, and RAGEval, Performance Evaluator for measuring model performance, Evaluation Report for summarizing performance, and Visualization for understanding and comparing results."
    },
    {
        "user_input":"EvalScope在模型评测、性能基准测试、多模态模型和RAG评测方面的功能和效果如何比较？",
        "retrieved_contexts":[
            "EvalScope是魔搭社区推出的模型评测与性能基准测试框架，支持多种模型评测，包括LLM、多模态LLM等。其架构包括模型适配器、数据适配器、评测后端、性能评测工具等模块。EvalScope支持多种评测场景，如RAG评测、竞技场模式等，并与ms-swift训练框架无缝集成。用户可以通过pip或源码安装EvalScope，并使用简单或自定义参数进行模型评测。EvalScope还提供了可视化工具和评测报告，帮助用户分析和优化模型性能。"
        ],
        "response":"EvalScope在模型评测、性能基准测试、多模态模型和RAG评测方面提供全面支持，具有模块化架构和无缝集成功能。它通过可视化工具和评测报告帮助用户分析和优化模型性能。",
        "reference":"EvalScope在模型评测、性能基准测试、多模态模型和RAG评测方面提供了多种功能和效果。它支持多种模型评测，包括LLM和多模态LLM，具有模型适配器、数据适配器、评测后端和性能评测工具等模块。EvalScope支持RAG评测和竞技场模式，并与ms-swift训练框架无缝集成，提供可视化工具和评测报告，帮助用户分析和优化模型性能。"
    },
    {
        "user_input":"在不同报告中，EvalScope在模型评测、性能基准测试、多模态模型和RAG评测方面的功能和效果如何比较？",
        "retrieved_contexts":[
            "EvalScope是魔搭社区推出的模型评测与性能基准测试框架，支持多种模型评测，包括LLM、多模态LLM等。其架构包括模型适配器、数据适配器、评测后端、性能评测工具等模块。EvalScope支持多种评测场景，如RAG评测、竞技场模式等，并与ms-swift训练框架无缝集成。用户可以通过pip或源码安装EvalScope，并使用简单或自定义参数进行模型评测。EvalScope还提供了可视化工具和评测报告，帮助用户分析和优化模型性能。"
        ],
        "response":"EvalScope在模型评测、性能基准测试、多模态模型和RAG评测方面功能全面，支持多种评测场景，并提供可视化工具和评测报告，帮助用户分析和优化模型性能。与ms-swift训练框架的无缝集成进一步提升了其使用效果。",
        "reference":"EvalScope在模型评测、性能基准测试、多模态模型和RAG评测方面表现出色。它支持多种模型评测，包括LLM和多模态LLM，并提供了丰富的评测场景和工具，如RAG评测和竞技场模式。EvalScope的架构模块化，易于集成和使用，用户可以通过pip或源码安装，并使用简单或自定义参数进行评测。此外，EvalScope还提供可视化工具和评测报告，帮助用户分析和优化模型性能。"
    },
    {
        "user_input":"在不同报告中，EvalScope框架在多模态模型的模型评测、性能基准测试和RAG评测中的应用和效果如何比较？",
        "retrieved_contexts":[
            "EvalScope是魔搭社区推出的模型评测与性能基准测试框架，支持多种模型评测，包括LLM、多模态LLM等。其架构包括模型适配器、数据适配器、评测后端、性能评测工具等模块。EvalScope支持多种评测场景，如RAG评测、竞技场模式等，并与ms-swift训练框架无缝集成。用户可以通过pip或源码安装EvalScope，并使用简单或自定义参数进行模型评测。EvalScope还提供了可视化工具和评测报告，帮助用户分析和优化模型性能。"
        ],
        "response":"我不知道。",
        "reference":"EvalScope框架在多模态模型的模型评测、性能基准测试和RAG评测中应用广泛，提供了多种评测场景和工具，帮助用户分析和优化模型性能。"
    },
    {
        "user_input":"如何在自定义数据集上进行评测？",
        "retrieved_contexts":[
            "🎉 新闻\n\n🔥 [2024.10.8] 支持RAG评测，包括使用MTEB\/CMTEB进行embedding模型和reranker的独立评测，以及使用RAGAS进行端到端评测。\n\n🔥 [2024.09.18] 我们的文档增加了博客模块，包含一些评测相关的技术调研和分享，欢迎📖阅读\n\n🔥 [2024.09.12] 支持 LongWriter 评测，您可以使用基准测试 LongBench-Write 来评测长输出的质量以及输出长度。\n\n🔥 [2024.08.30] 支持自定义数据集评测，包括文本数据集和多模态图文数据集。\n\n🔥 [2024.08.20] 更新了官方文档，包括快速上手、最佳实践和常见问题等，欢迎📖阅读。\n\n🔥 [2024.08.09] 简化安装方式，支持pypi安装vlmeval相关依赖；优化多模态模型评测体验，基于OpenAI API方式的评测链路，最高加速10倍。\n\n🔥 [2024.07.31] 重要修改：llmuses包名修改为evalscope，请同步修改您的代码。\n\n🔥 [2024.07.26] 支持VLMEvalKit作为第三方评测框架，发起多模态模型评测任务。\n\n🔥 [2024.06.29] 支持OpenCompass作为第三方评测框架，我们对其进行了高级封装，支持pip方式安装，简化了评测任务配置。\n\n🔥 [2024.06.13] EvalScope与微调框架SWIFT进行无缝对接，提供LLM从训练到评测的全链路支持 。\n\n🔥 [2024.06.13] 接入Agent评测集ToolBench。\n\n"
        ],
        "response":"您可以使用支持自定义数据集评测的功能来评测文本数据集和多模态图文数据集的表现。具体的评测工具和方法可以参考相关文档或使用第三方评测框架。",
        "reference":"在自定义数据集上进行评测可以通过支持的评测工具和框架来实现。例如，可以使用RAG评测、LongWriter评测、VLMEvalKit、OpenCompass等工具来评测文本数据集和多模态图文数据集的性能。"
    },
    {
        "user_input":"如何通过源码安装evalscope及其相关依赖？",
        "retrieved_contexts":[
            "🛠️ 环境准备\n\n方式1. 使用pip安装\n\n我们推荐使用conda来管理环境，并使用pip安装依赖: 1. 创建conda环境 (可选) ```shell\n\n建议使用 python 3.10\n\nconda create -n evalscope python=3.10\n\n激活conda环境\n\nconda activate evalscope 2. pip安装依赖shell pip install evalscope # 安装 Native backend (默认)\n\n额外选项\n\npip install evalscope[opencompass] # 安装 OpenCompass backend pip install evalscope[vlmeval] # 安装 VLMEvalKit backend pip install evalscope[all] # 安装所有 backends (Native, OpenCompass, VLMEvalKit) ```\n\n[!WARNING] 由于项目更名为evalscope，对于v0.4.3或更早版本，您可以使用以下命令安装： shell pip install llmuses<=0.4.3 使用llmuses导入相关依赖： python from llmuses import ...\n\n方式2. 使用源码安装\n\n下载源码 shell git clone https:\/\/github.com\/modelscope\/evalscope.git\n\n安装依赖 ```shell cd evalscope\/\n\npip install -e . # 安装 Native backend\n\n额外选项\n\npip install -e '.[opencompass]' # 安装 OpenCompass backend pip install -e '.[vlmeval]' # 安装 VLMEvalKit backend pip install -e '.[all]' # 安装所有 backends (Native, OpenCompass, VLMEvalKit) ```\n\n"
        ],
        "response":"通过源码安装evalscope及其相关依赖，可以先克隆源码仓库，然后进入目录并使用`pip install -e .`命令安装。若需安装额外的后端依赖，可以使用`pip install -e '.[opencompass]'`、`pip install -e '.[vlmeval]'`或`pip install -e '.[all]'`。",
        "reference":"通过源码安装evalscope及其相关依赖的步骤如下：\n\n1. 下载源码：\n   ```shell\n   git clone https:\/\/github.com\/modelscope\/evalscope.git\n   ```\n\n2. 安装依赖：\n   ```shell\n   cd evalscope\/\n   pip install -e . # 安装 Native backend\n   ```\n\n3. 额外选项：\n   - 安装 OpenCompass backend：\n     ```shell\n     pip install -e '.[opencompass]'\n     ```\n   - 安装 VLMEvalKit backend：\n     ```shell\n     pip install -e '.[vlmeval]'\n     ```\n   - 安装所有 backends (Native, OpenCompass, VLMEvalKit)：\n     ```shell\n     pip install -e '.[all]'\n     ```"
    },
    {
        "user_input":"在RAG评测中，embedding模型的作用是什么？",
        "retrieved_contexts":[
            "🎉 新闻\n\n🔥 [2024.10.8] 支持RAG评测，包括使用MTEB\/CMTEB进行embedding模型和reranker的独立评测，以及使用RAGAS进行端到端评测。\n\n🔥 [2024.09.18] 我们的文档增加了博客模块，包含一些评测相关的技术调研和分享，欢迎📖阅读\n\n🔥 [2024.09.12] 支持 LongWriter 评测，您可以使用基准测试 LongBench-Write 来评测长输出的质量以及输出长度。\n\n🔥 [2024.08.30] 支持自定义数据集评测，包括文本数据集和多模态图文数据集。\n\n🔥 [2024.08.20] 更新了官方文档，包括快速上手、最佳实践和常见问题等，欢迎📖阅读。\n\n🔥 [2024.08.09] 简化安装方式，支持pypi安装vlmeval相关依赖；优化多模态模型评测体验，基于OpenAI API方式的评测链路，最高加速10倍。\n\n🔥 [2024.07.31] 重要修改：llmuses包名修改为evalscope，请同步修改您的代码。\n\n🔥 [2024.07.26] 支持VLMEvalKit作为第三方评测框架，发起多模态模型评测任务。\n\n🔥 [2024.06.29] 支持OpenCompass作为第三方评测框架，我们对其进行了高级封装，支持pip方式安装，简化了评测任务配置。\n\n🔥 [2024.06.13] EvalScope与微调框架SWIFT进行无缝对接，提供LLM从训练到评测的全链路支持 。\n\n🔥 [2024.06.13] 接入Agent评测集ToolBench。\n\n"
        ],
        "response":"在RAG评测中，embedding模型用于生成文本的向量表示，以便在检索时进行相似性匹配和排序。它是实现信息检索和知识集成的重要组成部分。",
        "reference":"在RAG评测中，embedding模型用于进行独立评测，帮助评测模型的性能。"
    },
    {
        "user_input":"EvalScope的整体架构图中包含哪些模块，它们各自的功能是什么？",
        "retrieved_contexts":[
            "EvalScope 整体架构图.\n\nEvalScope包括以下模块：\n\nModel Adapter: 模型适配器，用于将特定模型的输出转换为框架所需的格式，支持API调用的模型和本地运行的模型。\n\nData Adapter: 数据适配器，负责转换和处理输入数据，以便适应不同的评测需求和格式。\n\nEvaluation Backend:\n\nNative：EvalScope自身的默认评测框架，支持多种评测模式，包括单模型评测、竞技场模式、Baseline模型对比模式等。\n\nOpenCompass：支持OpenCompass作为评测后端，对其进行了高级封装和任务简化，您可以更轻松地提交任务进行评测。\n\nVLMEvalKit：支持VLMEvalKit作为评测后端，轻松发起多模态评测任务，支持多种多模态模型和数据集。\n\nRAGEval：支持RAG评测，支持使用MTEB\/CMTEB进行embedding模型和reranker的独立评测，以及使用RAGAS进行端到端评测。\n\nThirdParty：其他第三方评测任务，如ToolBench。\n\nPerformance Evaluator: 模型性能评测，负责具体衡量模型推理服务性能，包括性能评测、压力测试、性能评测报告生成、可视化。\n\nEvaluation Report: 最终生成的评测报告，总结模型的性能表现，报告可以用于决策和进一步的模型优化。\n\nVisualization: 可视化结果，帮助用户更直观地理解评测结果，便于分析和比较不同模型的表现。\n\n"
        ],
        "response":"EvalScope包括以下模块：Model Adapter，用于将模型输出转换为所需格式；Data Adapter，负责输入数据转换；Evaluation Backend，支持多种评测框架；Performance Evaluator，衡量模型性能；Evaluation Report，生成评测报告；Visualization，提供评测结果的可视化。",
        "reference":"EvalScope的整体架构图中包含以下模块：\n\n1. Model Adapter: 模型适配器，用于将特定模型的输出转换为框架所需的格式，支持API调用的模型和本地运行的模型。\n\n2. Data Adapter: 数据适配器，负责转换和处理输入数据，以便适应不同的评测需求和格式。\n\n3. Evaluation Backend: 包括多个评测后端模块：\n   - Native: EvalScope自身的默认评测框架，支持多种评测模式。\n   - OpenCompass: 支持OpenCompass作为评测后端，简化任务提交。\n   - VLMEvalKit: 支持多模态评测任务。\n   - RAGEval: 支持RAG评测，使用MTEB\/CMTEB和RAGAS进行评测。\n   - ThirdParty: 其他第三方评测任务，如ToolBench。\n\n4. Performance Evaluator: 模型性能评测，负责衡量模型推理服务性能，包括性能评测、压力测试、报告生成和可视化。\n\n5. Evaluation Report: 最终生成的评测报告，总结模型的性能表现，用于决策和模型优化。\n\n6. Visualization: 可视化结果，帮助用户理解评测结果，便于分析和比较不同模型的表现。"
    }
]
```

</details>

### 配置评测任务

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
            "language": "chinese"
        },
    },
}
```
基本参数与自动评测任务配置一致，不同点在于：
- `critic_llm`：配置评测模型，参数同`generator_llm`
- `testset_file` ：指定评测数据集文件路径，默认为 `outputs/testset_with_answer.json`
- `metrics`: 指定评测指标，参考[metrics 列表](https://docs.ragas.io/en/latest/concepts/metrics/index.html)


### 执行评测

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

## 多模态图文RAG评测

本框架扩展了RAG评测，支持多模态图文评测，支持图文上下文。

### 数据集准备

```json
[
    {
        "user_input": "图片中的汽车品牌是什么？",
        "retrieved_contexts": [
            "custom_eval/multimodal/images/tesla.jpg"
        ],
        "response": "特斯拉是一个汽车品牌。",
        "reference": "图片中的汽车品牌是特斯拉。"
    },
    {
        "user_input": "那特斯拉Model X呢？",
        "retrieved_contexts": [
            "custom_eval/multimodal/images/tesla.jpg"
        ],
        "response": "猫很可爱。",
        "reference": "特斯拉Model X是一款由特斯拉制造的电动SUV。"
    }
]
```

需要的字段包括：
- user_input：用户输入
- response：模型生成的答案
- retrieved_contexts：检索得到的上下文列表，可以为文本或图片路径（本地路径或网络路径）
- reference：标准答案

### 配置评测任务
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
基本参数与RAG评测任务配置一致，不同点在于：
- `critic_llm`：配置评测模型，参数同`generator_llm`，模型需支持多模态交错图文输入。
- `metrics`: 指定评测指标，目前支持`MultiModalFaithfulness`和`MultiModalRelevance`，用于多模态图文评测。其余不涉及多模态数据的评测指标，如`AnswerCorrectness`等，请参考[metrics 列表](https://docs.ragas.io/en/latest/concepts/metrics/index.html)。
- `embeddings`为可选参数，可以不指定。

### 执行评测
```python
from evalscope.run import run_task

run_task(task_cfg=multi_modal_task_cfg)
```

输出结果:

```json
[
    {
        "user_input":"图片中的汽车品牌是什么？",
        "retrieved_contexts":[
            "custom_eval\/multimodal\/images\/tesla.jpg"
        ],
        "response":"特斯拉是一个汽车品牌。",
        "reference":"图片中的汽车品牌是特斯拉。",
        "faithful_rate":true,
        "relevance_rate":true
    },
    {
        "user_input":"那特斯拉Model X呢？",
        "retrieved_contexts":[
            "custom_eval\/multimodal\/images\/tesla.jpg"
        ],
        "response":"猫很可爱。",
        "reference":"特斯拉Model X是一款由特斯拉制造的电动SUV。",
        "faithful_rate":false,
        "relevance_rate":false
    }
]
```
