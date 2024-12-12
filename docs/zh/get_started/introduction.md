# 简介

EvalScope是[魔搭社区](https://modelscope.cn/)官方推出的模型评测与性能基准测试框架，内置多个常用测试基准和评测指标，如MMLU、CMMLU、C-Eval、GSM8K、ARC、HellaSwag、TruthfulQA、MATH和HumanEval等；支持多种类型的模型评测，包括LLM、多模态LLM、embedding模型和reranker模型。EvalScope还适用于多种评测场景，如端到端RAG评测、竞技场模式和模型推理性能压测等。此外，通过ms-swift训练框架的无缝集成，可一键发起评测，实现了模型训练到评测的全链路支持。

## 整体架构
![EvalScope 架构图](../_static/images/evalscope_framework.png)
*EvalScope 架构图.*

包括以下模块：

1. **Model Adapter**: 模型适配器，用于将特定模型的输出转换为框架所需的格式，支持API调用的模型和本地运行的模型。

2. **Data Adapter**: 数据适配器，负责转换和处理输入数据，以便适应不同的评测需求和格式。

3. **Evaluation Backend**: 
    - **Native**：EvalScope自身的**默认评测框架**，支持多种评测模式，包括单模型评测、竞技场模式、Baseline模型对比模式等。
    - **OpenCompass**：支持[OpenCompass](https://github.com/open-compass/opencompass)作为评测后端，对其进行了高级封装和任务简化，您可以更轻松地提交任务进行评测。
    - **VLMEvalKit**：支持[VLMEvalKit](https://github.com/open-compass/VLMEvalKit)作为评测后端，轻松发起多模态评测任务，支持多种多模态模型和数据集。
    - **ThirdParty**：其他第三方评测任务，如[ToolBench](../third_party/toolbench.md)。
    - **RAGEval**：支持RAG评测，支持使用[MTEB/CMTEB](../user_guides/backend/rageval_backend/mteb.md)进行embedding模型和reranker的独立评测，以及使用[RAGAS](../user_guides/backend/rageval_backend/ragas.md)进行端到端评测。
4. **Performance Evaluator**: 模型性能评测，负责具体衡量模型推理服务性能，包括性能评测、压力测试、性能评测报告生成、可视化。

5. **Evaluation Report**: 最终生成的评测报告，总结模型的性能表现，报告可以用于决策和进一步的模型优化。

6. **Visualization**: 可视化结果，帮助用户更直观地理解评测结果，便于分析和比较不同模型的表现。


## 框架特点
- **基准数据集**：预置了多个常用测试基准，包括：MMLU、CMMLU、C-Eval、GSM8K、ARC、HellaSwag、TruthfulQA、MATH、HumanEval等。
- **评测指标**：实现了多种常用评测指标。
- **模型接入**：统一的模型接入机制，兼容多个系列模型的Generate、Chat接口。
- **自动评测**：包括客观题自动评测和使用专家模型进行的复杂任务评测。
- **评测报告**：自动生成评测报告。
- **竞技场(Arena)模式**：用于模型间的比较以及模型的客观评测，支持多种评测模式，包括：
  - **Single mode**：对单个模型进行评分。
  - **Pairwise-baseline mode**：与基线模型进行对比。
  - **Pairwise (all) mode**：所有模型间的两两对比。
- **可视化工具**：提供直观的评测结果展示。
- **模型性能评测**：提供模型推理服务压测工具和详细统计，详见[模型性能评测文档](../user_guides/stress_test/index.md)。
- **OpenCompass集成**：支持OpenCompass作为评测后端，对其进行了高级封装和任务简化，您可以更轻松地提交任务进行评测。
- **VLMEvalKit集成**：支持VLMEvalKit作为评测后端，轻松发起多模态评测任务，支持多种多模态模型和数据集。
- **全链路支持**：通过与[ms-swift](https://github.com/modelscope/ms-swift)训练框架的无缝集成，实现模型训练、模型部署、模型评测、评测报告查看的一站式开发流程，提升用户的开发效率。

