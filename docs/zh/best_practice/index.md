# 最佳实践

本节提供了使用 EvalScope 评测各类大语言模型的实战案例和最佳实践。我们覆盖了主流模型（如 Qwen 系列、DeepSeek 系列、GPT 系列等）在不同场景下的评测方法,包括多模态模型、代码模型、推理模型、文生图模型等。

通过这些实践案例,您可以:
- 了解如何针对特定模型类型选择合适的评测数据集和配置
- 掌握多模态、代码生成、推理能力等专项评测的最佳实践
- 学习如何与 Swift 等训练框架集成,实现训练-评测闭环
- 参考完整的评测流程和配置示例,快速上手实际评测任务

建议您根据待评测模型的类型,选择对应的实践案例进行参考。

:::{toctree}
:maxdepth: 1

qwen3_omni.md
qwen3_vl.md
qwen3_next.md
gpt_oss.md
qwen3_coder.md
t2i_eval.md
qwen3.md
eval_qwq.md
iquiz.md
think_eval.md
deepseek_r1_distill.md
llm_full_stack.md
swift_integration.md
:::