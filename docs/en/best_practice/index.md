# Best Practices

This section provides practical cases and best practices for evaluating various large language models using EvalScope. We cover evaluation methods for mainstream models (such as Qwen series, DeepSeek series, GPT series, etc.) in different scenarios, including multimodal models, code models, reasoning models, text-to-image models, and more.

Through these practical cases, you can:
- Learn how to select appropriate evaluation datasets and configurations for specific model types
- Master best practices for specialized evaluations such as multimodal, code generation, and reasoning capabilities
- Learn how to integrate with training frameworks like Swift to achieve a training-evaluation closed loop
- Reference complete evaluation workflows and configuration examples to quickly get started with actual evaluation tasks

We recommend selecting the corresponding practical case based on the type of model you want to evaluate.

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
