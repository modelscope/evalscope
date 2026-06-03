<p align="center">
    <br>
    <img src="docs/en/_static/images/evalscope_logo.png"/>
    <br>
<p>

<p align="center">
  中文 &nbsp ｜ &nbsp <a href="README.md">English</a> &nbsp
</p>

<p align="center">
<img src="https://img.shields.io/badge/python-%E2%89%A53.10-5be.svg">
<a href="https://badge.fury.io/py/evalscope"><img src="https://badge.fury.io/py/evalscope.svg" alt="PyPI version" height="18"></a>
<a href="https://pypi.org/project/evalscope"><img alt="PyPI - Downloads" src="https://static.pepy.tech/badge/evalscope"></a>
<a href="https://github.com/modelscope/evalscope/pulls"><img src="https://img.shields.io/badge/PR-welcome-55EB99.svg"></a>
<a href='https://evalscope.readthedocs.io/zh-cn/latest/?badge=latest'><img src='https://readthedocs.org/projects/evalscope/badge/?version=latest' alt='Documentation Status' /></a>
<p>

<p align="center">
<a href="https://evalscope.readthedocs.io/zh-cn/latest/"> 📖  中文文档</a> &nbsp ｜ &nbsp <a href="https://evalscope.readthedocs.io/en/latest/"> 📖  English Documents</a>
<p>


> ⭐ 如果你喜欢这个项目，请点击右上角的 "Star" 按钮支持我们。你的支持是我们前进的动力！

## 📝 简介

EvalScope 是由[魔搭社区](https://modelscope.cn/)打造的一站式大模型评测框架。一行命令即可开始评测，支持模型能力评估、推理性能压测和结果可视化。

```bash
pip install evalscope
evalscope eval --model your-model-name --api-url $OPENAI_API_BASE_URL --api-key $OPENAI_API_KEY --eval-type openai_api --datasets gsm8k --limit 5
```

## ✨ 主要特性

- **📚 全面的评测基准**: 内置 MMLU, C-Eval, GSM8K 等多个业界公认的评测基准。
- **🧩 多模态与多领域支持**: 支持大语言模型 (LLM)、多模态 (VLM)、Embedding、Reranker、AIGC 等多种模型的评测。
- **🚀 多后端集成**: 无缝集成 OpenCompass, VLMEvalKit, RAGEval 等多种评测后端，满足不同评测需求。
- **🤖 Agent 评测模式**: 在受控的多轮 AgentLoop 中驱动 GSM8K、AIME、SWE-bench Agentic 等基准；支持可插拔的策略、工具与 Docker 沙箱，每条样本完整记录 Agent Trace 并可在仪表盘中按步骤回放。
- **⚡ 推理性能测试**: 提供强大的模型服务压力测试工具，支持 TTFT, TPOT 等多项性能指标。
- **📊 交互式报告**: 提供 WebUI 可视化界面，支持多维度模型对比、报告概览和详情查阅。
- **⚔️ 竞技场模式**: 支持多模型对战 (Pairwise Battle)，直观地对模型进行排名和评估。
- **🔧 高度可扩展**: 开发者可以轻松添加自定义数据集、模型和评测指标。

## 📊 可视化效果展示

EvalScope 提供交互式 Web Dashboard，支持多维度模型对比和深入分析。

<table>
  <tr>
    <td style="text-align: center;">
      <img src="https://sail-moe.oss-cn-hangzhou.aliyuncs.com/yunlin/images/evalscope/dashboard/dashboard_overview.png" alt="Dashboard" style="width: 100%;" />
      <p>仪表盘概览</p>
    </td>
    <td style="text-align: center;">
      <img src="https://sail-moe.oss-cn-hangzhou.aliyuncs.com/yunlin/images/evalscope/dashboard/compare_score_tab.png" alt="Model Compare" style="width: 90%;" />
      <p>多模型对比</p>
    </td>
  </tr>
  <tr>
    <td style="text-align: center;">
      <img src="https://sail-moe.oss-cn-hangzhou.aliyuncs.com/yunlin/images/evalscope/dashboard/report_overview_tab.png" alt="Report Overview" style="width: 100%;" />
      <p>报告概览</p>
    </td>
    <td style="text-align: center;">
      <img src="https://sail-moe.oss-cn-hangzhou.aliyuncs.com/yunlin/images/evalscope/dashboard/report_predictions_tab.png" alt="Report Predictions" style="width: 80%;" />
      <p>预测标签</p>
    </td>
  </tr>
</table>

详情请参考 [📖 可视化评测结果](https://evalscope.readthedocs.io/zh-cn/latest/get_started/visualization.html)。

## 🎉 内容更新

- 🔥 **[2026.06.02]** **RAG 评测**模块重构：升级 MTEB 2.x 与 RAGAS 0.4.x，配置统一为 Pydantic 模型。参考[RAGEval 使用文档](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/rageval_backend/index.html)。
- 🔥 **[2026.05.27]** 新增 **Trie agentic 轨迹回放**压测：三个数据集插件（`trie_agentic_coding` / `trie_code_qa` / `trie_office_work`）可回放真实多轮 Agent 轨迹，支持逐轮 token 上限和工具调用延迟模拟；同时新增 `--duration` 墙钟时间预算（适用于所有压测模式）和 `Turn` 数据类。
- 🔥 **[2026.05.27]** 新增 **Vendor Verifier 基准**（`k2_verifier`、`kimi_verifier`、`minimax_verifier`），用于验证第三方 API 部署是否忠实复现官方模型行为，共享 `VendorVerifierAdapter` 基类。
- 🔥 **[2026.05.26]** 新增 [GAIA](https://evalscope.readthedocs.io/zh-cn/latest/third_party/gaia.html) agent 基准（Docker sandbox 内多轮 ReAct + `bash`，复用官方规则评分器）和通用 [MCP 服务器](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/agent/native.html#mcp-工具接入)接入 —— 任何基于 `NativeAgentConfig` 的 benchmark 都可直接挂载 stdio / HTTP / SSE 的 MCP server（`fetch`、网页搜索、GitHub 等），无需 benchmark 端改动。
- 🔥 **[2026.05.22]** 新增 **外部 Agent Bridge** 模式：可直接评测 Anthropic [Claude Code](https://github.com/anthropics/claude-code)、OpenAI [Codex](https://github.com/openai/codex) 等成品 Agent CLI。Bridge 透明转发 CLI 的 LLM 请求（Anthropic Messages / OpenAI Chat / OpenAI Responses，含 SSE 流式响应）到评测模型，同时把完整交互轨迹录制为 `agent_trace`；通过 `@register_runner` 可接入任意第三方 CLI。详见[外部 Agent Bridge 指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/agent/bridge.html)。
- 🔥 **[2026.05.19]** 新增对 [SWE-bench_Pro](https://evalscope.readthedocs.io/zh-cn/latest/third_party/swe_bench_pro.html) 与 [τ³-bench](https://evalscope.readthedocs.io/zh-cn/latest/third_party/tau3_bench.html) 的支持：SWE-bench_Pro 是 Scale AI 推出的更具挑战性的多语言、长周期软件工程基准，相比原始 SWE-bench 数据污染更少、覆盖语言更广，**推荐替代原始 SWE-bench 使用**，每个实例的 Docker 镜像直接从 DockerHub 拉取，无需本地构建；τ³-bench 是 tau-bench 系列的 v1.0.0 版本，在 τ²-bench 基础上新增 `banking_knowledge` 知识检索领域（RAG）、修复 75+ 项任务，并提供可插拔的检索流水线（BM25 / 稠密嵌入 / 重排序器 / 沙箱 shell）。
- 🔥 **[2026.05.15]** 新增 **Agent 评测模式**：所有基于 `DefaultDataAdapter` 的基准（GSM8K、AIME、IFEval 等）现在均可通过多轮 AgentLoop 驱动，支持可插拔策略（`function_calling` / `react` / `swe_bench_*`）、工具（`bash` / `python_exec` / `submit`）以及 `local` / `docker` 运行环境，每条样本的 `agent_trace` 会随评测结果落盘，并在仪表盘的预测视图中按步骤回放。详见[Agent 评测指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/agent/native.html)。
- 🔥 **[2026.05.08]** 与 [LightSeek](https://lightseek.org/) 联合推出 [TokenSpeed](https://lightseek.org/blog/lightseek-tokenspeed.html)——面向 Agentic 工作负载的极速 LLM 推理引擎。EvalScope 提供 SWE-smith 压测流水线，基于真实 Coding Agent 轨迹衡量单 GPU 吞吐（TPM）与单用户延迟（TPS），作为 TokenSpeed 性能评测的官方基准工具。参考 [SWE-smith 使用指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/multi_turn.html#swe-smith) 快速上手。
- 🔥 **[2026.05.07]** 全新 Web 界面升级：使用 React + Vite 重构可视化平台，替换原有 Gradio 界面，提供更流畅的交互体验。
- 🔥 **[2026.04.23]** 支持在评测任务中记录性能（perf）指标，可在单次评测运行中同时追踪模型准确率与 TTFT、TPOT、吞吐量等推理效率指标。
- 🔥 **[2026.04.17]** 支持多轮对话性能压测，可对具备多轮上下文的对话模型服务进行负载测试，参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/examples.html)。
- 🔥 **[2026.04.10]** 新增支持 [TIR-Bench](https://arxiv.org/abs/2511.01833)（Thinking-with-Images Reasoning Benchmark），一个面向视觉语言模型的多模态推理基准。
- 🔥 **[2026.03.24]** 支持 Agent Skill，任何支持 Skill/Tool 调用的 Agent 模型均可通过自然语言直接驱动 EvalScope 完成模型评测、性能压测和结果可视化。
- 🔥 **[2026.03.09]** 支持评测进度追踪和自动生成HTML格式可视化报告。

<details><summary>更多历史更新</summary>

- 🔥 **[2026.03.02]** 支持Anthropic Claude API评测，通过`--eval-type anthropic_api`指定使用Anthropic API服务进行评测。
- 🔥 **[2026.02.03]** 全面更新数据集说明文档，添加数据信息统计、数据样例、使用方法等部分，参考[支持的数据集](https://evalscope.readthedocs.io/zh-cn/latest/get_started/supported_dataset/llm.html)
- 🔥 **[2026.01.13]** 支持Embedding和Rerank模型服务压测，参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/examples.html#embedding)
- 🔥 **[2025.12.26]** 支持Terminal-Bench-2.0，用于评估 AI Agent在 89 个真实世界的多步骤终端任务上的表现，参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/third_party/terminal_bench.html)
- 🔥 **[2025.12.18]** 支持SLA自动调优模型API服务，自动测试模型服务在特定时延、TTFT、吞吐量下的最高并发，参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/sla_auto_tune.html)
- 🔥 **[2025.12.16]** 支持Fleurs、LibriSpeech等音频评测基准；支持MultiplE、MBPP等多语言代码评测基准。
- 🔥 **[2025.12.02]** 支持自定义多模态VQA评测，参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/custom_dataset/vlm.html) ；支持模型服务压测在 ClearML 上可视化，参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/examples.html#clearml)。
- 🔥 **[2025.11.26]** 新增支持 OpenAI-MRCR、GSM8K-V、MGSM、MicroVQA、IFBench、SciCode 评测基准。
- 🔥 **[2025.11.18]** 支持自定义 Function-Call（工具调用）数据集，来测试模型能否适时并正确调用工具，参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/custom_dataset/llm.html#fc)
- 🔥 **[2025.11.14]** 新增支持SWE-bench_Verified, SWE-bench_Lite, SWE-bench_Verified_mini 代码评测基准，参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/third_party/swe_bench.html)。
- 🔥 **[2025.11.12]** 新增`pass@k`、`vote@k`、`pass^k`等指标聚合方法；新增支持A_OKVQA, CMMU, ScienceQ, V*Bench等多模态评测基准。
- 🔥 **[2025.11.07]** 新增支持τ²-bench，是 τ-bench 的扩展与增强版本，包含一系列代码修复，并新增了电信（telecom）领域的故障排查场景，参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/third_party/tau2_bench.html)。
- 🔥 **[2025.10.30]** 新增支持BFCL-v4，支持agent的网络搜索和长期记忆能力的评测，参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/third_party/bfcl_v4.html)。
- 🔥 **[2025.10.27]** 新增支持LogiQA, HaluEval, MathQA, MRI-QA, PIQA, QASC, CommonsenseQA等评测基准。感谢 @[penguinwang96825](https://github.com/penguinwang96825) 提供代码实现。
- 🔥 **[2025.10.26]** 新增支持Conll-2003, CrossNER, Copious, GeniaNER, HarveyNER, MIT-Movie-Trivia, MIT-Restaurant, OntoNotes5, WNUT2017 等命名实体识别评测基准。感谢 @[penguinwang96825](https://github.com/penguinwang96825) 提供代码实现。
- 🔥 **[2025.10.21]** 优化代码评测中的沙箱环境使用，支持在本地和远程两种模式下运行，具体参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/sandbox.html)。
- 🔥 **[2025.10.20]** 新增支持PolyMath, SimpleVQA, MathVerse, MathVision, AA-LCR 等评测基准；优化evalscope perf表现，对齐vLLM Bench，具体参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/vs_vllm_bench.html)。
- 🔥 **[2025.10.14]** 新增支持OCRBench, OCRBench-v2, DocVQA, InfoVQA, ChartQA, BLINK 等图文多模态评测基准。
- 🔥 **[2025.09.22]** 代码评测基准(HumanEval, LiveCodeBench)支持在沙箱环境中运行，要使用该功能需先安装[ms-enclave](https://github.com/modelscope/ms-enclave)。
- 🔥 **[2025.09.19]** 新增支持RealWorldQA、AI2D、MMStar、MMBench、OmniBench等图文多模态评测基准，和Multi-IF、HealthBench、AMC等纯文本评测基准。
- 🔥 **[2025.09.05]** 支持视觉-语言多模态大模型的评测任务，例如：MathVista、MMMU，更多支持数据集请[参考](https://evalscope.readthedocs.io/zh-cn/latest/get_started/supported_dataset/vlm.html)。
- 🔥 **[2025.09.04]** 支持图像编辑任务评测，支持[GEdit-Bench](https://modelscope.cn/datasets/stepfun-ai/GEdit-Bench) 评测基准，使用方法[参考](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/aigc/image_edit.html)。
- 🔥 **[2025.08.22]** Version 1.0 重构，不兼容的更新请[参考](https://evalscope.readthedocs.io/zh-cn/latest/get_started/basic_usage.html#v1-0)。
- 🔥 **[2025.07.18]** 模型压测支持随机生成图文数据，用于多模态模型压测，使用方法[参考](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/examples.html#id4)。
- 🔥 **[2025.07.16]** 支持[τ-bench](https://github.com/sierra-research/tau-bench)，用于评估 AI Agent在动态用户和工具交互的实际环境中的性能和可靠性，使用方法[参考](https://evalscope.readthedocs.io/zh-cn/latest/get_started/supported_dataset/llm.html#bench)。
- 🔥 **[2025.07.14]** 支持"人类最后的考试"([Humanity's-Last-Exam](https://modelscope.cn/datasets/cais/hle))，这一高难度评测基准，使用方法[参考](https://evalscope.readthedocs.io/zh-cn/latest/get_started/supported_dataset/llm.html#humanity-s-last-exam)。
- 🔥 **[2025.07.03]** 重构了竞技场模式，支持自定义模型对战，输出模型排行榜，以及对战结果可视化，使用[参考](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/arena.html)。
- 🔥 **[2025.06.28]** 优化自定义数据集评测，支持无参考答案评测；优化LLM裁判使用，预置"无参考答案直接打分" 和 "判断答案是否与参考答案一致"两种模式，使用[参考](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/custom_dataset/llm.html#qa)
- 🔥 **[2025.06.19]** 新增支持[BFCL-v3](https://modelscope.cn/datasets/AI-ModelScope/bfcl_v3)评测基准，用于评测模型在多种场景下的函数调用能力，使用[参考](https://evalscope.readthedocs.io/zh-cn/latest/third_party/bfcl_v3.html)。
- 🔥 **[2025.06.02]** 新增支持大海捞针测试（Needle-in-a-Haystack），指定`needle_haystack`即可进行测试，并在`outputs/reports`文件夹下生成对应的heatmap，直观展现模型性能，使用[参考](https://evalscope.readthedocs.io/zh-cn/latest/third_party/needle_haystack.html)。
- 🔥 **[2025.05.29]** 新增支持[DocMath](https://modelscope.cn/datasets/yale-nlp/DocMath-Eval/summary)和[FRAMES](https://modelscope.cn/datasets/iic/frames/summary)两个长文档评测基准，使用注意事项请查看[文档](https://evalscope.readthedocs.io/zh-cn/latest/get_started/supported_dataset/index.html)
- 🔥 **[2025.05.16]** 模型服务性能压测支持设置多种并发，并输出性能压测报告，[参考示例](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/quick_start.html#id3)。
- 🔥 **[2025.05.13]** 新增支持[ToolBench-Static](https://modelscope.cn/datasets/AI-ModelScope/ToolBench-Static)数据集，评测模型的工具调用能力，参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/third_party/toolbench.html)；支持[DROP](https://modelscope.cn/datasets/AI-ModelScope/DROP/dataPeview)和[Winogrande](https://modelscope.cn/datasets/AI-ModelScope/winogrande_val)评测基准，评测模型的推理能力。
- 🔥 **[2025.04.29]** 新增Qwen3评测最佳实践，[欢迎阅读📖](https://evalscope.readthedocs.io/zh-cn/latest/best_practice/qwen3.html)
- 🔥 **[2025.04.27]** 支持文生图评测：支持MPS、HPSv2.1Score等8个指标，支持EvalMuse、GenAI-Bench等评测基准，参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/aigc/t2i.html)
- 🔥 **[2025.04.10]** 模型服务压测工具支持`/v1/completions`端点（也是vLLM基准测试的默认端点）
- 🔥 **[2025.04.08]** 支持OpenAI API兼容的Embedding模型服务评测，查看[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/rageval_backend/mteb.html#configure-evaluation-parameters)
- 🔥 **[2025.03.27]** 新增支持[AlpacaEval](https://www.modelscope.cn/datasets/AI-ModelScope/alpaca_eval/dataPeview)和[ArenaHard](https://modelscope.cn/datasets/AI-ModelScope/arena-hard-auto-v0.1/summary)评测基准，使用注意事项请查看[文档](https://evalscope.readthedocs.io/zh-cn/latest/get_started/supported_dataset/index.html)
- 🔥 **[2025.03.20]** 模型推理服务压测支持random生成指定范围长度的prompt，参考[使用指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/examples.html#random)
- 🔥 **[2025.03.13]** 新增支持[LiveCodeBench](https://www.modelscope.cn/datasets/evalscope/livecodebench_code_generation_lite_parquet/summary)代码评测基准，指定`live_code_bench`即可使用；支持QwQ-32B 在LiveCodeBench上评测，参考[最佳实践](https://evalscope.readthedocs.io/zh-cn/latest/best_practice/eval_qwq.html)。
- 🔥 **[2025.03.11]** 新增支持[SimpleQA](https://modelscope.cn/datasets/AI-ModelScope/SimpleQA/summary)和[Chinese SimpleQA](https://modelscope.cn/datasets/AI-ModelScope/Chinese-SimpleQA/summary)评测基准，用与评测模型的事实正确性，指定`simple_qa`和`chinese_simpleqa`使用。同时支持指定裁判模型，参考[相关参数说明](https://evalscope.readthedocs.io/zh-cn/latest/get_started/parameters.html)。
- 🔥 **[2025.03.07]** 新增QwQ-32B模型评测最佳实践，评测了模型的推理能力以及推理效率，参考[📖QwQ-32B模型评测最佳实践](https://evalscope.readthedocs.io/zh-cn/latest/best_practice/eval_qwq.html)。
- 🔥 **[2025.03.04]** 新增支持[SuperGPQA](https://modelscope.cn/datasets/m-a-p/SuperGPQA/summary)数据集，其覆盖 13 个门类、72 个一级学科和 285 个二级学科，共 26,529 个问题，指定`super_gpqa`即可使用。
- 🔥 **[2025.03.03]** 新增支持评测模型的智商和情商，参考[📖智商和情商评测最佳实践](https://evalscope.readthedocs.io/zh-cn/latest/best_practice/iquiz.html)，来测测你家的AI有多聪明？
- 🔥 **[2025.02.27]** 新增支持评测推理模型的思考效率，参考[📖思考效率评测最佳实践](https://evalscope.readthedocs.io/zh-cn/latest/best_practice/think_eval.html)，该实现参考了[Overthinking](https://doi.org/10.48550/arXiv.2412.21187) 和 [Underthinking](https://doi.org/10.48550/arXiv.2501.18585)两篇工作。
- 🔥 **[2025.02.25]** 新增支持[MuSR](https://modelscope.cn/datasets/AI-ModelScope/MuSR)和[ProcessBench](https://www.modelscope.cn/datasets/Qwen/ProcessBench/summary)两个模型推理相关评测基准，datasets分别指定`musr`和`process_bench`即可使用。
- 🔥 **[2025.02.18]** 支持AIME25数据集，包含15道题目（Grok3 在该数据集上得分为93分）
- 🔥 **[2025.02.13]** 支持DeepSeek蒸馏模型评测，包括AIME24, MATH-500, GPQA-Diamond数据集，参考[最佳实践](https://evalscope.readthedocs.io/zh-cn/latest/best_practice/deepseek_r1_distill.html)；支持指定`eval_batch_size`参数，加速模型评测
- 🔥 **[2025.01.20]** 支持可视化评测结果，包括单模型评测结果和多模型评测结果对比，参考[📖可视化评测结果](https://evalscope.readthedocs.io/zh-cn/latest/get_started/visualization.html)
- 🔥 **[2025.01.07]** Native backend: 支持模型API评测，参考[📖模型API评测指南](https://evalscope.readthedocs.io/zh-cn/latest/get_started/basic_usage.html#api)；新增支持`ifeval`评测基准。
- 🔥🔥 **[2024.12.31]** 支持基准评测添加，参考[📖基准评测添加指南](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/add_benchmark.html)
- 🔥 **[2024.12.13]** 模型评测优化，不再需要传递`--template-type`参数；支持`evalscope eval --args`启动评测。
- 🔥 **[2024.11.26]** 模型推理压测工具重构完成：支持本地启动推理服务、支持Speed Benchmark。
- 🔥 **[2024.10.31]** 多模态RAG评测最佳实践发布。
- 🔥 **[2024.10.23]** 支持多模态RAG评测。
- 🔥 **[2024.10.8]** 支持RAG评测。
- 🔥 **[2024.09.18]** 文档添加博客模块。
- 🔥 **[2024.09.12]** 支持 LongWriter 评测。
- 🔥 **[2024.08.30]** 支持自定义数据集评测。
- 🔥 **[2024.08.20]** 更新了官方文档。
- 🔥 **[2024.08.09]** 简化安装方式，优化多模态模型评测体验。
- 🔥 **[2024.07.31]** 重要修改：`llmuses`包名修改为`evalscope`，请同步修改您的代码。
- 🔥 **[2024.07.26]** 支持**VLMEvalKit**作为第三方评测框架。
- 🔥 **[2024.06.29]** 支持**OpenCompass**作为第三方评测框架。
- 🔥 **[2024.06.13]** EvalScope与SWIFT微调框架集成；接入Agent评测集ToolBench。

</details>

## 🚀 快速开始

### 安装

```shell
pip install evalscope
```

> 详细安装说明（源码安装、额外依赖等）请参考 [📖 安装指南](https://evalscope.readthedocs.io/zh-cn/latest/get_started/installation.html)。

### 方式1. 评测在线模型 API（推荐入门方式，无需 GPU）

支持任意 OpenAI API 兼容的模型服务，只需配置 `$OPENAI_API_BASE_URL` 与 `$OPENAI_API_KEY` 即可开始评测：

```bash
evalscope eval \
 --model your-model-name \
 --api-url $OPENAI_API_BASE_URL \
 --api-key $OPENAI_API_KEY \
 --eval-type openai_api \
 --datasets gsm8k arc \
 --limit 5
```

### 方式2. 评测本地模型

使用本地模型（自动从 ModelScope 下载）：

```bash
evalscope eval \
 --model Qwen/Qwen2.5-0.5B-Instruct \
 --datasets gsm8k arc \
 --limit 5
```

### 方式3. 使用 Python 代码

```python
from evalscope import run_task, TaskConfig

task_cfg = TaskConfig(
    model='your-model-name',
    api_url='https://your-openai-compatible-endpoint/v1',
    api_key='your_api_key',
    eval_type='openai_api',
    datasets=['gsm8k', 'arc'],
    limit=5
)

run_task(task_cfg)
```

<details><summary><b>💡 提示：</b> <code>run_task</code> 还支持字典、YAML 或 JSON 文件作为配置。</summary>

**使用 Python 字典**

```python
from evalscope.run import run_task

task_cfg = {
    'model': 'Qwen/Qwen2.5-0.5B-Instruct',
    'datasets': ['gsm8k', 'arc'],
    'limit': 5
}
run_task(task_cfg=task_cfg)
```

**使用 YAML 文件** (`config.yaml`)
```yaml
model: Qwen/Qwen2.5-0.5B-Instruct
datasets:
  - gsm8k
  - arc
limit: 5
```
```python
from evalscope.run import run_task

run_task(task_cfg="config.yaml")
```
</details>

### 输出结果
评测完成后，您将在终端看到如下格式的报告：
```text
+-----------------------+----------------+-----------------+-----------------+---------------+-------+---------+
| Model Name            | Dataset Name   | Metric Name     | Category Name   | Subset Name   |   Num |   Score |
+=======================+================+=================+=================+===============+=======+=========+
| Qwen2.5-0.5B-Instruct | gsm8k          | AverageAccuracy | default         | main          |     5 |     0.4 |
+-----------------------+----------------+-----------------+-----------------+---------------+-------+---------+
| Qwen2.5-0.5B-Instruct | ai2_arc        | AverageAccuracy | default         | ARC-Easy      |     5 |     0.8 |
+-----------------------+----------------+-----------------+-----------------+---------------+-------+---------+
| Qwen2.5-0.5B-Instruct | ai2_arc        | AverageAccuracy | default         | ARC-Challenge |     5 |     0.4 |
+-----------------------+----------------+-----------------+-----------------+---------------+-------+---------+
```

**启动可视化面板**：
```bash
pip install 'evalscope[service]'
evalscope service
```
访问 `http://127.0.0.1:9000` 即可打开可视化界面。

## 📈 进阶用法

### 自定义评测参数

您可以通过命令行参数精细化控制模型加载、推理和数据集配置。

```shell
evalscope eval \
 --model Qwen/Qwen3-0.6B \
 --model-args '{"revision": "master", "precision": "torch.float16", "device_map": "auto"}' \
 --generation-config '{"do_sample":true,"temperature":0.6,"max_tokens":512}' \
 --dataset-args '{"gsm8k": {"few_shot_num": 0, "few_shot_random": false}}' \
 --datasets gsm8k \
 --limit 10
```

- `--model-args`: 模型加载参数，如 `revision`, `precision` 等。
- `--generation-config`: 模型生成参数，如 `temperature`, `max_tokens` 等。
- `--dataset-args`: 数据集配置参数，如 `few_shot_num` 等。

详情请参考 [📖 全部参数说明](https://evalscope.readthedocs.io/zh-cn/latest/get_started/parameters.html)。

### ⚔️ 竞技场模式 (Arena)

竞技场模式通过模型间的两两对战（Pairwise Battle）来评估模型性能，并给出胜率和排名，非常适合多模型横向对比。

```text
# 评测结果示例
Model           WinRate (%)  CI (%)
------------  -------------  ---------------
qwen2.5-72b            69.3  (-13.3 / +12.2)
qwen2.5-7b             50    (+0.0 / +0.0)
qwen2.5-0.5b            4.7  (-2.5 / +4.4)
```
详情请参考 [📖 竞技场模式使用指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/arena.html)。

### 🖊️ 自定义数据集评测

EvalScope 允许您轻松添加和评测自己的数据集。详情请参考 [📖 自定义数据集评测指南](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/custom_dataset/index.html)。

## ⚡ 推理性能评测工具

EvalScope 提供了一个强大的压力测试工具，用于评估大语言模型服务的性能。

- **关键指标**: 支持吞吐量 (Tokens/s)、首字延迟 (TTFT)、Token 生成延迟 (TPOT) 等。
- **结果记录**: 支持将结果记录到 `wandb` 和 `swanlab`。
- **速度基准**: 可生成类似官方报告的速度基准测试结果。

详情请参考 [📖 性能测试使用指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/index.html)。

<p align="center">
    <img src="docs/zh/user_guides/stress_test/images/multi_perf.png" style="width: 80%;">
</p>

## 🧪 其他评测后端
EvalScope 支持通过第三方评测框架（我们称之为"后端"）发起评测任务，以满足多样化的评测需求。

- **Native**: EvalScope 的默认评测框架，功能全面。
- **OpenCompass**: 专注于纯文本评测。 [📖 使用指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/opencompass_backend.html)
- **VLMEvalKit**: 专注于多模态评测。 [📖 使用指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/vlmevalkit_backend.html)
- **RAGEval**: 专注于 RAG 评测，支持 Embedding 和 Reranker 模型。 [📖 使用指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/rageval_backend/index.html)
- **第三方评测工具**: 支持 [ToolBench](https://evalscope.readthedocs.io/zh-cn/latest/third_party/toolbench.html) 等评测任务。

<details><summary>🏛️ 整体架构</summary>

<p align="center">
    <img src="https://sail-moe.oss-cn-hangzhou.aliyuncs.com/yunlin/images/evalscope/doc/EvalScope%E6%9E%B6%E6%9E%84%E5%9B%BE.png" style="width: 70%;">
    <br>EvalScope 整体架构图.
</p>

1.  **输入层**
    - **模型来源**: API模型（OpenAI API）、本地模型（ModelScope）
    - **数据集**: 标准评测基准（MMLU/GSM8k等）、自定义数据（MCQ/QA）

2.  **核心功能**
    - **多后端评估**: 原生后端、OpenCompass、MTEB、VLMEvalKit、RAGAS
    - **性能监控**: 支持多种模型服务 API 和数据格式，追踪 TTFT/TPOP 等指标
    - **工具扩展**: 集成 Tool-Bench, Needle-in-a-Haystack 等

3.  **输出层**
    - **结构化报告**: 支持 JSON, Table, Logs
    - **可视化平台**: 支持 Web Dashboard, Wandb, SwanLab

</details>


## ❤️ 社区与支持

欢迎加入我们的社区，与其他开发者交流并获取帮助。

[Discord Group](https://discord.gg/xc66bMxc4h)              |  微信群 | 钉钉群
:-------------------------:|:-------------------------:|:-------------------------:
<img src="docs/asset/discord_qr.png" width="160" height="160">  |  <img src="https://raw.githubusercontent.com/modelscope/ms-swift/main/asset/wechat.png" width="160" height="160"> | <img src="docs/asset/dingding.png" width="160" height="160">

## 👷‍♂️ 贡献

我们欢迎来自社区的任何贡献！如果您希望添加新的评测基准、模型或功能，请参考我们的 [贡献指南](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/add_benchmark.html)。

感谢所有为 EvalScope 做出贡献的开发者！

<a href="https://github.com/modelscope/evalscope/graphs/contributors" target="_blank">
  <table>
    <tr>
      <th colspan="2">
        <br><img src="https://contrib.rocks/image?repo=modelscope/evalscope"><br><br>
      </th>
    </tr>
  </table>
</a>

## 📚 引用

如果您在研究中使用了 EvalScope，请引用我们的工作：
```bibtex
@misc{evalscope_2024,
    title={{EvalScope}: Evaluation Framework for Large Models},
    author={ModelScope Team},
    year={2024},
    url={https://github.com/modelscope/evalscope}
}
```

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=modelscope/evalscope&type=Date)](https://star-history.com/#modelscope/evalscope&Date)
