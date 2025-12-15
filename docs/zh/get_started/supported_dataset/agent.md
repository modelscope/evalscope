# AGENT评测集

以下是支持的AGENT评测集列表，点击数据集标准名称可跳转详细信息。

| 数据集名称 | 标准名称 | 任务类别 |
|------------|----------|----------|
| `bfcl_v3` | [BFCL-v3](#bfcl-v3) | `Agent`, `FunctionCalling` |
| `bfcl_v4` | [BFCL-v4](#bfcl-v4) | `Agent`, `FunctionCalling` |
| `general_fc` | [General-FunctionCalling](#general-functioncalling) | `Agent`, `Custom`, `FunctionCalling` |
| `tau2_bench` | [τ²-bench](#τ²-bench) | `Agent`, `FunctionCalling`, `Reasoning` |
| `tau_bench` | [τ-bench](#τ-bench) | `Agent`, `FunctionCalling`, `Reasoning` |
| `tool_bench` | [ToolBench-Static](#toolbench-static) | `FunctionCalling`, `Reasoning` |

---

## 数据集详情

### BFCL-v3

[返回目录](#agent评测集)
- **数据集名称**: `bfcl_v3`
- **数据集ID**: [AI-ModelScope/bfcl_v3](https://modelscope.cn/datasets/AI-ModelScope/bfcl_v3/summary)
- **数据集介绍**:
  > Berkeley Function Calling Leaderboard (BFCL) 是首个专注于评估大语言模型（LLM）调用函数能力的**全面且可执行的函数调用评测**。与以往评测不同，BFCL 考虑了多种函数调用形式、多样化场景以及可执行性。评测前需安装 `pip install bfcl-eval==2025.10.27.1`。[使用示例](https://evalscope.readthedocs.io/zh-cn/latest/third_party/bfcl_v3.html)
- **任务类别**: `Agent`, `FunctionCalling`
- **评估指标**: `acc`
- **聚合方法**: `mean`
- **是否需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **评测数据集划分**: `train`
- **数据集子集**: `irrelevance`, `java`, `javascript`, `live_irrelevance`, `live_multiple`, `live_parallel_multiple`, `live_parallel`, `live_relevance`, `live_simple`, `multi_turn_base`, `multi_turn_long_context`, `multi_turn_miss_func`, `multi_turn_miss_param`, `multiple`, `parallel_multiple`, `parallel`, `simple`

- **额外参数**: 
```json
{
    "underscore_to_dot": {
        "type": "bool",
        "description": "Convert underscores to dots in function names for evaluation.",
        "value": true
    },
    "is_fc_model": {
        "type": "bool",
        "description": "Indicates the evaluated model natively supports function calling.",
        "value": true
    }
}
```

---

### BFCL-v4

[返回目录](#agent评测集)
- **数据集名称**: `bfcl_v4`
- **数据集ID**: [berkeley-function-call-leaderboard](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
- **数据集介绍**:
  > 函数调用是智能体（Agents）的基础构建模块，伯克利函数调用排行榜（BFCL）V4 提供了针对大语言模型（LLM）的综合性智能体评估。BFCL V4 智能体评估包含网页搜索、记忆读写和格式敏感性。结合跨语言函数调用能力，这些构成了当前驱动智能体 LLM 发展的核心基础，涵盖深度研究、编程代理和法律代理等极具挑战性的前沿领域。评估前需运行 `pip install bfcl-eval==2025.10.27.1`。[使用示例](https://evalscope.readthedocs.io/zh-cn/latest/third_party/bfcl_v4.html)
- **任务类别**: `Agent`, `FunctionCalling`
- **评估指标**: `acc`
- **聚合方法**: `mean`
- **是否需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **评测数据集划分**: `train`
- **数据集子集**: `irrelevance`, `live_irrelevance`, `live_multiple`, `live_parallel_multiple`, `live_parallel`, `live_relevance`, `live_simple`, `memory_kv`, `memory_rec_sum`, `memory_vector`, `multi_turn_base`, `multi_turn_long_context`, `multi_turn_miss_func`, `multi_turn_miss_param`, `multiple`, `parallel_multiple`, `parallel`, `simple_java`, `simple_javascript`, `simple_python`, `web_search_base`, `web_search_no_snippet`

- **额外参数**: 
```json
{
    "underscore_to_dot": {
        "type": "bool",
        "description": "Convert underscores to dots in function names for evaluation.",
        "value": true
    },
    "is_fc_model": {
        "type": "bool",
        "description": "Indicates the evaluated model natively supports function calling.",
        "value": true
    },
    "SERPAPI_API_KEY": {
        "type": "str | null",
        "description": "SerpAPI key enabling web-search capability in BFCL V4. Null disables web search.",
        "value": null
    }
}
```

---

### General-FunctionCalling

[返回目录](#agent评测集)
- **数据集名称**: `general_fc`
- **数据集ID**: [evalscope/GeneralFunctionCall-Test](https://modelscope.cn/datasets/evalscope/GeneralFunctionCall-Test/summary)
- **数据集介绍**:
  > 一个用于自定义评测的通用函数调用数据集。有关如何使用此基准的详细说明，请参阅[用户指南](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/custom_dataset/llm.html#fc)。
- **任务类别**: `Agent`, `Custom`, `FunctionCalling`
- **评估指标**: `count_finish_reason_tool_call`, `count_successful_tool_call`, `schema_accuracy`, `tool_call_f1`
- **聚合方法**: `f1`
- **是否需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **评测数据集划分**: `test`
- **数据集子集**: `default`


---

### τ²-bench

[返回目录](#agent评测集)
- **数据集名称**: `tau2_bench`
- **数据集ID**: [evalscope/tau2-bench-data](https://modelscope.cn/datasets/evalscope/tau2-bench-data/summary)
- **数据集介绍**:
  > τ²-bench（Tau Squared Bench）是原始 τ-bench（Tau Bench）的扩展和增强版本，旨在评估通过特定领域 API 工具和规则与用户交互的对话式 AI 代理。请在评估前使用 `pip install git+https://github.com/sierra-research/tau2-bench@v0.2.0` 安装并设置用户模型。[使用示例](https://evalscope.readthedocs.io/zh-cn/latest/third_party/tau2_bench.html)
- **任务类别**: `Agent`, `FunctionCalling`, `Reasoning`
- **评估指标**: 
- **聚合方法**: `mean_and_pass_hat_k`
- **是否需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **评测数据集划分**: `test`
- **数据集子集**: `airline`, `retail`, `telecom`

- **额外参数**: 
```json
{
    "user_model": {
        "type": "str",
        "description": "Model used to simulate the user in the environment.",
        "value": "qwen-plus"
    },
    "api_key": {
        "type": "str",
        "description": "API key for the user model backend.",
        "value": "EMPTY"
    },
    "api_base": {
        "type": "str",
        "description": "Base URL for the user model API requests.",
        "value": "https://dashscope.aliyuncs.com/compatible-mode/v1"
    },
    "generation_config": {
        "type": "dict",
        "description": "Default generation config for user model simulation.",
        "value": {
            "temperature": 0.0
        }
    }
}
```

---

### τ-bench

[返回目录](#agent评测集)
- **数据集名称**: `tau_bench`
- **数据集ID**: [tau-bench](https://github.com/sierra-research/tau-bench)
- **数据集介绍**:
  > 一个模拟用户（由语言模型模拟）与具备特定领域API工具和策略指南的语言代理之间动态对话的基准测试。评估前请先通过 `pip install git+https://github.com/sierra-research/tau-bench` 安装并设置用户模型。[使用示例](https://evalscope.readthedocs.io/zh-cn/latest/third_party/tau_bench.html)
- **任务类别**: `Agent`, `FunctionCalling`, `Reasoning`
- **评估指标**: 
- **聚合方法**: `mean_and_pass_hat_k`
- **是否需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **评测数据集划分**: `test`
- **数据集子集**: `airline`, `retail`

- **额外参数**: 
```json
{
    "user_model": {
        "type": "str",
        "description": "Model used to simulate the user in the environment.",
        "value": "qwen-plus"
    },
    "api_key": {
        "type": "str",
        "description": "API key for the user model backend.",
        "value": "EMPTY"
    },
    "api_base": {
        "type": "str",
        "description": "Base URL for the user model API requests.",
        "value": "https://dashscope.aliyuncs.com/compatible-mode/v1"
    },
    "generation_config": {
        "type": "dict",
        "description": "Default generation config for user model simulation.",
        "value": {
            "temperature": 0.0
        }
    }
}
```

---

### ToolBench-Static

[返回目录](#agent评测集)
- **数据集名称**: `tool_bench`
- **数据集ID**: [AI-ModelScope/ToolBench-Static](https://modelscope.cn/datasets/AI-ModelScope/ToolBench-Static/summary)
- **数据集介绍**:
  > ToolBench 是一个用于评估 AI 模型工具使用能力的基准，包含多个子集（如领域内和领域外），每个子集均提供需逐步推理才能得出正确答案的问题。[使用示例](https://evalscope.readthedocs.io/zh-cn/latest/third_party/toolbench.html)
- **任务类别**: `FunctionCalling`, `Reasoning`
- **评估指标**: `Act.EM`, `F1`, `HalluRate`, `Plan.EM`, `Rouge-L`
- **聚合方法**: `mean`
- **是否需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **评测数据集划分**: `test`
- **数据集子集**: `in_domain`, `out_of_domain`

