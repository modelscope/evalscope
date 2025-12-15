# AGENT Benchmarks

Below is the list of supported AGENT benchmarks. Click on a benchmark name to jump to details.

| Benchmark Name | Pretty Name | Task Categories |
|------------|----------|----------|
| `bfcl_v3` | [BFCL-v3](#bfcl-v3) | `Agent`, `FunctionCalling` |
| `bfcl_v4` | [BFCL-v4](#bfcl-v4) | `Agent`, `FunctionCalling` |
| `general_fc` | [General-FunctionCalling](#general-functioncalling) | `Agent`, `Custom`, `FunctionCalling` |
| `tau2_bench` | [τ²-bench](#τ²-bench) | `Agent`, `FunctionCalling`, `Reasoning` |
| `tau_bench` | [τ-bench](#τ-bench) | `Agent`, `FunctionCalling`, `Reasoning` |
| `tool_bench` | [ToolBench-Static](#toolbench-static) | `FunctionCalling`, `Reasoning` |

---

## Benchmark Details

### BFCL-v3

[Back to Top](#agent-benchmarks)
- **Dataset Name**: `bfcl_v3`
- **Dataset ID**: [AI-ModelScope/bfcl_v3](https://modelscope.cn/datasets/AI-ModelScope/bfcl_v3/summary)
- **Description**:
  > Berkeley Function Calling Leaderboard (BFCL), the **first comprehensive and executable function call evaluation** dedicated to assessing Large Language Models' (LLMs) ability to invoke functions. Unlike previous evaluations, BFCL accounts for various forms of function calls, diverse scenarios, and executability. Need to run `pip install bfcl-eval==2025.10.27.1` before evaluating. [Usage Example](https://evalscope.readthedocs.io/en/latest/third_party/bfcl_v3.html)
- **Task Categories**: `Agent`, `FunctionCalling`
- **Evaluation Metrics**: `acc`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `train`
- **Subsets**: `irrelevance`, `java`, `javascript`, `live_irrelevance`, `live_multiple`, `live_parallel_multiple`, `live_parallel`, `live_relevance`, `live_simple`, `multi_turn_base`, `multi_turn_long_context`, `multi_turn_miss_func`, `multi_turn_miss_param`, `multiple`, `parallel_multiple`, `parallel`, `simple`

- **Extra Parameters**: 
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

[Back to Top](#agent-benchmarks)
- **Dataset Name**: `bfcl_v4`
- **Dataset ID**: [berkeley-function-call-leaderboard](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
- **Description**:
  > With function-calling being the building blocks of Agents, the Berkeley Function-Calling Leaderboard (BFCL) V4 presents a holistic agentic evaluation for LLMs. BFCL V4 Agentic includes web search, memory, and format sensitivity. Together, the ability to web search, read and write from memory, and the ability to invoke functions in different languages present the building blocks for the exciting and extremely challenging avenues that power agentic LLMs today from deep-research, to agents for coding and law. Need to run `pip install bfcl-eval==2025.10.27.1` before evaluating. [Usage Example](https://evalscope.readthedocs.io/en/latest/third_party/bfcl_v4.html)
- **Task Categories**: `Agent`, `FunctionCalling`
- **Evaluation Metrics**: `acc`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `train`
- **Subsets**: `irrelevance`, `live_irrelevance`, `live_multiple`, `live_parallel_multiple`, `live_parallel`, `live_relevance`, `live_simple`, `memory_kv`, `memory_rec_sum`, `memory_vector`, `multi_turn_base`, `multi_turn_long_context`, `multi_turn_miss_func`, `multi_turn_miss_param`, `multiple`, `parallel_multiple`, `parallel`, `simple_java`, `simple_javascript`, `simple_python`, `web_search_base`, `web_search_no_snippet`

- **Extra Parameters**: 
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

[Back to Top](#agent-benchmarks)
- **Dataset Name**: `general_fc`
- **Dataset ID**: [evalscope/GeneralFunctionCall-Test](https://modelscope.cn/datasets/evalscope/GeneralFunctionCall-Test/summary)
- **Description**:
  > A general function calling dataset for custom evaluation. For detailed instructions on how to use this benchmark, please refer to the [User Guide](https://evalscope.readthedocs.io/en/latest/advanced_guides/custom_dataset/llm.html#fc).
- **Task Categories**: `Agent`, `Custom`, `FunctionCalling`
- **Evaluation Metrics**: `count_finish_reason_tool_call`, `count_successful_tool_call`, `schema_accuracy`, `tool_call_f1`
- **Aggregation Methods**: `f1`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `default`


---

### τ²-bench

[Back to Top](#agent-benchmarks)
- **Dataset Name**: `tau2_bench`
- **Dataset ID**: [evalscope/tau2-bench-data](https://modelscope.cn/datasets/evalscope/tau2-bench-data/summary)
- **Description**:
  > τ²-bench (Tau Squared Bench) is an extension and enhancement of the original τ-bench (Tau Bench), which is a benchmark designed to evaluate conversational AI agents that interact with users through domain-specific API tools and guidelines. Please install it with `pip install git+https://github.com/sierra-research/tau2-bench@v0.2.0` before evaluating and set a user model. [Usage Example](https://evalscope.readthedocs.io/en/latest/third_party/tau2_bench.html)
- **Task Categories**: `Agent`, `FunctionCalling`, `Reasoning`
- **Evaluation Metrics**: 
- **Aggregation Methods**: `mean_and_pass_hat_k`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `airline`, `retail`, `telecom`

- **Extra Parameters**: 
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

[Back to Top](#agent-benchmarks)
- **Dataset Name**: `tau_bench`
- **Dataset ID**: [tau-bench](https://github.com/sierra-research/tau-bench)
- **Description**:
  > A benchmark emulating dynamic conversations between a user (simulated by language models) and a language agent provided with domain-specific API tools and policy guidelines. Please install it with `pip install git+https://github.com/sierra-research/tau-bench` before evaluating and set a user model. [Usage Example](https://evalscope.readthedocs.io/en/latest/third_party/tau_bench.html)
- **Task Categories**: `Agent`, `FunctionCalling`, `Reasoning`
- **Evaluation Metrics**: 
- **Aggregation Methods**: `mean_and_pass_hat_k`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `airline`, `retail`

- **Extra Parameters**: 
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

[Back to Top](#agent-benchmarks)
- **Dataset Name**: `tool_bench`
- **Dataset ID**: [AI-ModelScope/ToolBench-Static](https://modelscope.cn/datasets/AI-ModelScope/ToolBench-Static/summary)
- **Description**:
  > ToolBench is a benchmark for evaluating AI models on tool use tasks. It includes various subsets such as in-domain and out-of-domain, each with its own set of problems that require step-by-step reasoning to arrive at the correct answer. [Usage Example](https://evalscope.readthedocs.io/en/latest/third_party/toolbench.html)
- **Task Categories**: `FunctionCalling`, `Reasoning`
- **Evaluation Metrics**: `Act.EM`, `F1`, `HalluRate`, `Plan.EM`, `Rouge-L`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `in_domain`, `out_of_domain`

