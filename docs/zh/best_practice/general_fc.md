# 从对话到Agent：大模型工具调用能力的量化评测

在大模型从单纯的“对话者”向“自主代理（Agent）”进化的过程中，**工具调用（Function Calling / Tool Use）** 能力是核心中的核心。

然而，在实际开发中，开发者们经常面临这样的痛点：模型宣称“支持工具调用”，但在真实业务场景中表现却不尽如人意——要么在用户闲聊时突然乱调 API，要么生成的 JSON 参数缺胳膊少腿，导致后端服务频频报错。

那么如何给你的模型接口调用颁发一张合格的“工具驾驶证”？测试标准是什么？评测流程该如何设计？

此类评估不仅取决于模型的内在能力，还受到服务商推理实现、后处理逻辑以及提示工程（Prompt Engineering）等差异的显著影响，这些因素均会导致实际集成效果的不同。即便是本地部署，推理引擎的选择或模型量化精度的差异，也会直接干扰工具调用能力，进而影响最终的落地效果。例如，**月之暗面**（Moonshot AI）专门针对其 K2 模型推出了验证标准，旨在帮助开发者评估并对比不同 API 供应商在工具调用（Tool-call）场景下的实际表现。

我们相信这样客观的评测，绝不只局限于具体的某个模型，而是每个需要基于大模型模型来搭建实际应用的开发者，都需要的能力。本文将介绍如何使用 **EvalScope** 评测框架，通过一套通用的、开箱即用的最佳实践，来对模型“会不会用工具、能不能用好工具”进行量化评估。

## 一、 为什么工具调用评测如此重要？

在传统的评测中，我们往往关注模型在知识问答、逻辑推理或文本生成质量上的表现。但在 Agent 时代，评价标准发生了质变：从关注“言之有理”转向了关注“执行无误”。模型不仅要理解人类语言，更要能精准地讲“机器语言”。

根据 MoonshotAI 开源的 [K2-Vendor-Verifier](https://github.com/MoonshotAI/K2-Vendor-Verifier) 项目揭示的行业现状，许多模型在处理工具调用时存在两个致命缺陷：

1.  **决策边界模糊（Call vs. Not Call）**：模型分不清“用户在闲聊”还是“用户在下指令”。例如用户说“今天天气不错”，模型不应尝试调用天气 API，但很多模型会错误触发。
    
2.  **参数幻觉（Schema Violation）**：模型生成的参数往往缺失必填项，或者类型错误（例如将字符串传给数字字段），导致 API 调用直接失败。
    

如果无法解决这两个问题，构建的 Agent 就永远无法上线。因此，我们需要一个**确定性的、可回归的**评测流程，不仅要测“能不能调通”，还要测“该不该调”以及“参数对不对”。

## 二、 我们的解决方案

为了解决上述痛点，**EvalScope** 框架集成了标准化的 **GeneralFunctionCall** 评测流。我们参考了 MoonshotAI K2 的验证逻辑，将其封装为简单易用的评测工具，并做了通用性方面的扩展。

这套方案的核心价值在于：

*   **双重验证机制**：既验证**决策准确性**（模型是否正确判断了该不该调工具），也验证**参数合法性**（生成的 JSON 是否完美符合 Schema 定义）。
    
*   **高质量可扩展的基准数据**：即支持直接使用标准的预先合成数据，也支持评测数据的自定义。
    
*   **自动化指标计算**：直接输出 F1 Score 和 Schema 通过率，无需人工介入。
    

## 三、 快速上手指南

本节将指导你如何使用预置数据集或自定义业务数据来评测你的模型。

在此之前，你需要安装EvalScope库，运行下面的代码：

```shell
pip install 'evalscope[service]' -U
```

### 1. 准备数据

为了让评测工具“有的放矢”，我们需要准备标准的 JSONL 格式数据。

每一行代表一个测试样本，核心包含三个字段：

*   `messages`: 对话上下文。
    
*   `tools`: 模型可用的工具定义（工具箱）。
    
*   `should_call_tool`: **这是关键标签**。它告诉评测器，在这个语境下，标准答案是“调用工具 (True)”还是“直接回复 (False)”。
    

**标准数据样本示例：**

```json
{
  "messages": [
    { "role": "system", "content": "你是助手" },
    { "role": "user", "content": "把 37 摄氏度转换为华氏度" }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "convert_temperature",
        "description": "温度转换",
        "parameters": {
          "type": "object",
          "properties": {
             "celsius": { "type": "number", "description": "摄氏温度" }
          },
          "required": ["celsius"],
          "additionalProperties": false
        }
      }
    }
  ],
  "should_call_tool": true
}

```
> **💡 提示**：

*   为了测试模型的“抗干扰能力”，请务必包含 `should_call_tool: false` 的负样本（例如用户只是在打招呼）。
    
*   如果你没有自己的数据，可以直接使用 EvalScope 预置的 [evalscope/GeneralFunctionCall-Test](https://www.modelscope.cn/datasets/evalscope/GeneralFunctionCall-Test) 数据集。
    

### 2. 运行评测

我们极大地简化了代码复杂度。建议通过 API 接入模型进行评测（最接近真实线上环境）。

**场景 A：使用预置数据集（推荐初次体验）**

这里使用预置数据集 [evalscope/GeneralFunctionCall-Test](https://www.modelscope.cn/datasets/evalscope/GeneralFunctionCall-Test) ，根据 MoonshotAI [开源样本](https://github.com/MoonshotAI/K2-Vendor-Verifier)整理，其核心的“是否应触发工具”标签由 K2-Thinking 模型生成。这确保了评测标准的一致性，极大地方便了开发者进行实验复现和自动化回归测试。

```python
from evalscope import TaskConfig, run_task
import os

# 配置评测任务
task_cfg = TaskConfig(
    model='qwen-plus',  # 被测模型名称
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1', # OpenAI 兼容格式的 API 地址
    api_key=os.getenv('DASHSCOPE_API_KEY'), # 从环境变量获取 Key
    datasets=['general_fc'],  # 核心参数：指定使用 '工具调用(general_fc)' 评测模式，默认使用 evalscope/GeneralFunctionCall-Test
)

# 启动任务，自动打印结果
run_task(task_cfg=task_cfg)

```

**场景 B：使用自定义业务数据**

如果你的业务场景特殊（例如金融、医疗），建议建立自己的测试集。

首先，创建一个名为 `example.jsonl` 的文件，内容如下（包含正样本和负样本）：

```json
{"messages":[{"role":"system","content":"你是助手"},{"role":"user","content":"请把 2 和 3 相加"}],"tools":[{"type":"function","function":{"name":"add","description":"将两个数字相加","parameters":{"type":"object","properties":{"a":{"type":"number","description":"第一个数字"},"b":{"type":"number","description":"第二个数字"}},"required":["a","b"],"additionalProperties":false}}}],"should_call_tool":true}
{"messages":[{"role":"system","content":"你是助手"},{"role":"user","content":"今天天气不错，我们聊聊天"}],"tools":[{"type":"function","function":{"name":"add","description":"将两个数字相加","parameters":{"type":"object","properties":{"a":{"type":"number","description":"第一个数字"},"b":{"type":"number","description":"第二个数字"}},"required":["a","b"],"additionalProperties":false}}}],"should_call_tool":false}
{"messages":[{"role":"system","content":"你是助手"},{"role":"user","content":"把 37 摄氏度转换为华氏度"}],"tools":[{"type":"function","function":{"name":"convert_temperature","description":"将摄氏度转换为华氏度","parameters":{"type":"object","properties":{"celsius":{"type":"number","description":"摄氏温度值"}},"required":["celsius"],"additionalProperties":false}}}],"should_call_tool":true}
```

将上述文件保存到 `custom_eval/text/fc/example.jsonl`，然后运行以下代码：

```python
from evalscope import TaskConfig, run_task
import os

task_cfg = TaskConfig(
    model='qwen-plus',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    datasets=['general_fc'], 
    # 通过 dataset_args 注入本地数据路径
    dataset_args={
        'general_fc': {
            "local_path": "custom_eval/text/fc",  # 你的 JSONL 文件所在文件夹路径
            "subset_list": ["example"]            # 对应文件名（不含扩展名）
        }
    },
)

run_task(task_cfg=task_cfg)

```

### 3. 解读报告

评测完成后，控制台将输出一份指标报告。你需要重点关注以下四个核心指标：

| 指标名称 | 含义 | 说明 |
| --- | --- | --- |
| `tool_call_f1` | **决策 F1 分数** | 衡量模型是否“头脑清醒”。分数低说明模型经常在不需要调工具时乱调，或该调的时候没反应。这是 Agent 智能程度的第一道门槛。 |
| `schema_accuracy` | **参数格式准确率** | 衡量模型生成的 JSON 参数是否符合 Schema 定义。如果此项低，说明模型经常产生“幻觉参数”，会导致代码运行崩溃。 |
| `count_finish_reason_tool_call` | 触发调用次数 | 模型预测需要调用工具的总次数。 |
| `count_successful_tool_call` | 完美调用次数 | 既**决策正确**，且**参数也完全正确**的样本数。 |

**结果示例：**

```text
+-----------+------------+-------------------------------+----------+-------+---------+
| Model     | Dataset    | Metric                        | Subset   |   Num |   Score |
+===========+============+===============================+==========+=======+=========+
| modeltest | general_fc | count_finish_reason_tool_call | default  |    10 |  3      |
+-----------+------------+-------------------------------+----------+-------+---------+
| modeltest | general_fc | count_successful_tool_call    | default  |    10 |  2      |
+-----------+------------+-------------------------------+----------+-------+---------+
| modeltest | general_fc | schema_accuracy               | default  |    10 |  0.6667 |
+-----------+------------+-------------------------------+----------+-------+---------+
| modeltest | general_fc | tool_call_f1                  | default  |    10 |  0.5    |
+-----------+------------+-------------------------------+----------+-------+---------+

```

**简要点评：** 在这个示例中一共发送了10次请求，模型触发了 3 次工具调用（`count_finish_reason_tool_call`），但只有 2 次是完全成功的（`count_successful_tool_call`）。

*   **问题分析**：`schema_accuracy` 为 0.6667，说明有一次调用虽然触发了，但参数格式是错的（可能是缺了字段或类型不对）；`tool_call_f1` 仅为 0.5，说明模型在“该不该调”这个决策上存在严重的误判（漏调或误调）。
    
*   **改进建议**：该模型可能需要通过更多包含复杂 Schema 的样本进行微调，或者优化 Prompt 中的 System 指令以明确工具调用边界。
    

**可视化：**

EvalScope 内置了可视化分析工具，让你非常方便查看每一个样本的详细情况。无需编写额外代码，只需在终端运行一条命令，即可启动本地服务：

```shell
evalscope service
```

打开浏览器访问 `http://127.0.0.1:9000`即可查看模型得分分布和在每个问题的回复和打分：

![image.png](https://sail-moe.oss-cn-hangzhou.aliyuncs.com/yunlin/images/evalscope/doc/best_practice/general_fc_overview.png)

![image.png](https://sail-moe.oss-cn-hangzhou.aliyuncs.com/yunlin/images/evalscope/doc/best_practice/general_fc_detail.png)

## 四、 总结

工具调用能力是连接 LLM 与真实世界的桥梁。通过引入 **EvalScope** 的工具调用准确性和有效性的验证标准，我们可以不再依赖主观感觉来评价模型，而是用数据说话。

这套最佳实践不仅能帮助你筛选出最适合业务的模型，还能作为后续模型微调（Fine-tuning）或 Prompt 优化的“标尺”，确保你的 Agent 能力稳步提升。

现在，就用这套工具为你的模型做一次全面的体检吧！