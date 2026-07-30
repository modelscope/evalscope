# MiniWoB


## 概述

MiniWoB 用于评估多模态智能体是否能够完成简短的浏览器任务，例如点击按钮、填写表单、滚动页面以及拖拽元素。

## 任务描述

- **任务类型**：交互式浏览器任务
- **输入**：任务目标、无障碍树（accessibility tree）和屏幕截图
- **输出**：通过函数调用选择的浏览器操作
- **数据集**：125 个 MiniWoB 任务
- **指标**：任务完成率（`success_rate`）和环境错误率（`error_rate`）

## 评估说明

- 默认运行对每个任务评估一个确定性回合（episode）。
- 设置 `repeats=5` 可启用五回合评估计划。
- 每个回合默认最多允许 10 次模型/工具交互。
- 模型必须支持图像输入和函数调用。
- 安装与示例请参阅 [MiniWoB 使用指南](../third_party/miniwob.html)。

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `miniwob` |
| **数据集 ID** | [BrowserGym](https://github.com/ServiceNow/BrowserGym) |
| **论文** | N/A |
| **标签** | `Agent`, `FunctionCalling`, `MultiModal`, `MultiTurn` |
| **指标** | `success_rate`, `error_rate` |
| **默认提示方式** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 125 |
| 提示词长度（平均） | 77 字符 |
| 提示词长度（最小/最大） | 77 / 77 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "4b7219db",
      "content": "The task goal and browser observation are supplied when the episode is reset."
    }
  ],
  "target": "1",
  "id": 0,
  "group_id": 0,
  "tools": [
    {
      "name": "browser_action",
      "description": "Execute exactly one BrowserGym MiniWoB action. Supported signatures: noop(wait_ms=1000), mouse_move(x, y), mouse_click(x, y, button=\"left\"), mouse_dblclick(x, y, button=\"left\"), mouse_down(x, y, button=\"left\"), mouse_up(x, y, button=\"left\"),  ... [TRUNCATED 44 chars] ... \"left\"), keyboard_press(key), keyboard_type(text), fill(bid, value). click accepts a string BID, for example click(\"13\"); use mouse_click(x, y) for visual targets. Coordinates are absolute screenshot pixels, not normalized 0-1000 coordinates.",
      "parameters": {
        "properties": {
          "action": {
            "type": "string",
            "description": "One BrowserGym function-call expression."
          }
        },
        "required": [
          "action"
        ]
      }
    }
  ],
  "metadata": {
    "task_name": "miniwob.ascending-numbers",
    "miniwob_category": "hidden test",
    "comment": "",
    "webgum_subset": "False",
    "similarity_group": "0",
    "browsergym_split": "test",
    "task_id": "miniwob.ascending-numbers",
    "seed": 1608637542,
    "repeat": 0
  }
}
```

*注：部分内容因显示需要已被截断。*

## 提示模板

**提示模板：**
```text
{question}
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets miniwob \
    --agent-config '{"mode":"native","strategy":"function_calling","max_steps":10}' \
    --limit 10  # 正式评估时请删除此行
```

### 使用 Python

```python
from evalscope import TaskConfig, run_task
from evalscope.api.agent import NativeAgentConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['miniwob'],
    agent_config=NativeAgentConfig(
        strategy='function_calling',
        max_steps=10,
    ),
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
