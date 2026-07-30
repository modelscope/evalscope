# MiniWoB


MiniWoB 通过 OpenEnv 和 BrowserGym 对多模态浏览器智能体在 125 个简短交互任务上的表现进行评估。
默认运行时每个任务使用一个确定性随机种子；若要运行完整的五种子评估，请设置 `repeats=5`（或 `--repeats 5`）。
每个 episode 默认最多允许 10 次模型/工具调用。

主要评估指标为 `success_rate`；`error_rate` 单独报告环境失败情况。默认观测包含可访问性树（accessibility tree）和截图，因此模型必须支持图像输入和函数调用。

有关安装、运行时配置、协议细节以及完整评估示例，请参阅 [MiniWoB 使用指南](../third_party/miniwob.html)。


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `miniwob` |
| **数据集 ID** | [BrowserGym](https://github.com/ServiceNow/BrowserGym) |
| **论文** | N/A |
| **标签** | `Agent`, `FunctionCalling`, `MultiModal`, `MultiTurn` |
| **指标** | `success_rate`, `error_rate` |
| **默认示例数量** | 0-shot |
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
      "id": "481bffa9",
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
    "openenv_task_name": "ascending-numbers",
    "observation_mode": "axtree_screenshot",
    "openenv_version": "0.4.1",
    "openenv_commit": "65c506ef94bb1f7279cb4359673b3ef81031d01f",
    "openenv_patch_sha256": "465b23aaf7b3b2cadd681495d694a7dad5ca1b36be0cfb5ce5780b94ac354668",
    "browsergym_version": "0.14.3",
    "browsergym_commit": "0a785fbed075224ae81ca9c1fe924f66050696fe",
    "miniwob_commit": "7fd85d71a4b60325c6585396ec4f48377d049838",
    "csv_sha256": "37117db27909a17b1b78035528472922c98c479a54619ac398dc256a7d2fef09",
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
