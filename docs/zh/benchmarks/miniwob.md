# MiniWoB


## 概述

MiniWoB 用于评估浏览器智能体在点击、表单填写、拖放和导航等简短交互任务上的表现。EvalScope 负责管理 episode 调度、模型循环、评分、轨迹记录和报告生成。一个固定版本的 OpenEnv v0.4.1 BrowserGym 服务负责环境生命周期管理以及 reset/step/reward 协议。

## 评估

- 默认调度包含 125 个程序化 episode：对应 125 个 BrowserGym 0.14.3 任务，每个任务使用一个确定性随机种子。设置 `TaskConfig.repeats=5`（或在 CLI 中传入 `--repeats 5`）可运行完整的 BrowserGym 调度，即 625 个 episode（每个任务使用五个不同的确定性种子）。
- 任务目录从一个固定的 BrowserGym GitHub 提交中下载一次，经校验和验证后本地缓存。**不使用** ModelScope 或 Hugging Face 数据集。
- 主要指标为 `success_rate`；`error_rate` 单独报告 OpenEnv 运行时失败情况。
- 每个 episode 默认使用 10 步动作预算，与 BrowserGym Experiments 一致。仅在诊断或自定义运行时才通过 `NativeAgentConfig.max_steps` 覆盖该值。
- `agent_config.task_environment.observation_mode` 控制观测表示形式。默认值为 `axtree_screenshot`：每次 reset 和 step 都会同时提供无障碍树（accessibility tree）和 PNG 截图。仅当明确需要纯文本诊断运行时，才使用 `axtree`。
- 截图模式要求模型支持图像输入并具备函数调用能力。纯文本模型可能会拒绝请求、忽略图像，或仅基于不完整的无障碍树进行操作；此类得分不能代表默认的多模态配置下的真实性能。

## 动作与运行时配置

本地运行时会对 OpenEnv v0.4.1 应用一个由 EvalScope 管理、经校验和固定的补丁，使 BrowserGym 使用其官方的 `miniwob_all` 动作配置，并保留每个 MiniWoB 任务原有的视口（viewport）和超时设置，而非覆盖为 OpenEnv 服务器的默认值。BrowserGym 本身未被 fork 或修改。报告中会记录 OpenEnv 的源代码提交哈希和补丁校验和。

默认动作配置和 10 步预算与 BrowserGym 0.14.3 一致。完整的 BrowserGym 评估协议还要求 `TaskConfig.repeats=5` 且使用完整调度；仅当上述三个条件全部满足时，报告才会设置 `official_browsergym_evaluation_protocol=true`。来自较轻量的单种子默认配置、有限运行或自定义步数预算的得分，**不得**直接与官方排行榜结果比较。

## 依赖要求

通过 `pip install 'evalscope[miniwob]'` 安装。
MiniWoB 目前仅支持本地 `ms_enclave_docker` 运行时，该运行时需要 Docker，并在首次使用时从固定的 OpenEnv GitHub 提交构建打过补丁的镜像。
在构建镜像前设置 `EVALSCOPE_PIP_INDEX_URL` 可指定自定义的 Python 包索引。
推荐最大并发数为 `eval_batch_size=4`。

本地模式示例：

```python
TaskConfig(model='qwen3-vl-plus', datasets=['miniwob'], eval_batch_size=4)
```

完整五种子调度示例：

```python
TaskConfig(model='qwen3-vl-plus', datasets=['miniwob'], repeats=5, eval_batch_size=4)
```


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `miniwob` |
| **数据集ID** | [BrowserGym](https://github.com/ServiceNow/BrowserGym) |
| **论文** | N/A |
| **标签** | `Agent`, `FunctionCalling`, `MultiModal`, `MultiTurn` |
| **指标** | `success_rate`, `error_rate` |
| **默认示例数量** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 125 |
| 提示词长度（平均） | 85 字符 |
| 提示词长度（最小/最大） | 85 / 85 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "6f138905",
      "content": "The task goal and browser observation are supplied when the OpenEnv episode is reset."
    }
  ],
  "target": "1",
  "id": 0,
  "group_id": 0,
  "tools": [
    {
      "name": "browser_action",
      "description": "This is the only browser tool. Always call the tool named browser_action; never call click, fill, press, or another BrowserGym action as a tool name. Put exactly one OpenEnv BrowserGym action expression in the action string. Supported signatu ... [TRUNCATED 461 chars] ... st be absolute pixels in the supplied screenshot, not normalized 0-1000 coordinates. The observation states the exact screenshot width and height. Examples: mouse_click(420, 260), fill(\"7\", \"text\"), keyboard_press(\"ENTER\"), or scroll(0, 300).",
      "parameters": {
        "properties": {
          "action": {
            "type": "string",
            "description": "Exactly one BrowserGym function-call expression."
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
    "profile": "openenv_v0.4.1_miniwob_all_10_steps",
    "max_steps": 10,
    "repeats": 1,
    "official_browsergym_action_config": true,
    "official_browsergym_evaluation_protocol": false,
    "openenv_version": "0.4.1",
    "openenv_commit": "65c506ef94bb1f7279cb4359673b3ef81031d01f",
    "openenv_patch_sha256": "465b23aaf7b3b2cadd681495d694a7dad5ca1b36be0cfb5ce5780b94ac354668",
    "browsergym_version": "0.14.3",
    "browsergym_commit": "0a785fbed075224ae81ca9c1fe924f66050696fe",
    "miniwob_commit": "7fd85d71a4b60325c6585396ec4f48377d049838",
    "csv_sha256": "37117db27909a17b1b78035528472922c98c479a54619ac398dc256a7d2fef09",
    "runtime_mode": null,
    "observation_mode": "axtree_screenshot",
    "seed": 1608637542,
    "repeat": 0
  }
}
```

*注：部分内容因显示原因已被截断。*

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
    --limit 10  # 正式评估时请删除此行
```

### 使用 Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['miniwob'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
