# MiniWoB (OpenEnv profile)


## 概述

MiniWoB 用于评估浏览器智能体在点击、表单填写、拖放和导航等短时交互任务上的表现。EvalScope 负责管理 episode 调度、模型循环、评分、轨迹记录和报告生成。一个固定的 OpenEnv v0.4.1 BrowserGym 服务负责环境生命周期管理以及 reset/step/reward 协议。

## 评估

- 调度包含 625 个程序化 episodes：125 个 BrowserGym 0.14.3 任务，每个任务使用五个确定性随机种子。
- 任务目录从一个固定的 BrowserGym GitHub 提交版本一次性下载，并经过校验和验证后本地缓存。未使用 ModelScope 或 Hugging Face 数据集。
- 主要指标为 `success_rate`；`error_rate` 单独报告 OpenEnv 运行时失败情况。
- 每个 episode 使用固定的 20 步动作预算。
- 默认的 `observation_mode` 为 `axtree_screenshot`：每次 reset 和 step 均同时提供无障碍树（accessibility tree）和 PNG 截图。仅当明确需要纯文本诊断运行时才使用 `axtree` 模式。
- 截图模式要求模型支持图像输入并具备函数调用能力。纯文本模型可能会拒绝请求、忽略图像，或仅基于不完整的无障碍树进行操作；此类得分不能代表默认的多模态配置性能。

## 动作与运行时配置

本地运行时会对 OpenEnv v0.4.1 应用一个由 EvalScope 管理且通过校验和固定的补丁，使得 BrowserGym 使用其官方的 `miniwob_all` 动作配置，并保留每个 MiniWoB 任务原有的视口大小和超时设置，而非覆盖为 OpenEnv 服务器的默认值。BrowserGym 本身未被 fork 或修改。报告中会记录 OpenEnv 的源代码提交哈希和补丁校验和。

该动作配置与 BrowserGym 0.14.3 一致，但 EvalScope 配置使用 20 步预算，而非 BrowserGym Experiments 官方的 10 步预算。因此，报告中标记为 `official_browsergym_action_config=true` 且 `official_browsergym_evaluation_protocol=false`；其得分不得与官方排行榜直接比较。

## 依赖要求

通过以下命令安装：
`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple 'evalscope[miniwob]'`。

MiniWoB 目前仅支持本地 `ms_enclave_docker` 运行时，该运行时需要 Docker，并在首次使用时从一个固定的 OpenEnv GitHub 提交构建打过补丁的镜像。镜像从清华大学 PyPI 镜像安装 Python 依赖项。推荐的最大并发数为 `eval_batch_size=4`。

通用的 EvalScope `remote` 环境运行时仍可用于其他环境后端。但在 EvalScope 实现标准的能力/配置握手协议（用于验证远程动作映射和源配置）之前，MiniWoB 不接受该远程运行时。

本地模式示例：

```python
TaskConfig(model='qwen3-vl-plus', datasets=['miniwob'], eval_batch_size=4)
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
| 总样本数 | 625 |
| 提示词长度（平均） | 85 字符 |
| 提示词长度（最小/最大） | 85 / 85 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "f2bedbc8",
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
    "seed": 28,
    "repeat": 0,
    "profile": "openenv_v0.4.1_miniwob_all_20_steps",
    "max_steps": 20,
    "official_browsergym_action_config": true,
    "official_browsergym_evaluation_protocol": false,
    "openenv_version": "0.4.1",
    "openenv_commit": "65c506ef94bb1f7279cb4359673b3ef81031d01f",
    "openenv_patch_sha256": "b90bb3f1b91c60a8d4b7c888cccd78f1834754b696448da039e1bba7addd836a",
    "browsergym_version": "0.14.3",
    "browsergym_commit": "0a785fbed075224ae81ca9c1fe924f66050696fe",
    "miniwob_commit": "7fd85d71a4b60325c6585396ec4f48377d049838",
    "csv_sha256": "37117db27909a17b1b78035528472922c98c479a54619ac398dc256a7d2fef09",
    "runtime_mode": null,
    "observation_mode": "axtree_screenshot"
  }
}
```

*注：部分内容因显示原因已被截断。*

## 提示模板

**提示模板：**
```text
{question}
```

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `observation_mode` | `str` | `axtree_screenshot` | reset 和每次动作后提供的浏览器观测信息。可选值：['axtree', 'axtree_screenshot'] |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets miniwob \
    --agent-config '{"mode":"native","strategy":"miniwob_openenv_function_calling","max_steps":20}' \
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
        strategy='miniwob_openenv_function_calling',
        max_steps=20,
    ),
    dataset_args={
        'miniwob': {
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
