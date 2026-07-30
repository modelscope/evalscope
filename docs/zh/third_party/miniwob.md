# MiniWoB

MiniWoB 用于评测多模态浏览器智能体完成点击、表单填写、滚动和拖放等短任务的能力。EvalScope 通过由
BrowserGym 提供能力的 OpenEnv 服务运行每个 episode。

## 安装

```bash
pip install 'evalscope[miniwob]'
```

运行需要 Docker。首次评测会构建固定版本的本地镜像，后续运行会复用该镜像。如果镜像构建需要使用自定义
Python 软件源，请在评测前设置 `EVALSCOPE_PIP_INDEX_URL`。

## 快速开始

```bash
evalscope eval \
  --model qwen3-vl-plus \
  --datasets miniwob \
  --limit 10 \
  --eval-batch-size 4
```

默认观测同时包含无障碍树和截图，因此模型需要支持图像输入和 function calling。

默认数据集为 125 个任务各生成一个确定性 episode。使用 `--repeats 5` 可运行完整的五种子调度：

```bash
evalscope eval \
  --model qwen3-vl-plus \
  --datasets miniwob \
  --repeats 5 \
  --eval-batch-size 4
```

`limit` 在重复之前生效。例如，`--limit 10 --repeats 5` 会从 10 个任务生成 50 个 episode。

## 配置

多数用户只需要设置顶层评测参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `repeats` | `1` | 每个任务的确定性 episode 数 |
| `eval_batch_size` | `1` | 并发 episode 数；普通本地机器建议不超过 `4` |
| `limit` | 未设置 | 重复前选取的任务数 |

高级环境选项位于 `agent_config.task_environment`；`max_steps` 仍属于原生 Agent 配置：

```python
from evalscope import TaskConfig, run_task
from evalscope.api.agent import NativeAgentConfig

run_task(
    TaskConfig(
        model='qwen3-vl-plus',
        datasets=['miniwob'],
        repeats=5,
        eval_batch_size=4,
        agent_config=NativeAgentConfig(
            max_steps=10,
            task_environment={
                'backend': 'openenv',
                'observation_mode': 'axtree_screenshot',
                'runtime': {
                    'name': 'ms_enclave_docker',
                },
            },
        ),
    )
)
```

`observation_mode='axtree'` 只适合纯文本诊断，不能代表默认的多模态评测。

## 接入结构

`OpenEnvAdapter` 负责可复用的 episode 流程：

1. 启动配置的服务 runtime 并获取服务地址。
2. 创建 OpenEnv session 并 reset episode。
3. 运行 EvalScope 标准 AgentLoop。
4. 将工具动作转发给 `session.step(...)`。
5. 记录 reward、trace 和错误，然后关闭 session 与 runtime handle。

benchmark 子类只需要提供数据调度、镜像与环境变量、reset 参数、action 映射和 observation 格式化。action
映射不能完全通用：MiniWoB v0.4.1 通过 `action_str` 接收 BrowserGym 表达式，而 OpenApp 等其他 OpenEnv
环境使用结构化 action 字段。

模型只会看到一个 `browser_action` 函数，其 `action` 参数包含一个 BrowserGym `miniwob_all` 表达式，例如
`click("13")`、`fill("7", "text")` 或 `mouse_click(420, 260)`。坐标动作使用截图的绝对像素坐标。

## 可复现性

当前接入固定使用 OpenEnv v0.4.1 和 BrowserGym v0.14.3。本地镜像会应用带校验和的兼容补丁，使服务使用
BrowserGym 的 `miniwob_all` action 配置，并保留任务自己的 viewport 和 timeout。

BrowserGym 完整评测调度为每个任务五个确定性种子和 10 步预算。设置 `limit`、使用默认单种子调度或自定义
步数预算后，结果不应与完整调度结果直接比较。
