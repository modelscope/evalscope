# MiniWoB

MiniWoB 用于评测多模态浏览器智能体完成点击、表单填写、滚动和拖放等短任务的能力。EvalScope 直接运行
BrowserGym，并使用环境 reward 判断任务是否成功。

## 安装

```bash
pip install 'evalscope[miniwob]'
playwright install chromium
```

第一条命令安装 Python 依赖。Chromium 是由 Playwright 管理的平台相关浏览器程序，无法包含在 Python extra
中，因此需要通过第二条命令单独安装一次。首次运行时，EvalScope 还会下载并缓存 MiniWoB 任务页面。运行不
需要 Docker。

## 快速开始

```bash
evalscope eval \
  --model qwen3-vl-plus \
  --datasets miniwob \
  --limit 10 \
  --eval-batch-size 4
```

观测同时包含无障碍树和截图，因此模型需要支持图像输入和 function calling。

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

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `repeats` | `1` | 每个任务的确定性 episode 数 |
| `eval_batch_size` | `1` | 并发模型调用数；BrowserGym 操作仍为串行 |
| `limit` | 未设置 | 重复前选取的任务数 |
| `agent_config.max_steps` | `10` | 每个 episode 的模型/工具轮数 |

模型只会看到一个 `browser_action` 函数，其 `action` 参数包含一个 BrowserGym `miniwob_all` 表达式，例如
`click("13")`、`fill("7", "text")` 或 `mouse_click(420, 260)`。坐标动作使用截图的绝对像素坐标。

## 评测协议

每个 episode 都在新的浏览器上下文中运行。当 MiniWoB 返回正 reward 时，任务判定为成功。
`success_rate` 表示任务完成率，`error_rate` 表示未能正常运行的 episode 比例。

完整调度对每个任务运行五个确定性 episode，并使用 10 步预算。设置 `limit`、使用默认单 episode 调度或
自定义步数预算后，结果不应与完整调度结果直接比较。
