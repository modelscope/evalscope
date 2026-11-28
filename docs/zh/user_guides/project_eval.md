# 项目评测

EvalScope 可以通过 `project_eval` benchmark 执行外部生成的业务 Agent 评测集。
生成与执行是分开的：评测集在 EvalScope 外部编写，EvalScope 不读取项目源码树。

## 四步流程

### 1. JSONL 输入

每行一个任务：

```json
{"id":"refund-001","question":"把充值款退 45 元，先预览。","messages":[{"role":"user","content":"把充值款退 45 元，先预览。"},{"role":"user","content":"确认"}],"reference_answer":"退款 45 元，余额 1955 元。","rubric":[{"id":"c1","criterion":"存在一次已确认的 45 元退款","weight":1,"must_pass":true,"evidence_requirement":"outcome"},{"id":"c2","criterion":"回答完成预览/确认流程，不编造金额","weight":2,"must_pass":false,"evidence_requirement":"answer"}],"checks":[{"id":"audit-1","type":"sqlite","dsn":"agent.db","query":"select count(*) from agent_audit_log where session_id='refund-001' and confirmed=1","expect":1}],"exec":{"type":"agent_api","protocol":"rest_steps","agent_url_env":"EVAL_AGENT_URL","reset_url":"http://127.0.0.1:9201/v1/eval/reset","trace":{"mode":"auto"},"session":{"id_from":"sample_id"}}}
```

通过 `TaskConfig` 运行：

```python
from evalscope import TaskConfig, run_task

run_task(TaskConfig(
    model='business-agent',
    eval_type='mock_llm',
    datasets=['project_eval'],
    dataset_args={
        'project_eval': {
            'local_path': '/path/eval_set.v1.jsonl',
            'project_root': '/path/runtime-state',
            'agent_url': 'http://127.0.0.1:8080',
            # 可选：显式信任一个非 loopback 的只读状态 API。
            'check_url_prefixes': ['https://agent.company.com'],
        },
    },
    judge={
        'strategy': 'llm',
        'models': [{
            'model_id': 'your-judge-model',
            'api_url': 'https://api.example.com/v1',
            'api_key': '...',
            'eval_type': 'openai_api',
        }],
    },
))
```

`project_root` 是运行时状态目录，相对检查路径按它解析。任务带本地 checks
（file、JSON 或 SQLite）时它必填；缺失会直接报错，不再回退到当前工作目录。
只有远程 HTTP checks 时不需要本地 root。`check_url_prefixes` 默认为空，HTTP
检查只允许 loopback；业务状态 API 分开部署时再显式加入可信 http(s) 前缀
（不能包含 user info、query 或 fragment）。

### 2. Agent 调用

`exec.type: agent_api` 时，EvalScope 调用：

```http
POST {agent_url}/v1/eval/chat
Content-Type: application/json

{"session_id":"refund-001","message":"把充值款退 45 元，先预览。","system":"客服"}
```

响应需要字符串 `reply`；`steps` / `tool_calls` 可选携带内联工具调用。`system`
非空时会在每次 REST 请求中发送；单轮任务的 `message` 是 question，有
`messages` 时则逐条发送 user 消息。`exec.type: http` 直接发送配置的请求，响应
JSON 保存为 `agent_state` 并展示给 Judge。每个任务前会调用 loopback `reset_url`
恢复 seed 状态。

### 3. 保存终态证据

Agent 调用后，EvalScope 立即执行所有只读检查，把结构化结果写入预测 metadata：

```text
predictions/*.jsonl
  ├── metadata.agent_state        # 协议返回的 Agent 状态（存在时）
  └── metadata.runtime_checks[]
        ├── check_id
        ├── ok
        ├── evidence   # SQL rows、HTTP/JSON 值、文件路径等
        └── error
```

`judge_definition()` 读取 `metadata.runtime_checks`，不再重新读取实时业务状态。
因此缓存 `rerun_review` 使用的是该次 Agent 运行保存的证据。任务声明了 checks 时，
预测里没有保存检查、保存为空、记录无法解析，或 check id 与任务 checks 不一致，
review 会明确失败，而不是静默按另一份状态打分。

### 4. 报告

汇总报告包含 `task_pass`、`rubric_score` 和 `evidence_coverage`。逐题 rubric
verdicts 和检查证据写入 `reviews/*.jsonl`，属于报告产物；渲染它们的 Web 管理页
是后续改动。汇总指标不能替代逐题证据。

## 契约与限制

- `question` 必填；`messages` 是可选的 user 多轮补充。加载时拒绝重复 task id。
- 检查只读，支持文件、SQLite、JSON 和 HTTP/JSON；`http` 检查只允许 GET/HEAD，
  且 URL 必须是 http(s)。
- 带 `must_pass` outcome criterion 的任务，只要任意检查失败就判失败；只放必须通过的检查。
- 支持 L0 和 L2 inline trace；`trace.mode` 接受 `auto`、`inline`、`none`；
  `inline` 要求内联 steps，`auto` 接受 L0，`none` 关闭 trace 证据并清空内联
  trace id。L1（从外部 trace 系统 fetch）是后续改动，schema 不接受。
- Agent 不返回 trace 时，process 类判据为 `not_evaluable`；`must_pass` 的
  process 判据把本次运行标记为 `needs_review`，但不会直接判失败。若所有判据都是
  process 且没有 trace，该样本不计入指标并标记 `needs_review`。
- 题目生成、题目/Rubric 编辑和版本管理仍在 EvalScope 外部。Judge 看到的是实际
  user 轮次（有 `messages` 用 messages，否则用 question）、参考答案、预期/禁止
  事实、rubric、agent state 和检查证据，不会看到原始源码。
- 外部编码 Agent skill
  [evalscope-evalgen](https://github.com/frostloom/evalgen-skill) 负责生成并校验
  JSONL、对 Agent 做 dry-run；自动导入依赖后续的 Eval Sets 管理改动。
