# 项目评测

EvalScope 可以通过 `project_eval` benchmark 执行外部生成的业务 Agent 评测集。
生成与执行是分开的：评测集在 EvalScope 外部编写，EvalScope 不读取项目源码树。

## 通过 TaskConfig 运行

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

`local_path` 是 JSONL 文件，`project_root` 是存放业务数据库/文件的运行时状态目录，
`agent_url` 是 EvalScope 调用的业务 Agent。相对检查路径按 `project_root` 解析。

## 契约

每个 JSONL 任务包含问题或仅 user 的 `messages`、参考答案、预期/禁止事实、
rubric、只读检查和 `exec` 包络。检查可以查询 SQLite、HTTP/JSON、JSON 文件或
普通文件；`http` 检查只允许 GET/HEAD。

- `must_pass` 的 outcome criterion 只要有检查失败，任务直接判 0。
- 否则进入 rubric 裁判；`task_pass` 要求所有 `must_pass` 命中且
  `rubric_score >= pass_threshold`。
- 终态检查在 Agent 调用后立即执行并随预测保存；缓存 `rerun_review` 复用保存的
  证据，不会重新读取实时业务状态。
- 指标是 `task_pass`、`rubric_score` 和 `evidence_coverage`。

## 边界

- 这次改动只包含核心契约、Agent 客户端、检查、裁判和报告路径。Eval Sets 管理
  API/面板和生成 skill 是后续改动。
- 支持 L0 和 L2 inline trace；本版本不实现 `trace.mode=fetch`（L1），会直接报错。
- Agent 不返回 trace 时，process 类判据为 `not_evaluable`。
- 题目生成、题目/Rubric 编辑和版本管理仍在 EvalScope 外部。Judge 看到的是参考
  答案、预期/禁止事实、rubric 和检查证据，不会看到原始源码。

## 生成器

外部编码 Agent skill
[evalscope-evalgen](https://github.com/frostloom/evalgen-skill) 可以扫描项目、
生成并校验 JSONL，并对 Agent 做 dry-run。自动导入依赖后续的 Eval Sets 管理改动。
