# Project Eval

EvalScope can execute an externally generated business-agent eval set through
the `project_eval` benchmark. Generation and execution are separate: the eval
set is authored outside EvalScope, and EvalScope does not read the project source
tree.

## Four-step walkthrough

### 1. JSONL input

Each line is one task:

```json
{"id":"refund-001","question":"把充值款退 45 元，先预览。","messages":[{"role":"user","content":"把充值款退 45 元，先预览。"},{"role":"user","content":"确认"}],"reference_answer":"退款 45 元，余额 1955 元。","rubric":[{"id":"c1","criterion":"存在一次已确认的 45 元退款","weight":1,"must_pass":true,"evidence_requirement":"outcome"},{"id":"c2","criterion":"回答完成预览/确认流程，不编造金额","weight":2,"must_pass":false,"evidence_requirement":"answer"}],"checks":[{"id":"audit-1","type":"sqlite","dsn":"agent.db","query":"select count(*) from agent_audit_log where session_id='refund-001' and confirmed=1","expect":1}],"exec":{"type":"agent_api","protocol":"rest_steps","agent_url_env":"EVAL_AGENT_URL","reset_url":"http://127.0.0.1:9201/v1/eval/reset","trace":{"mode":"auto"},"session":{"id_from":"sample_id"}}}
```

Run it through `TaskConfig`:

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
            # Optional: explicitly trust a non-loopback read-only state API.
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

`project_root` is the runtime state directory; relative check paths resolve
against it. It is required when a task has local checks (file, JSON or SQLite);
a missing root fails the run instead of falling back to the current working
directory. Remote-only HTTP checks do not need a local root. `check_url_prefixes`
is empty by default, so HTTP checks are limited to loopback; add a trusted
http(s) prefix (without user info, query or fragment) when the business state
API is deployed separately.

### 2. Agent call

For `exec.type: agent_api`, EvalScope calls:

```http
POST {agent_url}/v1/eval/chat
Content-Type: application/json

{"session_id":"refund-001","message":"把充值款退 45 元，先预览。","system":"客服"}
```

The response needs a string `reply`; `steps` / `tool_calls` optionally carry
inline tool calls. A non-empty `system` is sent on every REST request. For a
single-turn task the `message` is the question; when `messages` is present,
each user turn is sent as the `message` instead. For `exec.type: http`, the
configured request is sent directly, its JSON body is saved as `agent_state`,
and that state is shown to the Judge. A loopback `reset_url` is called before
each task to restore the seed state.

### 3. Saved end-state evidence

Immediately after the Agent call, EvalScope runs every read-only check and
writes the structured results into the prediction metadata:

```text
predictions/*.jsonl
  ├── metadata.agent_state        # Agent state returned by the protocol, when present
  └── metadata.runtime_checks[]
        ├── check_id
        ├── ok
        ├── evidence   # SQL rows, HTTP/JSON value, file path, ...
        └── error
```

`judge_definition()` reads `metadata.runtime_checks`; it does not re-read live
business state. A cached `rerun_review` therefore uses the evidence captured
with that Agent run. When a task declares checks, a prediction with no saved
checks, an empty list, a corrupt record, or check ids that do not match the task
checks fails review loudly instead of silently scoring a different state.

### 4. Report

The aggregate report contains `task_pass`, `rubric_score` and
`evidence_coverage`. Per-sample rubric verdicts and check evidence are written
to `reviews/*.jsonl` and are part of the report artifacts; the management UI
that renders them is a follow-up. The summary is not a substitute for the
per-sample evidence.

## Contract and limits

- `question` is required; `messages` are an optional multi-turn supplement.
  Duplicate task ids are rejected when the eval set loads.
- Checks are read-only and support files, SQLite, JSON and HTTP/JSON; `http`
  checks allow GET/HEAD only and require an http(s) URL.
- A task with a `must_pass` outcome criterion fails when any of its checks
  fails; include only checks that must pass for the outcome.
- L0 and L2 inline trace are supported. `trace.mode` accepts `auto`, `inline`
  and `none`; `inline` requires inline steps, `auto` accepts L0, and `none`
  disables trace evidence and clears the inline trace id. L1 (`fetch` an
  external trace store) is a follow-up and is not accepted by the schema.
- Process criteria are `not_evaluable` when the Agent returns no trace; a
  `must_pass` process criterion marks the run `needs_review` without failing it.
  If every criterion is a process criterion and no trace exists, the sample is
  excluded from metrics and marked `needs_review`.
- Generation, task/Rubric editing and versioning stay outside EvalScope. The
  Judge sees the actual user turns (`messages` when present, otherwise
  `question`), reference answer, expected/forbidden facts, rubric criteria,
  agent state and check evidence, not raw source.
- The external coding-agent skill
  [evalscope-evalgen](https://github.com/frostloom/evalgen-skill) generates and
  validates the JSONL and can dry-run it against the Agent. Its automatic import
  needs the Eval Sets management follow-up.
