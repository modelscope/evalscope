# Project Eval

EvalScope can execute an externally generated business-agent eval set through
the `project_eval` benchmark. Generation and execution are separate: the eval
set is authored outside EvalScope, and EvalScope does not read the project source
tree.

## Run through TaskConfig

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

`local_path` is the JSONL file, `project_root` is the runtime state directory
that contains the business DB/files used by end-state checks, and `agent_url`
is the business Agent called by EvalScope. Relative check paths resolve against
`project_root`.

## Contract

Each JSONL task contains a question or user-only `messages`, a reference
answer, expected/forbidden facts, rubric criteria, read-only checks and an
`exec` envelope. Checks can query SQLite, HTTP/JSON, a JSON file or a file;
`http` checks allow GET/HEAD only.

- A `must_pass` outcome criterion with any failed check zeroes the task.
- Otherwise the rubric is judged; `task_pass` requires every `must_pass`
  criterion and `rubric_score >= pass_threshold`.
- End-state checks run immediately after the Agent call and are saved with the
  prediction. Cached `rerun_review` reuses that saved evidence; it does not
  re-read live business state.
- Metrics are `task_pass`, `rubric_score` and `evidence_coverage`.

## Boundaries

- This change adds the core contract, Agent clients, checks, judge and report
  path. The Eval Sets management API/dashboard and the generation skill are
  follow-ups.
- L0 and L2 inline trace are supported; `trace.mode=fetch` (L1) is not
  implemented and is rejected.
- Process criteria are `not_evaluable` when the Agent returns no trace.
- Generation, task/Rubric editing and versioning stay outside EvalScope. The
  Judge sees the reference answer, expected/forbidden facts, rubric criteria and
  check evidence, not raw source.

## Generator

The external coding-agent skill
[evalscope-evalgen](https://github.com/frostloom/evalgen-skill) scans a project,
generates and validates the JSONL, and can dry-run it against the Agent. Its
automatic import needs the Eval Sets management follow-up.
