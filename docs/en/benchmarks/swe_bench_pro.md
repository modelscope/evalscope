# SWE-bench_Pro


## Overview

SWE-bench_Pro is a challenging benchmark from Scale AI evaluating LLMs/Agents on long-horizon software engineering tasks across multiple programming languages. Given a codebase and an issue, the model must autonomously explore the repository, edit source files, and submit a patch through a multi-turn agent loop inside a per-instance Docker container.

## Task Description

- **Task Type**: Automated Software Engineering / Bug Fixing (Agentic)
- **Input**: GitHub issue description
- **Output**: Code patch (diff format) collected after autonomous editing
- **Languages**: Multiple (`repo_language` field; e.g. JavaScript/TypeScript, Python, Go)

## Key Features

- Multi-turn agent loop with per-instance DockerHub image (`jefzda/sweap-images:{tag}`)
- Sentinel-based patch submission protocol (`COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`)
- Container-side evaluation: `git apply` patch, run instance's `run_script.sh`, parse with `parser.py`, then check `(fail_to_pass | pass_to_pass) ⊆ PASSED`
- Supports both `toolcall` (function-calling) and `backticks` (text-based) action protocols

## Evaluation Notes

- Requires `pip install evalscope[sandbox]` (provides Docker SDK via ms-enclave)
- Requires the `scaleapi/SWE-bench_Pro-os` repository for per-instance run scripts and Dockerfiles. By default this is auto-cloned to `~/.cache/evalscope/swe_bench_pro/SWE-bench_Pro-os` and pinned to commit `ca10a60`. To use an existing clone, set `extra_params.swe_bench_pro_repo_path`.
- Both the agent loop and the per-instance evaluation share a single sandbox configuration via `TaskConfig.sandbox.default_config` (passed straight to ms_enclave `DockerSandboxConfig`). Set `memory_limit` / `cpu_limit` there to avoid OOM-Killed test runs (e.g. NodeBB); `platform` defaults to `linux/amd64` so amd64-only sweap-images work on Apple Silicon out of the box.

See the [user guide](https://evalscope.readthedocs.io/en/latest/third_party/swe_bench_pro.html) for setup, parameters, and troubleshooting.


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `swe_bench_pro` |
| **Dataset ID** | [ScaleAI/SWE-bench_Pro](https://modelscope.cn/datasets/ScaleAI/SWE-bench_Pro/summary) |
| **Paper** | N/A |
| **Tags** | `Coding` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 731 |
| Prompt Length (Mean) | 1297.47 chars |
| Prompt Length (Min/Max) | 419 / 8036 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "3874fd29",
      "content": "\"**Title: Email Validation Status Not Handled Correctly in ACP and Confirmation Logic**\\n\\n**Description:**\\n\\nThe Admin Control Panel (ACP) does not accurately reflect the email validation status of users. Also, validation and confirmation p ... [TRUNCATED 846 chars] ... mail validation.\\n\\nThe email status was unclear or incorrect in ACP.\\n\\n\\\"Validate\\\" and \\\"Send validation email\\\" actions failed when the expected data was missing.\\n\\n**Labels:**\\n\\nbug, back-end, authentication, ui/ux, email-confirmation\""
    }
  ],
  "id": 0,
  "group_id": 0,
  "tools": [
    {
      "name": "bash",
      "description": "Execute a bash command inside the sandbox environment. Returns the combined stdout / stderr output of the command.",
      "parameters": {
        "properties": {
          "command": {
            "type": "string",
            "description": "The bash command to execute."
          },
          "timeout": {
            "type": "number",
            "description": "Maximum execution time in seconds (default: 60).",
            "default": 60
          }
        },
        "required": [
          "command"
        ]
      }
    }
  ],
  "metadata": {
    "instance_id": "instance_NodeBB__NodeBB-04998908ba6721d64eba79ae3b65a351dcfbc5b5-vnan",
    "repo": "NodeBB/NodeBB",
    "base_commit": "1e137b07052bc3ea0da44ed201702c94055b8ad2",
    "problem_statement": "\"**Title: Email Validation Status Not Handled Correctly in ACP and Confirmation Logic**\\n\\n**Description:**\\n\\nThe Admin Control Panel (ACP) does not accurately reflect the email validation status of users. Also, validation and confirmation p ... [TRUNCATED 846 chars] ... mail validation.\\n\\nThe email status was unclear or incorrect in ACP.\\n\\n\\\"Validate\\\" and \\\"Send validation email\\\" actions failed when the expected data was missing.\\n\\n**Labels:**\\n\\nbug, back-end, authentication, ui/ux, email-confirmation\"",
    "patch": "diff --git a/public/language/en-GB/admin/manage/users.json b/public/language/en-GB/admin/manage/users.json\nindex 6b668a31ef8e..9486295bc3ef 100644\n--- a/public/language/en-GB/admin/manage/users.json\n+++ b/public/language/en-GB/admin/manage/us ... [TRUNCATED 12136 chars] ... class=\"notvalidated fa fa-check text-muted\" title=\"not validated\"></i>\n+\t\t\t\t\t\t\t\t<i class=\"noemail fa fa-fw fa-ban text-muted\"\"></i>\n \t\t\t\t\t\t\t\t<em class=\"text-muted\">[[admin/manage/users:users.no-email]]</em>\n \t\t\t\t\t\t\t\t{{{ end }}}\n \t\t\t\t\t\t\t</td>\n",
    "test_patch": "diff --git a/test/database/keys.js b/test/database/keys.js\nindex 3941edb65a93..fde4bbc442cf 100644\n--- a/test/database/keys.js\n+++ b/test/database/keys.js\n@@ -35,6 +35,17 @@ describe('Key methods', () => {\n \t\t});\n \t});\n \n+\tit('should return m ... [TRUNCATED 952 chars] ... {uid}`, 1000);\n+\t\t\tconst code = await db.get(`confirm:byUid:${uid}`);\n+\t\t\tawait db.setObjectField(`confirm:${code}`, 'expires', Date.now() + 1000);\n \t\t\tconst ok = await user.email.canSendValidation(uid, email);\n-\n \t\t\tassert(ok);\n \t\t});\n \t});\n",
    "fail_to_pass": "[\"test/database.js | Test database test/database/keys.js::Key methods should return multiple keys and null if key doesn't exist\", 'test/database.js | Test database test/database/keys.js::Key methods should return empty array if keys is empty array or falsy', 'test/user/emails.js | email confirmation (library methods) canSendValidation should return true if it has been long enough to re-send confirmation']",
    "pass_to_pass": "[\"test/database.js | Test database should work\", \"test/database.js | Test database info should return info about database\", \"test/database.js | Test database info should not error and return info if client is falsy\", \"test/database.js | Test  ... [TRUNCATED 45280 chars] ... pending or set\", \"test/user/emails.js | email confirmation (v3 api) should confirm their email (using the pending validation)\", \"test/user/emails.js | email confirmation (v3 api) should still confirm the email (as email is set in user hash)\"]",
    "before_repo_set_cmd": "git reset --hard 1e137b07052bc3ea0da44ed201702c94055b8ad2\ngit clean -fd \ngit checkout 1e137b07052bc3ea0da44ed201702c94055b8ad2 \ngit checkout 04998908ba6721d64eba79ae3b65a351dcfbc5b5 -- test/database/keys.js test/user/emails.js",
    "selected_test_files_to_run": "[\"test/database.js\", \"test/database/keys.js\", \"test/user/emails.js\"]",
    "repo_language": "js",
    "requirements": "\"- The loadUserInfo(callerUid, uids) function should include logic to retrieve and attach `email:pending` and `email:expired` flags to each user object. These flags must be derived by resolving `confirm:byUid:<uid>` keys via the new `getConfi ... [TRUNCATED 3170 chars] ... rval check must compare the stored TTL timestamp if available (or, if TTL is unavailable, use the current time as the baseline) plus the configured interval against the max confirmation period, ensuring the system prevents excessive resends.\"",
    "interface": "\"Type: Method\\n\\nName: db.mget\\n\\nPath: src/database/mongo/main.js, src/database/postgres/main.js, src/database/redis/main.js\\n\\nInput: keys: string[] (An array of database keys to retrieve.)\\n\\nOutput: Promise<(string | null)[]> (A promise t ... [TRUNCATED 487 chars] ...  to the email address string, or `null` if no suitable email is found.)\\n\\nDescription: A utility function that retrieves the most appropriate email address for an administrative action like \\\"force validate\\\" or \\\"resend validation email\\\".\"",
    "issue_specificity": "[\"major_bug\",\"data_bug\",\"ui_ux_bug\"]",
    "issue_categories": "[\"back_end_knowledge\",\"database_knowledge\",\"authentication_authorization_knowledge\",\"ui_ux_knowledge\"]",
    "dockerhub_tag": "nodebb.nodebb-NodeBB__NodeBB-04998908ba6721d64eba79ae3b65a351dcfbc5b5",
    "docker_image": "jefzda/sweap-images:nodebb.nodebb-NodeBB__NodeBB-04998908ba6721d64eba79ae3b65a351dcfbc5b5"
  }
}
```

## Prompt Template

**Prompt Template:**
```text
{question}
```

## Extra Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `swe_bench_pro_repo_path` | `str` | `` | Local path to a clone of scaleapi/SWE-bench_Pro-os. If empty, auto-cloned to ~/.cache/evalscope/swe_bench_pro/SWE-bench_Pro-os and pinned to commit ca10a60. |
| `dockerhub_username` | `str` | `jefzda` | DockerHub user/org hosting the sweap-images repository. |
| `action_protocol` | `str` | `toolcall` | Agent action protocol: "toolcall" (function-calling) or "backticks" (text-based fallback for models without function-calling support). Choices: ['toolcall', 'backticks'] |
| `max_steps` | `int` | `250` | Maximum number of agent steps per sample. |
| `command_timeout` | `float` | `60.0` | Default per-bash-command timeout in seconds. |
| `eval_timeout` | `int` | `3600` | Per-instance evaluation timeout in seconds. |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets swe_bench_pro \
    --limit 10  # Remove this line for formal evaluation
```

### Using Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['swe_bench_pro'],
    dataset_args={
        'swe_bench_pro': {
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


