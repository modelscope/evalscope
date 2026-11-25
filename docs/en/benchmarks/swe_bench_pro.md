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

- Supports official E2B and Aliyun Agent Sandbox through `TaskConfig.sandbox.engine='e2b'`, avoiding local Docker and local image downloads. See the [sandbox guide](https://evalscope.readthedocs.io/en/latest/user_guides/sandbox.html#e2b).
- Requires `pip install evalscope[sandbox]` (provides sandbox runtimes via ms-enclave)
- Requires the `scaleapi/SWE-bench_Pro-os` repository for per-instance run scripts and Dockerfiles. By default this is auto-cloned to `~/.cache/evalscope/swe_bench_pro/SWE-bench_Pro-os` and pinned to commit `ca10a60`. To use an existing clone, set `extra_params.swe_bench_pro_repo_path`.
- Both the agent loop and the per-instance evaluation share a single sandbox configuration via `TaskConfig.sandbox.default_config`, passed to ms-enclave for the selected engine. For Docker, set `memory_limit` / `cpu_limit` there to avoid OOM-Killed test runs (e.g. NodeBB); `platform` defaults to `linux/amd64` so amd64-only sweap-images work on Apple Silicon out of the box.

See the [user guide](https://evalscope.readthedocs.io/en/latest/third_party/swe_bench_pro.html) for setup, parameters, and troubleshooting.


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `swe_bench_pro` |
| **Dataset ID** | [ScaleAI/SWE-bench_Pro](https://modelscope.cn/datasets/ScaleAI/SWE-bench_Pro/summary) |
| **Paper** | N/A |
| **Tags** | `Coding` |
| **Metrics** | `accuracy` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 642 |
| Prompt Length (Mean) | 1013.12 chars |
| Prompt Length (Min/Max) | 191 / 8010 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "10506721",
      "content": "## Title: Post cache access and slug existence checks behave inconsistently\n\n\n### Description\nPost cache availability depends on when different application modules load and access the cache, which can cause configuration-dependent initialization and inconsistent cache operations.\nSlug existence checks also only handle a single slug reliably. Supplying multiple slugs does not return an ordered result for each input, and arrays containing invalid values are not rejected consistently."
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
    "instance_id": "instance_NodeBB__NodeBB-00c70ce7b0541cfc94afe567921d7668cdc8f4ac-vnan",
    "repo": "NodeBB/NodeBB",
    "base_commit": "ae3fa85f40db0dca83d19dda1afb96064dffcc83",
    "problem_statement": "## Title: Post cache access and slug existence checks behave inconsistently\n\n\n### Description\nPost cache availability depends on when different application modules load and access the cache, which can cause configuration-dependent initialization and inconsistent cache operations.\nSlug existence checks also only handle a single slug reliably. Supplying multiple slugs does not return an ordered result for each input, and arrays containing invalid values are not rejected consistently.",
    "patch": "diff --git a/src/controllers/admin/cache.js b/src/controllers/admin/cache.js\nindex 43d6f4ddcab4..2faad03fc2bb 100644\n--- a/src/controllers/admin/cache.js\n+++ b/src/controllers/admin/cache.js\n@@ -6,7 +6,7 @@ const utils = require('../../utils' ... [TRUNCATED 7281 chars] ... ess-useragent');\n const favicon = require('serve-favicon');\n-const detector = require('spider-detector');\n+const detector = require('@nodebb/spider-detector');\n const helmet = require('helmet');\n \n const Benchpress = require('benchpressjs');\n",
    "test_patch": "diff --git a/test/mocks/databasemock.js b/test/mocks/databasemock.js\nindex 5cb8af1d34..507f29d6fc 100644\n--- a/test/mocks/databasemock.js\n+++ b/test/mocks/databasemock.js\n@@ -194,7 +194,7 @@ async function setupMockDefaults() {\n \tmeta.config. ... [TRUNCATED 4772 chars] ...  } });\n+\t\t\tassert.strictEqual(await meta.slugTaken('test-category-handle'), true);\n+\t\t\tassert.deepStrictEqual(await meta.slugTaken(['test-category-handle', 'doesnot exist']), [true, false]);\n \t\t});\n \n \t\tit('should delete user', async () => {\n",
    "fail_to_pass": "[\"test/user.js | User | socket methods | should return true/false if user/group exists or not\", \"test/socket.io.js | socket.io | should toggle caches\"]",
    "pass_to_pass": "[]",
    "before_repo_set_cmd": "git reset --hard ae3fa85f40db0dca83d19dda1afb96064dffcc83\ngit clean -fd\ngit checkout ae3fa85f40db0dca83d19dda1afb96064dffcc83\ngit apply --verbose /tests/test_patch.patch",
    "selected_test_files_to_run": "[\"test/mocks/databasemock.js\", \"test/translator.js\", \"test/user.js\", \"test/socket.io.js\", \"test/database.js\", \"test/meta.js\"]",
    "repo_language": "js",
    "requirements": "- The post cache must use a shared instance so all cache operations observe the same state.\n\n- The post cache module must expose `getOrCreate()`, which returns the shared cache instance and creates it only when first requested.\n\n- All modules ... [TRUNCATED 1394 chars] ... er with display name `John Smith` exists.\n\n- `User.getUidsByUserslugs` must return `null` for each slug that matches no user, and for known slugs each value must be strictly equal (`===`) to what `User.getUidByUserslug` returns for that slug.",
    "interface": "- Path: `src/posts/cache.js`\n- Name: `getOrCreate`\n- Type: function\n- Input: `NA`\n- Output: `Cache instance object`\n- Description: Returns the shared post cache instance, creating it on first use.\n\n- Path: `src/posts/cache.js`\n- Name: `reset` ... [TRUNCATED 365 chars] ...  other entry in place.\n\n- Path: `src/user/index.js`\n- Name: `getUidsByUserslugs`\n- Type: function\n- Input: `userslugs (string array)`\n- Output: `array of uid values`\n- Description: Returns the user ids corresponding to each userslug in order.",
    "issue_specificity": "[\"integration_bug\"]",
    "issue_categories": "[\"back_end_knowledge\",\"database_knowledge\",\"performance_knowledge\"]",
    "dockerhub_tag": "nodebb.nodebb-NodeBB__NodeBB-00c70ce7b0541cfc94afe567921d7668cdc8f4ac",
    "docker_image": "jefzda/sweap-images:nodebb.nodebb-NodeBB__NodeBB-00c70ce7b0541cfc94afe567921d7668cdc8f4ac"
  }
}
```

*Note: Some content was truncated for display.*

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
| `eval_timeout` | `int` | `3600` | Per-instance evaluation timeout in seconds. |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets swe_bench_pro \
    --agent-config '{"mode":"native","strategy":"swe_bench_toolcall","max_steps":250}' \
    --limit 10  # Remove this line for formal evaluation
```

### Using Python

```python
from evalscope import TaskConfig, run_task
from evalscope.api.agent import NativeAgentConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['swe_bench_pro'],
    agent_config=NativeAgentConfig(
        strategy='swe_bench_toolcall',
        max_steps=250,
    ),
    dataset_args={
        'swe_bench_pro': {
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
