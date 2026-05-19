# SWE-bench_Pro


## 概述

SWE-bench_Pro 是由 Scale AI 提供的一项具有挑战性的基准测试，用于评估大语言模型（LLM）或智能体在多种编程语言环境下执行长周期软件工程任务的能力。给定一个代码库和一个问题描述，模型必须在每个实例专属的 Docker 容器内，通过多轮智能体交互循环，自主探索代码仓库、编辑源文件，并最终提交补丁。

## 任务描述

- **任务类型**：自动化软件工程 / 缺陷修复（智能体驱动）
- **输入**：GitHub issue 描述
- **输出**：自主编辑后收集的代码补丁（diff 格式）
- **支持语言**：多种（由 `repo_language` 字段指定；例如 JavaScript/TypeScript、Python、Go）

## 核心特性

- 基于每实例 DockerHub 镜像（`jefzda/sweap-images:{tag}`）的多轮智能体交互循环
- 基于哨兵（sentinel）的补丁提交协议（`COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`）
- 容器端评估流程：应用 `git apply` 打补丁，运行实例对应的 `run_script.sh` 脚本，通过 `parser.py` 解析结果，并验证 `(fail_to_pass | pass_to_pass) ⊆ PASSED`
- 同时支持 `toolcall`（函数调用）和 `backticks`（基于文本）两种动作协议

## 评估说明

- 需要安装 `pip install evalscope[sandbox]`（通过 ms-enclave 提供 Docker SDK）
- 需要 `scaleapi/SWE-bench_Pro-os` 仓库以获取每实例的运行脚本和 Dockerfile。默认情况下，该仓库会自动克隆至 `~/.cache/evalscope/swe_bench_pro/SWE-bench_Pro-os` 并固定到 commit `ca10a60`。若要使用已有克隆，请设置 `extra_params.swe_bench_pro_repo_path`。
- 智能体循环与每实例评估共享同一个沙箱配置（通过 `TaskConfig.sandbox.default_config` 直接传递给 ms_enclave 的 `DockerSandboxConfig`）。建议在此处设置 `memory_limit` / `cpu_limit`，以避免因内存不足导致测试被终止（例如 NodeBB 实例）；`platform` 默认为 `linux/amd64`，因此仅支持 amd64 的 sweap-images 可在 Apple Silicon 设备上开箱即用。

有关环境设置、参数配置及故障排查，请参阅[用户指南](https://evalscope.readthedocs.io/zh-cn/latest/third_party/swe_bench_pro.html)。

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `swe_bench_pro` |
| **数据集ID** | [ScaleAI/SWE-bench_Pro](https://modelscope.cn/datasets/ScaleAI/SWE-bench_Pro/summary) |
| **论文** | N/A |
| **标签** | `Coding` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 731 |
| 提示词长度（平均） | 1297.47 字符 |
| 提示词长度（最小/最大） | 419 / 8036 字符 |

## 样例示例

**子集**: `default`

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

## 提示模板

**提示模板：**
```text
{question}
```

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `swe_bench_pro_repo_path` | `str` | `` | 指向 `scaleapi/SWE-bench_Pro-os` 本地克隆路径。若为空，则自动克隆至 `~/.cache/evalscope/swe_bench_pro/SWE-bench_Pro-os` 并固定到 commit `ca10a60`。 |
| `dockerhub_username` | `str` | `jefzda` | 托管 sweap-images 仓库的 DockerHub 用户或组织名。 |
| `action_protocol` | `str` | `toolcall` | 智能体动作协议："toolcall"（函数调用）或 "backticks"（针对不支持函数调用模型的文本回退方案）。可选值：['toolcall', 'backticks'] |
| `max_steps` | `int` | `250` | 每个样本允许的最大智能体步数。 |
| `command_timeout` | `float` | `60.0` | 每条 bash 命令的默认超时时间（秒）。 |
| `eval_timeout` | `int` | `3600` | 每实例评估的超时时间（秒）。 |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets swe_bench_pro \
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
    datasets=['swe_bench_pro'],
    dataset_args={
        'swe_bench_pro': {
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```