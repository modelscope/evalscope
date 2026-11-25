# SWE-bench_Pro

## 概述

SWE-bench_Pro 是由 Scale AI 推出的一个高难度基准测试，用于评估大语言模型（LLM）/智能体（Agent）在跨多种编程语言的长周期软件工程任务中的表现。给定一个代码库和一个问题描述，模型必须自主探索仓库、编辑源文件，并在每个实例独立的 Docker 容器内通过多轮智能体循环提交补丁。

## 任务描述

- **任务类型**：自动化软件工程 / Bug 修复（智能体式）
- **输入**：GitHub issue 描述
- **输出**：代码补丁（diff 格式），在自主编辑后收集
- **语言**：多种（`repo_language` 字段；例如 JavaScript/TypeScript、Python、Go）

## 主要特点

- 多轮代理循环，每个实例使用独立的 DockerHub 镜像 (`jefzda/sweap-images:{tag}`)
- 基于哨兵标记的补丁提交协议 (`COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`)
- 容器端评估流程：应用 `git apply` 补丁，运行实例的 `run_script.sh`，使用 `parser.py` 进行解析，然后检查 `(fail_to_pass | pass_to_pass) ⊆ PASSED`
- 支持两种动作协议：`toolcall`（函数调用）和 `backticks`（基于文本）

## 评估说明

- 通过 `TaskConfig.sandbox.engine='e2b'` 支持官方 E2B 和阿里云 Agent Sandbox，避免使用本地 Docker 和本地镜像下载。请参阅 [沙箱指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/sandbox.html#e2b)。
- 需要 `pip install evalscope[sandbox]`（通过 ms-enclave 提供沙箱运行时）
- 需要 `scaleapi/SWE-bench_Pro-os` 仓库以获取每个实例的运行脚本和 Dockerfiles。默认情况下，该仓库会自动克隆到 `~/.cache/evalscope/swe_bench_pro/SWE-bench_Pro-os` 并固定到 commit `ca10a60`。若要使用现有的克隆副本，请设置 `extra_params.swe_bench_pro_repo_path`。
- agent loop 和每个实例的评估通过 `TaskConfig.sandbox.default_config` 共享单个沙箱配置，并将其传递给所选引擎的 ms-enclave。对于 Docker，请在此处设置 `memory_limit` / `cpu_limit` 以避免测试运行被 OOM-Killed（例如 NodeBB）；`platform` 默认为 `linux/amd64`，因此仅限 amd64 的 sweap-images 可以在 Apple Silicon 上开箱即用。

有关设置、参数和故障排除，请参阅 [user guide](https://evalscope.readthedocs.io/zh-cn/latest/third_party/swe_bench_pro.html)。

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `swe_bench_pro` |
| **数据集ID** | [ScaleAI/SWE-bench_Pro](https://modelscope.cn/datasets/ScaleAI/SWE-bench_Pro/summary) |
| **论文** | N/A |
| **标签** | `Coding` |
| **指标** | `accuracy` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |

## 数据统计

| 指标 | 数值 |
|--------|-------|
| 总样本数 | 642 |
| 提示词长度（平均值） | 1013.12 字符 |
| 提示词长度（最小/最大） | 191 / 8010 字符 |

## 样例示例

**子集**: `default`

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

*注意：为便于显示，部分内容已截断。*

## 提示模板

**提示模板：**
```text
{question}
```

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `swe_bench_pro_repo_path` | `str` | `` | scaleapi/SWE-bench_Pro-os 克隆的本地路径。如果为空，则自动克隆到 ~/.cache/evalscope/swe_bench_pro/SWE-bench_Pro-os 并固定到提交 ca10a60。 |
| `dockerhub_username` | `str` | `jefzda` | 托管 sweap-images 仓库的 DockerHub 用户/组织。 |
| `eval_timeout` | `int` | `3600` | 每个实例评估的超时时间（秒）。 |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets swe_bench_pro \
    --agent-config '{"mode":"native","strategy":"swe_bench_toolcall","max_steps":250}' \
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
    datasets=['swe_bench_pro'],
    agent_config=NativeAgentConfig(
        strategy='swe_bench_toolcall',
        max_steps=250,
    ),
    dataset_args={
        'swe_bench_pro': {
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
