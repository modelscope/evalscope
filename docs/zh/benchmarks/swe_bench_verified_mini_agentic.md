# SWE-bench_Verified_Mini_Agentic

## 概述

SWE-bench Verified Mini Agentic 是 SWE-bench Verified Mini 的 agentic 模式评估。SWE-bench Verified Mini 是一个包含 50 个样本的精简子集，它在保持与完整 Verified 数据集相同的性能分布、测试通过率和难度水平的同时，仅需 5GB 存储空间（而非 130GB）。模型必须通过多轮 agent 循环自主探索、编辑并提交补丁。

## 任务描述

- **任务类型**：自动化软件工程 / Bug 修复（智能体式）
- **输入**：GitHub issue 描述（无 oracle 文件上下文）
- **输出**：从 `git diff` 中收集的代码补丁（diff 格式），通过自主编辑生成
- **规模**：50 个样本（完整 Verified 数据集为 500 个样本）

## 主要特点

- SWE-bench Verified 的代表性 50 样本子集
- 与完整数据集相同的难度分布
- 显著降低存储需求（5GB vs 130GB）
- 支持多轮代理循环，每个实例配备独立的 Docker 沙箱
- 适用于快速代理评估和开发迭代

## 评估说明

- 通过 `TaskConfig.sandbox.engine='e2b'` 支持官方 E2B 和阿里云 Agent Sandbox，避免使用本地 Docker 和本地镜像下载。请参阅 [沙箱指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/sandbox.html#e2b)。
- 在评估前需要 `pip install swebench==4.1.0`
- Docker 镜像会自动构建/拉取
- 详细设置请参阅 [使用文档](https://evalscope.readthedocs.io/zh-cn/latest/third_party/swe_bench.html)
- 适用于快速原型开发 agent 策略和初步模型评估

## Agentic 模式

该基准测试在针对每个实例的 SWE-bench Docker 容器内驱动一个多轮代理循环（镜像 mini-swe-agent 的 `swebench.yaml`）。模型发出 `bash` 命令以探索 `/testbed`、编辑源文件，并最终通过打印哨兵标记 `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` 及其后的补丁内容来提交其 `git diff` 补丁。

默认的 `swe_bench_toolcall` 策略使用 OpenAI 函数调用，并配备单个 `bash` 工具。不支持函数调用的模型可以通过 `NativeAgentConfig.strategy` 选择 `swe_bench_backticks`；该策略期望每轮包含一个 ` ```mswea_bash_command ``` ` 代码块。

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `swe_bench_verified_mini_agentic` |
| **数据集ID** | [evalscope/swe-bench-verified-mini](https://modelscope.cn/datasets/evalscope/swe-bench-verified-mini/summary) |
| **论文** | N/A |
| **标签** | `Coding` |
| **指标** | `accuracy` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |

## 数据统计

| 指标 | 数值 |
|--------|-------|
| 总样本数 | 50 |
| 提示词长度（平均值） | 1268.5 字符 |
| 提示词长度（最小/最大值） | 257 / 5362 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "1150d5f7",
      "content": "AuthenticationForm's username field doesn't set maxlength HTML attribute.\nDescription\n\t\nAuthenticationForm's username field doesn't render with maxlength HTML attribute anymore.\nRegression introduced in #27515 and 5ceaf14686ce626404afb6a5fbd3d8286410bf13.\n​https://groups.google.com/forum/?utm_source=digest&utm_medium=email#!topic/django-developers/qnfSqro0DlA\n​https://forum.djangoproject.com/t/possible-authenticationform-max-length-regression-in-django-2-1/241\n"
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
    "problem_statement": "AuthenticationForm's username field doesn't set maxlength HTML attribute.\nDescription\n\t\nAuthenticationForm's username field doesn't render with maxlength HTML attribute anymore.\nRegression introduced in #27515 and 5ceaf14686ce626404afb6a5fbd3d8286410bf13.\n​https://groups.google.com/forum/?utm_source=digest&utm_medium=email#!topic/django-developers/qnfSqro0DlA\n​https://forum.djangoproject.com/t/possible-authenticationform-max-length-regression-in-django-2-1/241\n",
    "instance_id": "django__django-11790",
    "base_commit": "b1d6b35e146aea83b171c1b921178bbaae2795ed",
    "patch": "diff --git a/django/contrib/auth/forms.py b/django/contrib/auth/forms.py\n--- a/django/contrib/auth/forms.py\n+++ b/django/contrib/auth/forms.py\n@@ -191,7 +191,9 @@ def __init__(self, request=None, *args, **kwargs):\n \n         # Set the max len ... [TRUNCATED 322 chars] ... username_max_length\n+        self.fields['username'].widget.attrs['maxlength'] = username_max_length\n         if self.fields['username'].label is None:\n             self.fields['username'].label = capfirst(self.username_field.verbose_name)\n \n",
    "PASS_TO_PASS": [
      "test_html_autocomplete_attributes (auth_tests.test_forms.AdminPasswordChangeFormTest)",
      "test_missing_passwords (auth_tests.test_forms.AdminPasswordChangeFormTest)",
      "test_non_matching_passwords (auth_tests.test_forms.AdminPasswordChangeFormTest)",
      "test_one_password (auth_tests.test_forms.AdminPasswordChangeFormTest)",
      "test_password_whitespace_not_stripped (auth_tests.test_forms.AdminPasswordChangeFormTest)",
      "test_success (auth_tests.test_forms.AdminPasswordChangeFormTest)",
      "test_field_order (auth_tests.test_forms.PasswordChangeFormTest)",
      "test_html_autocomplete_attributes (auth_tests.test_forms.PasswordChangeFormTest)",
      "test_incorrect_password (auth_tests.test_forms.PasswordChangeFormTest)",
      "test_password_verification (auth_tests.test_forms.PasswordChangeFormTest)",
      "... [TRUNCATED 67 more items] ..."
    ],
    "FAIL_TO_PASS": [
      "test_username_field_max_length_defaults_to_254 (auth_tests.test_forms.AuthenticationFormTest)",
      "test_username_field_max_length_matches_user_model (auth_tests.test_forms.AuthenticationFormTest)"
    ],
    "test_patch": "diff --git a/tests/auth_tests/test_forms.py b/tests/auth_tests/test_forms.py\n--- a/tests/auth_tests/test_forms.py\n+++ b/tests/auth_tests/test_forms.py\n@@ -423,6 +423,7 @@ def test_username_field_max_length_matches_user_model(self):\n         C ... [TRUNCATED 543 chars] ... )\n         self.assertEqual(form.fields['username'].max_length, 254)\n+        self.assertEqual(form.fields['username'].widget.attrs.get('maxlength'), 254)\n         self.assertEqual(form.errors, {})\n \n     def test_username_field_label(self):\n",
    "version": "3.1",
    "repo": "django/django",
    "environment_setup_commit": "0668164b4ac93a5be79f5b87fae83c657124d9ab",
    "hints_text": "Regression test.",
    "created_at": "2019-09-17T14:33:44Z",
    "docker_image": "swebench/sweb.eval.arm64.django_1776_django-11790:latest"
  }
}
```

*注：为便于显示，部分内容已截断。*

## 提示模板

**提示模板：**
```text
{question}
```

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `build_docker_images` | `bool` | `True` | 为每个样本在本地构建 Docker 镜像。 |
| `pull_remote_images_if_available` | `bool` | `True` | 在构建之前尝试拉取现有的远程 Docker 镜像。 |
| `force_arch` | `str` | `` | 可选地强制指定用于镜像构建/拉取的架构。选项：['', 'arm64', 'x86_64'] |
| `dockerhub_username` | `str` | `swebench` | 用于远程 SWE-bench 镜像的 DockerHub 用户/组织命名空间。 |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets swe_bench_verified_mini_agentic \
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
    datasets=['swe_bench_verified_mini_agentic'],
    agent_config=NativeAgentConfig(
        strategy='swe_bench_toolcall',
        max_steps=250,
    ),
    dataset_args={
        'swe_bench_verified_mini_agentic': {
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
