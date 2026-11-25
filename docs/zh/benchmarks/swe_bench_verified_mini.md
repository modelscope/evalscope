# SWE-bench_Verified_mini

## 概述

SWE-bench Verified Mini 是 SWE-bench Verified 的一个精简子集，包含 50 个精心挑选的样本。它在保持与完整数据集相同的性能分布、测试通过率和难度水平的同时，仅需 5GB 存储空间，而非 130GB。

## 任务描述

- **任务类型**：自动化软件工程 / Bug 修复
- **输入**：包含仓库上下文的 GitHub issue 描述
- **输出**：解决该问题的代码补丁（diff 格式）
- **规模**：50 个样本（完整 Verified 数据集为 500 个样本）

## 主要特点

- SWE-bench Verified 的代表性 50 样本子集
- 与完整数据集具有相同的难度分布
- 显著降低存储需求（5GB vs 130GB）
- 适用于快速评估和开发迭代
- 保持基准测试的统计有效性

## 评估说明

- 通过 `TaskConfig.sandbox.engine='e2b'` 支持官方 E2B 和阿里云 Agent Sandbox，避免使用本地 Docker 和本地镜像下载。请参阅 [沙箱指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/sandbox.html#e2b)。
- 评估前需要 `pip install swebench==4.1.0`
- Docker 镜像会自动构建/拉取
- 详细设置请参阅 [使用文档](https://evalscope.readthedocs.io/zh-cn/latest/third_party/swe_bench.html)
- 适用于快速原型开发和初步模型评估

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `swe_bench_verified_mini` |
| **数据集ID** | [evalscope/swe-bench-verified-mini](https://modelscope.cn/datasets/evalscope/swe-bench-verified-mini/summary) |
| **论文** | N/A |
| **标签** | `Coding` |
| **指标** | `accuracy` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 50 |
| 提示词长度（平均值） | 69595.46 字符 |
| 提示词长度（最小/最大值） | 7182 / 500669 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "f28996e6",
      "content": "You will be provided with a partial code base and an issue statement explaining a problem to resolve.\n<issue>\nAuthenticationForm's username field doesn't set maxlength HTML attribute.\nDescription\n\t\nAuthenticationForm's username field doesn't  ... [TRUNCATED 20925 chars] ... 2.0\n+        while y != y1:\n+            points.append((x, y))\n+            err -= dx\n+            if err < 0:\n+                x += sx\n+                err += dy\n+            y += sy\n \n+    points.append((x, y))\n     return points\n</patch>\n\n"
    }
  ],
  "id": 0,
  "group_id": 0,
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

*注意：部分内容为显示目的已截断。*

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
| `inference_dataset_id` | `str` | `princeton-nlp/SWE-bench_oracle` | 用于从选定的数据集中心获取推理上下文的 Oracle 数据集 ID。 |
| `inference_dataset_revision` | `str | null` | `None` | Oracle 推理数据集的可选修订版本。 |
| `force_arch` | `str` | `` | 可选地，强制针对特定架构拉取/构建 Docker 镜像。选项：['', 'arm64', 'x86_64'] |
| `dockerhub_username` | `str` | `swebench` | 远程 SWE-bench 镜像的 DockerHub 用户/组织命名空间。 |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets swe_bench_verified_mini \
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
    datasets=['swe_bench_verified_mini'],
    dataset_args={
        'swe_bench_verified_mini': {
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
