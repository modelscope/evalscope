# SWE-bench_Verified_mini


## Overview

SWE-bench Verified Mini is a compact subset of SWE-bench Verified, containing 50 carefully selected samples that maintain the same distribution of performance, test pass rates, and difficulty as the full dataset while requiring only 5GB of storage instead of 130GB.

## Task Description

- **Task Type**: Automated Software Engineering / Bug Fixing
- **Input**: GitHub issue description with repository context
- **Output**: Code patch (diff format) that resolves the issue
- **Size**: 50 samples (vs 500 in full Verified set)

## Key Features

- Representative 50-sample subset of SWE-bench Verified
- Same difficulty distribution as full dataset
- Dramatically reduced storage requirements (5GB vs 130GB)
- Ideal for quick evaluation and development iteration
- Maintains statistical validity for benchmarking

## Evaluation Notes

- Supports official E2B and Aliyun Agent Sandbox through `TaskConfig.sandbox.engine='e2b'`, avoiding local Docker and local image downloads. See the [sandbox guide](https://evalscope.readthedocs.io/en/latest/user_guides/sandbox.html#e2b).
- Requires `pip install swebench==4.1.0` before evaluation
- Docker images are built/pulled automatically
- See the [usage documentation](https://evalscope.readthedocs.io/en/latest/third_party/swe_bench.html) for detailed setup
- Good for rapid prototyping and initial model assessment


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `swe_bench_verified_mini` |
| **Dataset ID** | [evalscope/swe-bench-verified-mini](https://modelscope.cn/datasets/evalscope/swe-bench-verified-mini/summary) |
| **Paper** | N/A |
| **Tags** | `Coding` |
| **Metrics** | `accuracy` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 50 |
| Prompt Length (Mean) | 69595.46 chars |
| Prompt Length (Min/Max) | 7182 / 500669 chars |

## Sample Example

**Subset**: `default`

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

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
{question}
```

## Extra Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `build_docker_images` | `bool` | `True` | Build Docker images locally for each sample. |
| `pull_remote_images_if_available` | `bool` | `True` | Attempt to pull existing remote Docker images before building. |
| `inference_dataset_id` | `str` | `princeton-nlp/SWE-bench_oracle` | Oracle dataset ID used to fetch inference context from the selected dataset hub. |
| `inference_dataset_revision` | `str | null` | `None` | Optional revision of the oracle inference dataset. |
| `force_arch` | `str` | `` | Optionally force the docker images to be pulled/built for a specific architecture. Choices: ['', 'arm64', 'x86_64'] |
| `dockerhub_username` | `str` | `swebench` | DockerHub user/org namespace for remote SWE-bench images. |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets swe_bench_verified_mini \
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
    datasets=['swe_bench_verified_mini'],
    dataset_args={
        'swe_bench_verified_mini': {
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
