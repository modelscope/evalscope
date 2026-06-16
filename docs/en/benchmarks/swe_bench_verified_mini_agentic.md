# SWE-bench_Verified_Mini_Agentic


## Overview

SWE-bench Verified Mini Agentic is the agentic-mode evaluation of SWE-bench Verified Mini, a compact 50-sample subset that maintains the same distribution of performance, test pass rates, and difficulty as the full Verified set while requiring only 5GB of storage instead of 130GB. The model must autonomously explore, edit, and submit a patch through a multi-turn agent loop.

## Task Description

- **Task Type**: Automated Software Engineering / Bug Fixing (Agentic)
- **Input**: GitHub issue description (no oracle file context)
- **Output**: Code patch (diff format) collected from `git diff` after autonomous editing
- **Size**: 50 samples (vs 500 in full Verified set)

## Key Features

- Representative 50-sample subset of SWE-bench Verified
- Same difficulty distribution as the full dataset
- Dramatically reduced storage requirements (5GB vs 130GB)
- Multi-turn agent loop with per-instance Docker sandbox
- Ideal for quick agentic evaluation and development iteration

## Evaluation Notes

- Requires `pip install swebench==4.1.0` before evaluation
- Docker images are built/pulled automatically
- See the [usage documentation](https://evalscope.readthedocs.io/en/latest/third_party/swe_bench.html) for detailed setup
- Good for rapid prototyping of agent strategies and initial model assessment

## Agentic Mode

This benchmark drives a multi-turn agent loop (mirrors mini-swe-agent's
`swebench.yaml`) inside a per-instance SWE-bench Docker container. The
model issues `bash` commands to explore `/testbed`, edit source files,
and finally submits its `git diff` patch by printing the sentinel
`COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` followed by the patch contents.

`extra_params.action_protocol` selects between:
- `toolcall` (default): OpenAI function-calling protocol with a single
  `bash` tool. Recommended for any model that supports tool calling.
- `backticks`: text-based fallback expecting one
  ` ```mswea_bash_command ``` ` block per turn. For models without
  function-calling support.


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `swe_bench_verified_mini_agentic` |
| **Dataset ID** | [evalscope/swe-bench-verified-mini](https://modelscope.cn/datasets/evalscope/swe-bench-verified-mini/summary) |
| **Paper** | N/A |
| **Tags** | `Coding` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 50 |
| Prompt Length (Mean) | 1268.5 chars |
| Prompt Length (Min/Max) | 257 / 5362 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "363bb967",
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

## Prompt Template

**Prompt Template:**
```text
{question}
```

## Extra Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `action_protocol` | `str` | `toolcall` | Agent action protocol: "toolcall" (mainline OpenAI function-calling, mirrors mini-swe-agent swebench.yaml) or "backticks" (textbased mswea_bash_command fallback for models without function-calling support). Choices: ['toolcall', 'backticks'] |
| `max_steps` | `int` | `250` | Maximum number of agent steps per sample. |
| `command_timeout` | `float` | `60.0` | Default per-bash-command timeout in seconds. |
| `build_docker_images` | `bool` | `True` | Build Docker images locally for each sample. |
| `pull_remote_images_if_available` | `bool` | `True` | Attempt to pull existing remote Docker images before building. |
| `force_arch` | `str` | `` | Optionally force a specific architecture for image build/pull. Choices: ['', 'arm64', 'x86_64'] |
| `dockerhub_username` | `str` | `swebench` | DockerHub user/org namespace for remote SWE-bench images. |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets swe_bench_verified_mini_agentic \
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
    datasets=['swe_bench_verified_mini_agentic'],
    dataset_args={
        'swe_bench_verified_mini_agentic': {
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


