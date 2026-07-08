# SWE-bench_Multilingual_Agentic


## Overview

SWE-bench Multilingual Agentic is the agentic-mode evaluation of SWE-bench Multilingual, a 300-task SWE-bench-style benchmark spanning 42 repositories and 9 programming languages. The model autonomously explores, edits, and submits a patch through a multi-turn agent loop inside a per-instance Docker container.

## Task Description

- **Task Type**: Automated Software Engineering / Bug Fixing (Agentic, Multilingual)
- **Input**: GitHub issue description (no oracle file context)
- **Output**: Code patch (diff format) collected from `git diff` after autonomous editing
- **Languages**: C, C++, Go, Java, JavaScript/TypeScript, PHP, Ruby, and Rust

## Key Features

- 300 curated Issue-Pull Request tasks
- 42 real-world repositories across 9 programming languages
- Multi-turn agent loop with per-instance SWE-bench Docker sandbox
- SWE-bench-compatible patch evaluation using fail-to-pass and pass-to-pass tests

## Evaluation Notes

- Requires `pip install swebench==4.1.0` before evaluation
- Uses the official SWE-bench Multilingual x86_64 instance images and sets Docker platform to `linux/amd64` automatically
- Docker images are built/pulled automatically for each instance
- Timeout of 1800 seconds (30 min) per instance for final patch validation
- See the [usage documentation](https://evalscope.readthedocs.io/en/latest/third_party/swe_bench.html) for detailed setup instructions
- Supports both local image building and remote image pulling

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
| **Benchmark Name** | `swe_bench_multilingual_agentic` |
| **Dataset ID** | [SWE-bench/SWE-bench_Multilingual](https://modelscope.cn/datasets/SWE-bench/SWE-bench_Multilingual/summary) |
| **Paper** | N/A |
| **Tags** | `Coding` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 300 |
| Prompt Length (Mean) | 2197.94 chars |
| Prompt Length (Min/Max) | 124 / 69351 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "52ece3cf",
      "content": "Support Post aggregation function pow(f1,f2) to cater for square, cube , square root.\n### Description\r\n\r\nPlease describe the feature or change with as much detail as possible. \r\n\r\nAs of now the only supported arithmetic functions are +, -, *, ... [TRUNCATED 284 chars] ... \r\n\r\nThe proposal is to add a `pow` function which enables all the about usecase .  Square of a number can be represent by pow(f1,2) , Cube can be represented as power(f1 ,3)  , Squar root of a number can be represented by power(f1,0.5) ,\r\n\r\n\n"
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
    "problem_statement": "Support Post aggregation function pow(f1,f2) to cater for square, cube , square root.\n### Description\r\n\r\nPlease describe the feature or change with as much detail as possible. \r\n\r\nAs of now the only supported arithmetic functions are +, -, *, ... [TRUNCATED 284 chars] ... \r\n\r\nThe proposal is to add a `pow` function which enables all the about usecase .  Square of a number can be represent by pow(f1,2) , Cube can be represented as power(f1 ,3)  , Squar root of a number can be represented by power(f1,0.5) ,\r\n\r\n\n",
    "instance_id": "apache__druid-13704",
    "base_commit": "51dfde02840017092486fb75be2b16566aff6a19",
    "patch": "diff --git a/docs/querying/post-aggregations.md b/docs/querying/post-aggregations.md\nindex c75d122eb20a..935ca8fbce16 100644\n--- a/docs/querying/post-aggregations.md\n+++ b/docs/querying/post-aggregations.md\n@@ -36,7 +36,7 @@ There are several ... [TRUNCATED 1110 chars] ...      }\n+    },\n+\n+    POW(\"pow\") {\n+      @Override\n+      public double compute(double lhs, double rhs)\n+      {\n+        return Math.pow(lhs, rhs);\n+      }\n     };\n \n     private static final Map<String, Ops> LOOKUP_MAP = new HashMap<>();\n",
    "PASS_TO_PASS": [
      "org.apache.druid.query.aggregation.post.ArithmeticPostAggregatorTest#testDiv",
      "org.apache.druid.query.aggregation.post.ArithmeticPostAggregatorTest#testQuotient"
    ],
    "FAIL_TO_PASS": [
      "org.apache.druid.query.aggregation.post.ArithmeticPostAggregatorTest#testPow"
    ],
    "test_patch": "diff --git a/processing/src/test/java/org/apache/druid/query/aggregation/post/ArithmeticPostAggregatorTest.java b/processing/src/test/java/org/apache/druid/query/aggregation/post/ArithmeticPostAggregatorTest.java\nindex a93034427539..7e1d4d112 ... [TRUNCATED 1358 chars] ... s(1.0, agg.compute(ImmutableMap.of(\"value\", 1)));\n+    Assert.assertEquals(1.0, agg.compute(ImmutableMap.of(\"value\", -1)));\n+    Assert.assertEquals(1.0, agg.compute(ImmutableMap.of(\"value\", .5)));\n+  }\n   @Test\n   public void testDiv()\n   {\n",
    "version": "13704",
    "repo": "apache/druid",
    "environment_setup_commit": null,
    "hints_text": "",
    "created_at": "2023-01-23 04:10:47",
    "docker_image": "swebench/sweb.eval.x86_64.apache_1776_druid-13704:latest"
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
    --datasets swe_bench_multilingual_agentic \
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
    datasets=['swe_bench_multilingual_agentic'],
    dataset_args={
        'swe_bench_multilingual_agentic': {
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
