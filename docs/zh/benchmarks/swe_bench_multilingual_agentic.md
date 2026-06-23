# SWE-bench_Multilingual_Agentic


## 概述

SWE-bench Multilingual Agentic 是 SWE-bench Multilingual 的代理模式（agentic-mode）评估版本。该基准测试包含 300 个 SWE-bench 风格的任务，覆盖 42 个代码仓库和 9 种编程语言。模型在每个实例专属的 Docker 容器中，通过多轮代理循环自主探索、编辑代码并提交补丁。

## 任务描述

- **任务类型**：自动化软件工程 / 缺陷修复（代理式、多语言）
- **输入**：GitHub issue 描述（不含 oracle 文件上下文）
- **输出**：自主编辑后通过 `git diff` 收集的代码补丁（diff 格式）
- **支持语言**：C、C++、Go、Java、JavaScript/TypeScript、PHP、Ruby 和 Rust

## 主要特性

- 精心筛选的 300 个 Issue-Pull Request 任务
- 覆盖 9 种编程语言的 42 个真实开源仓库
- 每个实例使用独立的 SWE-bench Docker 沙箱环境进行多轮代理交互
- 使用 fail-to-pass 和 pass-to-pass 测试对补丁进行 SWE-bench 兼容性评估

## 评估说明

- 评估前需安装 `pip install swebench==4.1.0`
- 自动使用官方 SWE-bench Multilingual x86_64 实例镜像，并自动设置 Docker 平台为 `linux/amd64`
- 每个实例的 Docker 镜像会自动构建或拉取
- 每个实例最终补丁验证的超时时间为 1800 秒（30 分钟）
- 详细设置说明请参阅 [使用文档](https://evalscope.readthedocs.io/zh-cn/latest/third_party/swe_bench.html)
- 支持本地构建镜像和远程拉取镜像两种方式

## 代理模式

本基准测试在每个实例的 SWE-bench Docker 容器内驱动一个多轮代理循环（与 mini-swe-agent 的 `swebench.yaml` 配置一致）。模型通过发出 `bash` 命令探索 `/testbed` 目录、编辑源文件，并在完成任务时打印哨兵字符串 `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` 后附上补丁内容以提交最终 `git diff` 补丁。

`extra_params.action_protocol` 可选择以下两种协议：
- `toolcall`（默认）：使用 OpenAI 函数调用协议，仅提供一个 `bash` 工具。推荐用于支持工具调用的模型。
- `backticks`：基于文本的备用方案，每轮期望一个 ` ```mswea_bash_command ``` ` 代码块。适用于不支持函数调用的模型。

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `swe_bench_multilingual_agentic` |
| **数据集ID** | [SWE-bench/SWE-bench_Multilingual](https://modelscope.cn/datasets/SWE-bench/SWE-bench_Multilingual/summary) |
| **论文** | N/A |
| **标签** | `Coding` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 300 |
| 提示词长度（平均） | 2197.94 字符 |
| 提示词长度（最小/最大） | 124 / 69351 字符 |

## 样例示例

**子集**: `default`

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

## 提示模板

**提示模板：**
```text
{question}
```

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `action_protocol` | `str` | `toolcall` | 代理动作协议："toolcall"（主流 OpenAI 函数调用方式，与 mini-swe-agent swebench.yaml 一致）或 "backticks"（针对不支持函数调用模型的基于文本的 mswea_bash_command 回退方案）。可选值：['toolcall', 'backticks'] |
| `max_steps` | `int` | `250` | 每个样本的最大代理步数。 |
| `command_timeout` | `float` | `60.0` | 每个 bash 命令的默认超时时间（秒）。 |
| `build_docker_images` | `bool` | `True` | 为每个样本在本地构建 Docker 镜像。 |
| `pull_remote_images_if_available` | `bool` | `True` | 在构建前尝试拉取已存在的远程 Docker 镜像。 |
| `dockerhub_username` | `str` | `swebench` | 远程 SWE-bench 镜像在 DockerHub 上的用户/组织命名空间。 |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets swe_bench_multilingual_agentic \
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
    datasets=['swe_bench_multilingual_agentic'],
    dataset_args={
        'swe_bench_multilingual_agentic': {
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```