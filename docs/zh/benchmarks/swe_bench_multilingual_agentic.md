# SWE-bench_Multilingual_Agentic

## 概述

SWE-bench Multilingual Agentic 是 SWE-bench Multilingual 的 agentic 模式评估，该基准测试包含 300 个任务，涵盖 42 个代码仓库和 9 种编程语言。模型在针对每个实例的 Docker 容器中，通过多轮 agent 循环自主探索、编辑并提交补丁。

## 任务描述

- **任务类型**：自动化软件工程 / Bug 修复（智能体模式，多语言）
- **输入**：GitHub issue 描述（不包含 oracle 文件上下文）
- **输出**：从 `git diff` 中收集的代码补丁（diff 格式），该补丁是在自主编辑后生成的
- **语言**：C, C++, Go, Java, JavaScript/TypeScript, PHP, Ruby 和 Rust

## 主要特点

- 300 个精心筛选的 Issue-Pull Request 任务
- 涵盖 9 种编程语言的 42 个真实世界代码仓库
- 多轮 agent 循环，每个实例配备独立的 SWE-bench Docker sandbox
- 采用 fail-to-pass 和 pass-to-pass 测试进行兼容 SWE-bench 的补丁评估

## 评估说明

- 通过 `TaskConfig.sandbox.engine='e2b'` 支持官方 E2B 和阿里云 Agent Sandbox，避免使用本地 Docker 和本地镜像下载。请参阅 [沙箱指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/sandbox.html#e2b)。
- 在评估前需要 `pip install swebench==4.1.0`
- 使用官方 SWE-bench Multilingual x86_64 实例镜像，并自动将 Docker 平台设置为 `linux/amd64`
- 为每个实例自动构建/拉取 Docker 镜像
- 每个实例的最终补丁验证超时时间为 1800 秒（30 分钟）
- 详细设置说明请参阅 [使用文档](https://evalscope.readthedocs.io/zh-cn/latest/third_party/swe_bench.html)
- 支持本地镜像构建和远程镜像拉取

## Agentic 模式

此基准测试在针对每个实例的 SWE-bench Docker 容器中运行一个多轮代理循环（镜像 mini-swe-agent 的 `swebench.yaml`）。模型发出 `bash` 命令以探索 `/testbed`、编辑源文件，并最终通过打印哨兵标记 `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` 后跟补丁内容来提交其 `git diff` 补丁。

默认的 `swe_bench_toolcall` 策略使用 OpenAI 函数调用，并包含一个单一的 `bash` 工具。不支持函数调用的模型可以通过 `NativeAgentConfig.strategy` 选择 `swe_bench_backticks`；该策略期望每轮对话中包含一个 ` ```mswea_bash_command ``` ` 代码块。

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `swe_bench_multilingual_agentic` |
| **数据集ID** | [SWE-bench/SWE-bench_Multilingual](https://modelscope.cn/datasets/SWE-bench/SWE-bench_Multilingual/summary) |
| **论文** | N/A |
| **标签** | `Coding` |
| **指标** | `accuracy` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |

## 数据统计

| 指标 | 数值 |
|--------|-------|
| 总样本数 | 300 |
| 提示词长度（平均值） | 2197.94 字符 |
| 提示词长度（最小/最大值） | 124 / 69351 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "72eec5c7",
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

*注：部分内容为显示目的已截断。*

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
| `dockerhub_username` | `str` | `swebench` | 远程 SWE-bench 镜像的 DockerHub 用户/组织命名空间。 |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets swe_bench_multilingual_agentic \
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
    datasets=['swe_bench_multilingual_agentic'],
    agent_config=NativeAgentConfig(
        strategy='swe_bench_toolcall',
        max_steps=250,
    ),
    dataset_args={
        'swe_bench_multilingual_agentic': {
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
