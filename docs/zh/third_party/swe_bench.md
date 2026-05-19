# SWE-bench

## 简介

[SWE-bench](https://www.swebench.com/) 是评估大型语言模型（LLMs）在真实软件工程任务上能力的基准测试套件，每个样本对应一个真实 GitHub 仓库的 Issue–PR 配对，模型需要产出可通过单元测试的代码补丁（patch）。

EvalScope 集成 SWE-bench 后提供两种评测模式：

- **Oracle 单轮模式**：将检索到的相关代码上下文一次性提供给模型，模型直接生成 patch
- **Agentic 多轮模式**：仅向模型提供 issue 描述，模型在每个实例独立的 SWE-bench Docker 容器中通过 `bash` 多轮探索、修改源码并提交 patch（兼容 [mini-swe-agent](https://github.com/princeton-nlp/mini-swe-agent) 流程）

## 两种模式对比

| 维度 | Oracle 单轮模式 | Agentic 多轮模式 |
|------|----------------|------------------|
| 输入信息 | `problem_statement` + 预先检索的代码上下文 | 仅 `problem_statement` |
| 上下文获取方式 | 由 `inference_dataset_id` 预先注入（oracle / BM25） | 模型自主在容器内 `bash` 探索 `/testbed` |
| 模型交互 | 单轮请求，直接输出 patch | 多轮 agent loop（默认上限 250 步） |
| 工具调用 | 无 | `bash` 工具，支持 `toolcall` / `backticks` 两种协议 |
| 提交机制 | 模型输出 diff 即视为提交 | 模型打印哨兵 `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` 后由系统抓取 `git diff` |
| Docker 使用阶段 | 仅评测阶段运行测试 | 推理 + 评测阶段都需要 |
| 评测数据集 | `swe_bench_verified` / `swe_bench_verified_mini` / `swe_bench_lite` | `swe_bench_verified_agentic` / `swe_bench_verified_mini_agentic` / `swe_bench_lite_agentic` |
| 评测代价 | 相对低，单轮推理 | 较高，多轮推理 + 长时间容器交互 |

**适用场景**

- **Oracle 模式**：适合快速建立基线、对比不同检索策略效果，或在算力/时间有限时进行低成本试跑
- **Agentic 模式**：适合评估模型作为「软件工程 agent」的端到端能力（自主探索代码、编写补丁、迭代调试），更贴近真实开发场景

## 数据集说明

### 核心评测数据集（两种模式共用名字根）

下列数据集通过 Issue–PR 配对与单元测试验证评估模型，从全量到精选满足不同评估需求：

- **SWE-bench Verified（`swe_bench_verified`）**：从 SWE-bench 测试集中精选的 500 个人工验证样本，质量经过严格把关
- **SWE-bench Verified Mini（`swe_bench_verified_mini`）**：50 样本的轻量子集，存储需求约 5GB（vs 原版 130GB），保持与原集相同的难度分布
- **SWE-bench Lite（`swe_bench_lite`）**：来自 11 个流行 Python 项目的 300 个 Issue–PR 配对

### Oracle 模式专属：推理数据集（`inference_dataset_id`）

不同检索方式格式化的 SWE-bench 版本，差异在于代码上下文的检索方式与上限：

- **`princeton-nlp/SWE-bench_oracle`**：使用 Oracle 检索（理想化检索结果，用作上限基准），**默认值**
- **`princeton-nlp/SWE-bench_bm25_13K`**：BM25 检索，代码上下文限制 13,000 tokens（cl100k_base）
- **`princeton-nlp/SWE-bench_bm25_27K`**：BM25 检索，限制 27,000 tokens
- **`princeton-nlp/SWE-bench_bm25_40K`**：BM25 检索，限制 40,000 tokens

### Agentic 模式专属：评测数据集

在核心数据集名字后加 `_agentic` 后缀触发 agentic 多轮评测：

- **`swe_bench_verified_agentic`** — SWE-bench Verified（500 样本）
- **`swe_bench_verified_mini_agentic`** — SWE-bench Verified Mini（50 样本，约 5GB）
- **`swe_bench_lite_agentic`** — SWE-bench Lite（300 样本）

## 安装依赖

SWE-bench 使用 Docker 来确保评估的可复现性。

1. **安装 Docker**：参考 [Docker 安装指南](https://docs.docker.com/engine/install/) 在你的机器上安装 Docker
2. **Linux 用户额外配置**：建议查看 [安装后配置步骤](https://docs.docker.com/engine/install/linux-postinstall/) 以获得更好的体验
3. 安装 Python 依赖：

```bash
pip install evalscope
pip install swebench==4.1.0
```

**提示**：正确配置 Docker 是运行 SWE-bench 评估的前提条件，请确保安装完成后 Docker 服务正常运行。

```{note}
首次运行 swe_bench 任务时，系统需要构建/下载必要的 Docker 镜像，资源消耗较高：

- **时间消耗**：完整 SWE-bench Verified 数据集的镜像构建可能耗时数小时
- **存储需求**：完整集需预留约 130GB 存储；Mini 集约 5GB

**建议**：首次运行请确保磁盘空间和时间预算充足，并选择网络状况良好的环境。
```

## Oracle 单轮评测模式

下面以 `qwen-plus` 模型评测 `swe_bench_verified` 为例：

```python
import os
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='qwen-plus',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    eval_type='openai_api',  # 使用 API 模型服务
    datasets=['swe_bench_verified'],  # 也可选择 'swe_bench_verified_mini' 或 'swe_bench_lite'
    dataset_args={
        'swe_bench_verified': {
            'extra_params': {
                'build_docker_images': True,                              # 是否构建评测所需的 Docker 镜像，首次运行建议 True
                'pull_remote_images_if_available': True,                  # 远程有可用镜像时优先拉取，建议 True
                'inference_dataset_id': 'princeton-nlp/SWE-bench_oracle'  # 推理数据集，可选 BM25_13K / 27K / 40K / oracle
            }
        }
    },
    eval_batch_size=5,  # 推理批大小 / 并行评测任务的工作线程数（docker 容器数量）
    limit=5,            # 限制评测数量便于快速测试，正式评测时建议去掉
    generation_config={
        'temperature': 0.1,
    }
)
run_task(task_cfg=task_cfg)
```

评测中间结果保存在 `outputs/xxxxx/swebench_log` 目录下，包含 `patch.diff` 等文件。最终结果示例：

```text
+-----------+--------------------+----------+----------+-------+---------+---------+
| Model     | Dataset            | Metric   | Subset   |   Num |   Score | Cat.0   |
+===========+====================+==========+==========+=======+=========+=========+
| qwen-plus | swe_bench_verified | mean_acc | default  |     5 |     0.2 | default |
+-----------+--------------------+----------+----------+-------+---------+---------+
```

## Agentic 多轮评测模式

在 agentic 模式下，模型只接收原始 `problem_statement`（不提供 oracle 上下文），并在每个实例独立的 SWE-bench Docker 容器中驱动多轮 agent 循环：使用 `bash` 探索 `/testbed`、编辑源文件，最终通过打印哨兵字符串 `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` 提交 `git diff`。

### 动作协议

`action_protocol` 参数控制模型与 bash 的交互方式：

| 协议 | 描述 | 适用模型 |
|------|------|----------|
| `toolcall`（默认） | OpenAI function-calling，注册单个 `bash` 工具，支持并行工具调用 | 支持 function-calling 的模型 |
| `backticks` | 模型将命令包裹在 `` ```mswea_bash_command ... ``` `` 围栏块中，每轮一条命令 | 不支持 function-calling 的模型 |

### 运行示例

下面示例完全对齐 [tests/benchmark/test_agent.py](https://github.com/modelscope/evalscope/blob/main/tests/benchmark/test_agent.py) 中的 `test_swe_bench_verified_mini_agentic` 用例：

```python
import os
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='qwen3-max',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    eval_type='openai_api',
    datasets=['swe_bench_verified_mini_agentic'],  # 也可选 'swe_bench_verified_agentic' 或 'swe_bench_lite_agentic'
    dataset_args={
        'swe_bench_verified_mini_agentic': {
            'extra_params': {
                'action_protocol': 'toolcall',           # 动作协议：'toolcall' 或 'backticks'
                'max_steps': 250,                        # agent 循环最大步数
                'command_timeout': 60.0,                 # 单条 bash 命令的超时（秒）
                'build_docker_images': True,             # 准备评测所需镜像，首次运行建议 True
                'pull_remote_images_if_available': True, # 远程有可用镜像时优先拉取
                # 'force_arch': 'arm64',                 # Apple Silicon 用户可显式指定为 'arm64'，默认 '' 自动检测
            }
        }
    },
    eval_batch_size=5,  # 并行容器数量
    limit=3,            # 限制评测数量便于快速测试，正式评测时建议去掉
    generation_config={
        'temperature': 0.7,
        'parallel_tool_calls': True,
        'stream': True,
    }
)
run_task(task_cfg=task_cfg)
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `action_protocol` | str | `toolcall` | 模型与 bash 的交互协议，可选 `toolcall` / `backticks` |
| `max_steps` | int | `250` | 每个样本 agent 循环的最大步数 |
| `command_timeout` | float | `60.0` | 单条 bash 命令超时秒数 |
| `build_docker_images` | bool | `True` | 是否准备 Docker 镜像（构建或拉取） |
| `pull_remote_images_if_available` | bool | `True` | 是否优先从远程仓库拉取已构建好的镜像 |
| `force_arch` | str | `''` | 强制指定镜像架构，可选 `''` / `arm64` / `x86_64`，留空自动检测 |

```{note}
Agentic 模式在**推理和评测两个阶段**都需要 Docker 运行。每个样本会从预构建的 SWE-bench 镜像启动独立容器，因此 `eval_batch_size` 实际上控制了同时存在的容器数量，请根据机器资源酌情调整。
```
