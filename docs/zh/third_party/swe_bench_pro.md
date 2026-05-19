# SWE-bench_Pro

## 简介

[SWE-bench_Pro](https://github.com/scaleapi/SWE-bench_Pro-os) 是 Scale AI 在 SWE-bench 之上构建的更具挑战性的基准测试，用于评估大语言模型（LLM）/Agent 在**长周期、多语言、真实代码库**软件工程任务中的能力。给定一个代码库和一个 issue 描述，模型需要在每个实例独立的 Docker 容器中通过多轮 agent 循环自主探索代码、编辑源文件并提交可通过隐藏单元测试的补丁。

```{tip}
**强烈推荐使用 SWE-bench_Pro 而非原始 SWE-bench。** 与 `swe_bench_verified` / `swe_bench_lite` 相比：

- **更难、未被充分污染**：样本来自更新、商业化或非主流仓库，而非容易在训练数据中泄漏的 Django/Flask 等热门 Python 项目
- **多语言**：覆盖 Python、JavaScript/TypeScript、Go 等多种语言（`repo_language` 字段），更全面地反映真实工程能力
- **更长的任务窗**：每个实例平均涉及更多文件、更长的修改路径，能更好区分前沿模型
- **官方维护的 Docker 镜像**：每个实例的预构建镜像直接从 DockerHub 拉取（`jefzda/sweap-images:{tag}`），**无需本地构建**，相比原始 SWE-bench 动辄数小时的镜像构建过程节省大量时间

原始 `swe_bench_*` 系列建议仅用于历史对比或快速冒烟。
```

## 任务描述

- **任务类型**：自动化软件工程 / 缺陷修复（Agentic）
- **输入**：GitHub issue 描述（`problem_statement`）
- **输出**：经过自主编辑后收集到的代码补丁（unified diff 格式）
- **编程语言**：多语言（`repo_language` 字段；如 JavaScript/TypeScript、Python、Go 等）

## 关键特性

- 多轮 agent 循环，每个实例使用独立的 DockerHub 镜像（`jefzda/sweap-images:{tag}`）
- 哨兵字符串提交协议（`COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`）
- 容器内评测：`git apply` 补丁 → 执行实例专属的 `run_script.sh` → 通过 `parser.py` 解析输出 → 校验 `(fail_to_pass | pass_to_pass) ⊆ PASSED`
- 同时支持 `toolcall`（函数调用）与 `backticks`（基于文本，适配无 function-calling 能力的模型）两种动作协议
- **推理阶段与评测阶段共用同一份 sandbox 配置**：`memory_limit` / `cpu_limit` / `platform` 等只需配置一次

## 安装依赖

SWE-bench_Pro 依赖 Docker 来保证评测可复现。

1. **安装 Docker**：参考 [Docker 安装指南](https://docs.docker.com/engine/install/)
2. **Linux 用户额外配置**：建议执行 [Linux 安装后步骤](https://docs.docker.com/engine/install/linux-postinstall/) 以避免每次都需要 `sudo`
3. 安装 Python 依赖：

```bash
pip install 'evalscope[sandbox]'
```

`evalscope[sandbox]` 会安装 [`ms-enclave`](https://pypi.org/project/ms-enclave/) 与 docker SDK。

```{note}
首次运行时会自动从 DockerHub 拉取每个实例的镜像（每个镜像约 1–4 GB），并自动克隆 `scaleapi/SWE-bench_Pro-os` 仓库至 `~/.cache/evalscope/swe_bench_pro/SWE-bench_Pro-os` 并固定到 commit `ca10a60`，用于读取每实例的 `run_script.sh`、`parser.py` 与 Dockerfile。

请确保磁盘空间和网络环境充足。
```

## 运行示例

下面示例对齐 [tests/benchmark/test_agent.py](https://github.com/modelscope/evalscope/blob/main/tests/benchmark/test_agent.py) 中的 `test_swe_bench_pro` 用例：

```python
import os
from evalscope import TaskConfig, run_task
from evalscope.config import SandboxTaskConfig

task_cfg = TaskConfig(
    model='qwen3-max',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    eval_type='openai_api',
    datasets=['swe_bench_pro'],
    dataset_args={
        'swe_bench_pro': {
            'extra_params': {
                'action_protocol': 'toolcall',  # 'toolcall' 或 'backticks'
                'max_steps': 250,               # agent 循环最大步数
                'command_timeout': 60.0,        # 单条 bash 命令超时（秒）
                'eval_timeout': 1800,           # 单实例评测超时（秒）
            }
        }
    },
    # 推理与评测阶段共用同一份 sandbox 配置
    sandbox=SandboxTaskConfig(
        default_config={
            'platform': 'linux/amd64',  # 默认值，sweap-images 仅有 amd64
            'memory_limit': '12g',      # 容器内存上限，避免 OOM-Killed（如 NodeBB）
            'cpu_limit': 4.0,           # 容器 CPU 配额（CPUs 数）
        },
    ),
    eval_batch_size=4,  # 同时存在的容器数量
    limit=5,            # 快速试跑；正式评测请去掉
    generation_config={
        'temperature': 0.7,
        'parallel_tool_calls': True,
        'stream': True,
    }
)
run_task(task_cfg=task_cfg)
```

中间产物保存在 `outputs/<timestamp>/swe_bench_pro_log/<instance_id>/`：
- `workspace/patch.diff`：模型提交的补丁
- `workspace/stdout.log` / `workspace/stderr.log`：测试运行的日志
- `workspace/output.json`：`parser.py` 解析后的结构化测试结果
- `container.log`：容器整体执行日志

## 参数说明

### `extra_params`（数据集级）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `action_protocol` | str | `toolcall` | 模型与 bash 的交互协议，可选 `toolcall` / `backticks` |
| `max_steps` | int | `250` | 每个样本 agent 循环最大步数 |
| `command_timeout` | float | `60.0` | 单条 bash 命令超时（秒） |
| `eval_timeout` | int | `3600` | 单实例评测超时（秒） |
| `swe_bench_pro_repo_path` | str | `''` | 已有 `SWE-bench_Pro-os` 克隆的本地路径；留空则自动克隆并固定到 `ca10a60` |
| `dockerhub_username` | str | `jefzda` | 托管 sweap-images 的 DockerHub 用户/组织名 |

### `sandbox.default_config`（任务级，推理与评测共用）

`TaskConfig.sandbox.default_config` 会被透传到 ms-enclave 的 `DockerSandboxConfig`，并在容器创建时映射到 docker 的 `mem_limit` / `cpu_quota` 等参数。常用字段：

| 字段 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `platform` | str | `linux/amd64` | 容器平台。sweap-images 仅提供 amd64 镜像，Apple Silicon 需保持此默认值（通过 QEMU 模拟）|
| `memory_limit` | str | `4g`（ms-enclave 默认） | 容器内存上限，例如 `'8g'`、`'12g'`。**强烈建议显式设置**，部分实例（如 NodeBB）的测试运行容易 OOM-Killed |
| `cpu_limit` | float | `2.0`（ms-enclave 默认） | 容器 CPU 数配额。多核机器建议提升至 4 以上以加速测试 |
| `network_enabled` | bool | `true` | 容器内是否启用网络 |

```{note}
**为什么要在 `sandbox.default_config` 而不是 `extra_params` 里配置内存/CPU？**

SWE-bench_Pro 的推理阶段（agent 循环）与评测阶段（运行 `run_script.sh`）使用的是**同一份** sandbox 配置。在 `TaskConfig.sandbox.default_config` 中配置一次，两个阶段都会生效，无需重复设置。
```

## 常见问题

### 1. `OOM-Killed` / 测试中途容器被杀

部分实例（如 `nodebb__nodebb-*`）的测试套件内存占用较高。请提升 `memory_limit`：

```python
sandbox=SandboxTaskConfig(default_config={'memory_limit': '16g', 'cpu_limit': 8.0})
```

### 2. Apple Silicon 上速度极慢

sweap-images 是 amd64-only 镜像，在 Apple Silicon 上通过 QEMU 模拟运行，**单实例评测可能耗时 10–30 分钟**。这是预期行为，可考虑：
- 在 amd64 Linux 机器上运行；或
- 提升 `eval_timeout`（默认 3600 秒可能不够）；
- 减小 `eval_batch_size`，避免过多容器并发导致主机过载。

### 3. `git clone https://github.com/scaleapi/SWE-bench_Pro-os.git` 失败

网络受限时自动克隆会失败。可手动克隆并通过参数传入：

```bash
git clone https://github.com/scaleapi/SWE-bench_Pro-os.git /path/to/SWE-bench_Pro-os
git -C /path/to/SWE-bench_Pro-os checkout ca10a60
```

```python
'extra_params': {'swe_bench_pro_repo_path': '/path/to/SWE-bench_Pro-os', ...}
```

### 4. DockerHub 拉取速率限制 / 拉取失败

未登录的匿名用户每 6 小时仅能拉取 100 次镜像。建议：

```bash
docker login
```

或先用 `docker pull jefzda/sweap-images:<tag>` 预热常用镜像。

### 5. `docker_image missing for instance ...`

通常意味着 `_post_process_samples` 没有跑到，或样本字段缺失 `repo` / `instance_id`。请确认数据集为 `ScaleAI/SWE-bench_Pro` 且未被自定义改动覆盖。

### 6. `Missing run_script for instance_xxx`

`SWE-bench_Pro-os` 仓库未停留在 commit `ca10a60`，或缺少子目录。删除 `~/.cache/evalscope/swe_bench_pro/SWE-bench_Pro-os` 重新克隆，或显式指定 `swe_bench_pro_repo_path`。

### 7. `pip install evalscope[sandbox]` 找不到 docker SDK

确认环境中 docker SDK 与 ms-enclave 已安装：

```bash
python -c "import docker, ms_enclave; print(docker.__version__, ms_enclave.__version__)"
```

## 引用

- Paper: [SWE-bench_Pro](https://scale.com/research/swe-bench-pro)
- Repo: <https://github.com/scaleapi/SWE-bench_Pro-os>
- Dataset: <https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro>
