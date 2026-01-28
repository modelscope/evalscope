# SWE-bench_Verified


## 概述

SWE-bench Verified 是从 SWE-bench 中人工验证筛选出的 500 个样本子集，旨在测试系统自动解决真实 GitHub 问题的能力。每个样本均来自热门 Python 仓库中的真实 bug 修复或功能实现。

## 任务描述

- **任务类型**：自动化软件工程 / Bug 修复
- **输入**：包含仓库上下文的 GitHub issue 描述
- **输出**：解决该 issue 的代码补丁（diff 格式）
- **仓库范围**：12 个热门 Python 项目（如 Django、Flask、Requests 等）

## 主要特性

- 500 对经人工验证的 Issue-Pull Request 数据
- 来自生产级 Python 仓库的真实世界 bug
- 通过单元测试进行验证评估
- 基于 Docker 的隔离执行环境
- 同时考察对 bug 的理解能力与代码修改能力

## 评估说明

- 评估前需先安装 `pip install swebench==4.1.0`
- 每个仓库的 Docker 镜像会自动构建或拉取
- 每个实例超时时间为 1800 秒（30 分钟）
- 详细设置说明请参阅 [使用文档](https://evalscope.readthedocs.io/zh-cn/latest/third_party/swe_bench.html)
- 支持本地构建镜像和远程拉取镜像两种方式


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `swe_bench_verified` |
| **数据集ID** | [princeton-nlp/SWE-bench_Verified](https://modelscope.cn/datasets/princeton-nlp/SWE-bench_Verified/summary) |
| **论文** | N/A |
| **标签** | `Coding` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估分割** | `test` |


## 数据统计

*统计数据暂不可用。*

## 样例示例

*样例示例暂不可用。*

## 提示模板

**提示模板：**
```text
{question}
```

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `inference_dataset_id` | `str` | `princeton-nlp/SWE-bench_oracle` | 用于获取推理上下文的 Oracle 数据集 ID。 |
| `build_docker_images` | `bool` | `True` | 为每个样本在本地构建 Docker 镜像。 |
| `pull_remote_images_if_available` | `bool` | `True` | 在构建前尝试拉取已存在的远程 Docker 镜像。 |
| `force_arch` | `str` | `` | 可选地强制为特定架构拉取/构建 Docker 镜像。可选值：['', 'arm64', 'x86_64'] |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets swe_bench_verified \
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
    datasets=['swe_bench_verified'],
    dataset_args={
        'swe_bench_verified': {
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```