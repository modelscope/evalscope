# SWE-bench_Lite

## 概述

SWE-bench Lite 是 SWE-bench 的一个精选子集，包含来自 11 个热门 Python 仓库的 300 个 Issue-Pull Request 配对。它为评估自动化软件工程能力提供了一个更易上手的入口。

## 任务描述

- **任务类型**：自动化软件工程 / 缺陷修复
- **输入**：带有仓库上下文的 GitHub issue 描述
- **输出**：解决该 issue 的代码补丁（diff 格式）
- **规模**：300 个精心挑选的测试实例

## 主要特性

- 包含 300 个测试用的 Issue-Pull Request 配对
- 覆盖 11 个热门 Python 仓库
- 真实世界的缺陷及其已验证的解决方案
- 通过单元测试进行评估验证
- 相比完整版 SWE-bench 更易于管理，但仍具挑战性

## 评估说明

- 评估前需先安装 `pip install swebench==4.1.0`
- 每个仓库的 Docker 镜像会自动构建或拉取
- 详细设置说明请参阅 [使用文档](https://evalscope.readthedocs.io/zh-cn/latest/third_party/swe_bench.html)
- 是用于初步模型对比的常用基准变体

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `swe_bench_lite` |
| **数据集ID** | [princeton-nlp/SWE-bench_Lite](https://modelscope.cn/datasets/princeton-nlp/SWE-bench_Lite/summary) |
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
| `build_docker_images` | `bool` | `True` | 为每个样本在本地构建 Docker 镜像。 |
| `pull_remote_images_if_available` | `bool` | `True` | 在构建前尝试拉取已存在的远程 Docker 镜像。 |
| `inference_dataset_id` | `str` | `princeton-nlp/SWE-bench_oracle` | 用于获取推理上下文的 Oracle 数据集 ID。 |
| `force_arch` | `str` | `` | 可选地强制为特定架构拉取/构建 Docker 镜像。选项：['', 'arm64', 'x86_64'] |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets swe_bench_lite \
    --limit 10  # 正式评估时请移除此行
```

### 使用 Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['swe_bench_lite'],
    dataset_args={
        'swe_bench_lite': {
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请移除此行
)

run_task(task_cfg=task_cfg)
```