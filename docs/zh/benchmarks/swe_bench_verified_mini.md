# SWE-bench_Verified_mini


## 概述

SWE-bench Verified Mini 是 SWE-bench Verified 的一个紧凑子集，包含 50 个精心挑选的样本，在保持与完整数据集相同的性能分布、测试通过率和难度的同时，仅需 5GB 存储空间（而非 130GB）。

## 任务描述

- **任务类型**：自动化软件工程 / 缺陷修复
- **输入**：带有仓库上下文的 GitHub Issue 描述
- **输出**：解决该 Issue 的代码补丁（diff 格式）
- **规模**：50 个样本（完整 Verified 集合为 500 个）

## 主要特性

- SWE-bench Verified 的代表性 50 样本子集
- 与完整数据集具有相同的难度分布
- 存储需求大幅降低（5GB vs 130GB）
- 非常适合快速评估和开发迭代
- 在基准测试中保持统计有效性

## 评估说明

- 评估前需先安装 `pip install swebench==4.1.0`
- Docker 镜像会自动构建或拉取
- 详细设置请参阅 [使用文档](https://evalscope.readthedocs.io/zh-cn/latest/third_party/swe_bench.html)
- 适用于快速原型设计和模型初步评估


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `swe_bench_verified_mini` |
| **数据集ID** | [evalscope/swe-bench-verified-mini](https://modelscope.cn/datasets/evalscope/swe-bench-verified-mini/summary) |
| **论文** | N/A |
| **标签** | `Coding` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

*统计数据不可用。*

## 样例示例

*样例示例不可用。*

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