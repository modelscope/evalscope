# SWE-bench_Lite

## 概述

SWE-bench Lite 是 SWE-bench 的一个精选子集，包含来自 11 个热门 Python 代码仓库的 300 个 Issue-Pull Request 配对。它为评估自动化软件工程能力提供了一个更易上手的切入点。

## 任务描述

- **任务类型**：自动化软件工程 / Bug修复
- **输入**：包含仓库上下文的GitHub问题描述
- **输出**：解决该问题的代码补丁（diff格式）
- **规模**：精心挑选的300个测试实例

## 主要特点

- 包含 300 个 Issue-Pull Request 测试对
- 覆盖 11 个流行的 Python 代码仓库
- 具有已验证解决方案的真实世界 Bug
- 通过单元测试进行验证评估
- 比完整的 SWE-bench 更易于管理，但仍具挑战性

## 评估说明

- 通过 `TaskConfig.sandbox.engine='e2b'` 支持官方 E2B 和阿里云 Agent Sandbox，避免使用本地 Docker 和本地镜像下载。请参阅 [沙箱指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/sandbox.html#e2b)。
- 在评估前需要执行 `pip install swebench==4.1.0`
- 针对每个代码仓库自动构建/拉取 Docker 镜像
- 详细设置说明请参阅 [使用文档](https://evalscope.readthedocs.io/zh-cn/latest/third_party/swe_bench.html)
- 适用于初步模型对比的流行基准测试变体

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `swe_bench_lite` |
| **数据集ID** | [princeton-nlp/SWE-bench_Lite](https://modelscope.cn/datasets/princeton-nlp/SWE-bench_Lite/summary) |
| **论文** | N/A |
| **标签** | `Coding` |
| **指标** | `accuracy` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 300 |
| 提示词长度（平均值） | 57725.21 字符 |
| 提示词长度（最小/最大值） | 5169 / 372080 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "dcd8fb65",
      "content": "You will be provided with a partial code base and an issue statement explaining a problem to resolve.\n<issue>\nModeling's `separability_matrix` does not compute separability correctly for nested CompoundModels\nConsider the following model:\r\n\r\n ... [TRUNCATED 18305 chars] ... 2.0\n+        while y != y1:\n+            points.append((x, y))\n+            err -= dx\n+            if err < 0:\n+                x += sx\n+                err += dy\n+            y += sy\n \n+    points.append((x, y))\n     return points\n</patch>\n\n"
    }
  ],
  "id": 0,
  "group_id": 0,
  "metadata": {
    "problem_statement": "Modeling's `separability_matrix` does not compute separability correctly for nested CompoundModels\nConsider the following model:\r\n\r\n```python\r\nfrom astropy.modeling import models as m\r\nfrom astropy.modeling.separable import separability_matri ... [TRUNCATED 762 chars] ...       [ True,  True, False, False],\r\n       [False, False,  True,  True],\r\n       [False, False,  True,  True]])\r\n```\r\nSuddenly the inputs and outputs are no longer separable?\r\n\r\nThis feels like a bug to me, but I might be missing something?\n",
    "instance_id": "astropy__astropy-12907",
    "base_commit": "d16bfe05a744909de4b27f5875fe0d4ed41ce607",
    "patch": "diff --git a/astropy/modeling/separable.py b/astropy/modeling/separable.py\n--- a/astropy/modeling/separable.py\n+++ b/astropy/modeling/separable.py\n@@ -242,7 +242,7 @@ def _cstack(left, right):\n         cright = _coord_matrix(right, 'right', noutp)\n     else:\n         cright = np.zeros((noutp, right.shape[1]))\n-        cright[-right.shape[0]:, -right.shape[1]:] = 1\n+        cright[-right.shape[0]:, -right.shape[1]:] = right\n \n     return np.hstack([cleft, cright])\n \n",
    "PASS_TO_PASS": [
      "astropy/modeling/tests/test_separable.py::test_coord_matrix",
      "astropy/modeling/tests/test_separable.py::test_cdot",
      "astropy/modeling/tests/test_separable.py::test_cstack",
      "astropy/modeling/tests/test_separable.py::test_arith_oper",
      "astropy/modeling/tests/test_separable.py::test_separable[compound_model0-result0]",
      "astropy/modeling/tests/test_separable.py::test_separable[compound_model1-result1]",
      "astropy/modeling/tests/test_separable.py::test_separable[compound_model2-result2]",
      "astropy/modeling/tests/test_separable.py::test_separable[compound_model3-result3]",
      "astropy/modeling/tests/test_separable.py::test_separable[compound_model4-result4]",
      "astropy/modeling/tests/test_separable.py::test_separable[compound_model5-result5]",
      "... [TRUNCATED 3 more items] ..."
    ],
    "FAIL_TO_PASS": [
      "astropy/modeling/tests/test_separable.py::test_separable[compound_model6-result6]",
      "astropy/modeling/tests/test_separable.py::test_separable[compound_model9-result9]"
    ],
    "test_patch": "diff --git a/astropy/modeling/tests/test_separable.py b/astropy/modeling/tests/test_separable.py\n--- a/astropy/modeling/tests/test_separable.py\n+++ b/astropy/modeling/tests/test_separable.py\n@@ -28,6 +28,13 @@\n p1 = models.Polynomial1D(1, nam ... [TRUNCATED 931 chars] ...          [True,  True,  False, False, False],\n+                        [False, False, True,  False, False],\n+                        [False, False, False, True,  False],\n+                        [False, False, False, False, True]]))),\n }\n \n \n",
    "version": "4.3",
    "repo": "astropy/astropy",
    "environment_setup_commit": "298ccb478e6bf092953bca67a3d29dc6c35f6752",
    "hints_text": "",
    "created_at": "2022-03-03T15:14:54Z",
    "docker_image": "swebench/sweb.eval.arm64.astropy_1776_astropy-12907:latest"
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
| `inference_dataset_id` | `str` | `princeton-nlp/SWE-bench_oracle` | 用于从选定的数据集中心获取推理上下文的 Oracle 数据集 ID。 |
| `inference_dataset_revision` | `str | null` | `None` | Oracle 推理数据集的可选修订版本。 |
| `force_arch` | `str` | `` | 可选地强制针对特定架构拉取/构建 docker 镜像。选项：['', 'arm64', 'x86_64'] |
| `dockerhub_username` | `str` | `swebench` | 远程 SWE-bench 镜像的 DockerHub 用户/组织命名空间。 |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets swe_bench_lite \
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
    datasets=['swe_bench_lite'],
    dataset_args={
        'swe_bench_lite': {
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
