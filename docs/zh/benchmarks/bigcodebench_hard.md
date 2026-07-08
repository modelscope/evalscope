# BigCodeBench-Hard


## 概述

BigCodeBench-Hard 是 BigCodeBench 的一个精选子集，包含 148 个更贴近真实编程场景的任务。这些任务需要更复杂的推理能力和多步骤的问题解决能力。

## 任务描述

- **任务类型**：代码生成（Python）
- **输入**：编程任务描述（文档字符串或自然语言指令）
- **输出**：完整的 Python 函数实现
- **难度**：高于 BigCodeBench-Full，更接近真实世界的复杂度

## 主要特性

- 从 BigCodeBench 中精选出的 148 个具有挑战性的任务
- 要求复杂的推理能力和多工具协同使用
- 提供两种评估模式：Complete（文档字符串补全）和 Instruct（自然语言指令）
- 使用 `unittest.TestCase` 进行全面的正确性验证
- 支持 pass@k 指标计算

## 评估说明

- **需要沙箱环境**：要求预装 70 多个 Python 库的沙箱环境
- 通过 `split` 参数可选择两种模式：`complete`（文档字符串补全）或 `instruct`（自然语言指令）
- 每个问题默认超时时间为 240 秒
- 启用 `calibrate` 选项会在生成代码前添加 `code_prompt`，以对齐函数签名
- 沙箱环境设置详见 [沙箱文档](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/sandbox.html)

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `bigcodebench_hard` |
| **数据集ID** | [evalscope/bigcodebench-hard](https://modelscope.cn/datasets/evalscope/bigcodebench-hard/summary) |
| **论文** | N/A |
| **标签** | `Coding` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估划分版本** | `v0.1.4` |
| **聚合方式** | `mean_and_pass_at_k` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 148 |
| 提示词长度（平均） | 777.94 字符 |
| 提示词长度（最小/最大） | 274 / 1816 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "207adc46",
      "content": "Download all files from a specific directory on an FTP server using wget in a subprocess. Args: ftp_server (str): The FTP server address. Default is 'ftp.dlptest.com'. ftp_user (str): The FTP server username. Default is 'dlpuser'. ftp_passwor ... [TRUNCATED 798 chars] ...  FTP server.\nYou should write self-contained code starting with:\n```\nimport subprocess\nimport ftplib\nimport os\ndef task_func(ftp_server='ftp.dlptest.com', ftp_user='dlpuser', ftp_password='rNrKYTX9g7z3RgJRmxWuGHbeu', ftp_dir='/ftp/test'):\n```"
    }
  ],
  "target": "    # Attempt to connect to the FTP server\n    try:\n        ftp_obj = ftplib.FTP(ftp_server)\n    except Exception as e:\n        raise Exception(f'Failed to connect to FTP server {ftp_server}: {str(e)}')\n\n    # Attempt to login to the FTP serv ... [TRUNCATED 625 chars] ...        command = f'wget ftp://{ftp_user}:{ftp_password}@{ftp_server}{ftp_dir}/{filename} -P {download_dir}'\n        subprocess.call(command, shell=True)\n        downloaded_files.append(filename)\n\n    ftp_obj.quit()\n    return downloaded_files",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "task_id": "BigCodeBench/13",
    "entry_point": "task_func",
    "complete_prompt": "import subprocess\nimport ftplib\nimport os\n\ndef task_func(ftp_server='ftp.dlptest.com', ftp_user='dlpuser', ftp_password='rNrKYTX9g7z3RgJRmxWuGHbeu', ftp_dir='/ftp/test'):\n    \"\"\"\n    Download all files from a specific directory on an FTP serv ... [TRUNCATED 909 chars] ... ctory. Outputs the message \"Failed to change to directory {ftp_dir} on server {ftp_server}: {str(e)}\"\n    \n    Requirements:\n    - subprocess\n    - ftplib\n    - os\n\n    Example:\n    >>> task_func()\n    ['file1.txt', 'file2.jpg', ...]\n    \"\"\"\n",
    "code_prompt": "import subprocess\nimport ftplib\nimport os\ndef task_func(ftp_server='ftp.dlptest.com', ftp_user='dlpuser', ftp_password='rNrKYTX9g7z3RgJRmxWuGHbeu', ftp_dir='/ftp/test'):\n",
    "test": "import unittest\nfrom unittest.mock import patch\nimport os\nclass TestCases(unittest.TestCase):\n    def setUp(self):\n        \"\"\"Setup a clean test environment before each test.\"\"\"\n        if not os.path.exists(\"downloaded_files\"):\n            o ... [TRUNCATED 2565 chars] ... with self.assertRaises(Exception) as context:\n            task_func(ftp_dir=\"/invalid_directory\")\n        self.assertEqual(str(context.exception), f'Failed to change to directory /invalid_directory on server ftp.dlptest.com: {error_message}')"
  }
}
```

## 提示模板

**提示模板：**
```text
{prompt}
```

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `split` | `str` | `instruct` | 评估模式："complete"（文档字符串补全）或 "instruct"（自然语言指令）。可选值：['complete', 'instruct'] |
| `version` | `str` | `default` | 数据集版本。使用 "default" 表示最新可用版本。 |
| `calibrate` | `bool` | `True` | 是否在解决方案前添加 `code_prompt` 以对齐函数签名。 |
| `docker_build_context` | `str` | `` | 可选的本地 Docker 构建上下文。设置后将覆盖默认沙箱镜像。 |
| `dockerfile` | `str` | `Dockerfile` | `docker_build_context` 内的 Dockerfile 路径。 |
| `force_rebuild` | `bool` | `False` | 是否强制重建可选的本地 Docker 镜像。 |

## 沙箱配置

此基准测试需要沙箱环境来执行代码。

```json
{
  "image": "bigcodebench/bigcodebench-evaluate:latest",
  "tools_config": {
    "shell_executor": {},
    "python_executor": {}
  },
  "memory_limit": "4g"
}
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets bigcodebench_hard \
    --sandbox '{"enabled": true}' \
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
    datasets=['bigcodebench_hard'],
    sandbox={'enabled': True},
    dataset_args={
        'bigcodebench_hard': {
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
