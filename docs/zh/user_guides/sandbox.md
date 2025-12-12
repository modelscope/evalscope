# 沙箱环境使用

为了完成LLM代码能力评测，我们需要搭建一套独立的评测环境，避免在开发环境执行错误代码从而造成不可避免的损失。目前，EvalScope 接入了[ms-enclave](https://github.com/modelscope/ms-enclave) 沙箱环境，允许用户在受控的环境中评测模型的代码能力，例如使用HumanEval、LiveCodeBench等评测基准。

接下来介绍两种不同的沙箱使用方式：

- 本地使用：在本地机器上搭建沙箱环境，并在本地进行评测，需本地机器支持Docker；
- 异地使用：在远程服务器上搭建沙箱环境，并通过API接口进行评测，需远端机器支持Docker。

```{tip}
- 数据集默认的**沙箱环境配置**可以在数据集配置中查看，例如[HumanEval](../get_started/supported_dataset/llm.md#humaneval)。
- **沙箱工作进程数** 需要根据本地机器的资源情况进行调整，建议设置为不超过本地CPU核心数的一半，例如本地有8核CPU，则建议设置为4，尤其是多语言镜像(`volcengine/sandbox-fusion`)，其消耗资源更多，建议适当减少工作进程数。
```

## 1. 本地使用

使用Docker在本地机器上搭建沙箱环境，并在本地进行评测，需本地机器支持Docker

### 安装环境

1. **安装Docker**：请确保您的机器上已安装Docker。您可以从[Docker官网](https://www.docker.com/get-started)下载并安装Docker。

2. **安装沙箱环境相关依赖**：在本地使用的Python环境中安装`ms-enclave`等包：

```bash
pip install evalscope[sandbox]
```

### 配置参数
在运行评测时，添加`use_sandbox`和`sandbox_type`参数以自动启用沙箱环境，其余参数与普通评测相同：

下面是一个在HumanEval上进行模型评测的完整示例代码：
```python
from dotenv import dotenv_values
env = dotenv_values('.env')
from evalscope import TaskConfig, run_task

task_config = TaskConfig(
    model='qwen-plus',
    datasets=['humaneval'],
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=env.get('DASHSCOPE_API_KEY'),
    eval_type='openai_api',
    eval_batch_size=5,
    limit=5,
    generation_config={
        'max_tokens': 4096,
        'temperature': 0.0,
        'seed': 42,
    },
    use_sandbox=True, # 启用沙箱
    sandbox_type='docker', # 指定沙箱类型
    judge_worker_num=5, # 指定评测时的沙箱工作进程数
)

run_task(task_config)
```

在模型评测过程中，EvalScope会自动启动并管理沙箱环境，确保代码在隔离的环境中运行。控制台有如下输出：
```text
[INFO:ms_enclave] Local sandbox manager started
...
```

## 2. 异地使用

在远程服务器上搭建沙箱环境，并通过API接口进行评测，需远端机器支持Docker。

### 安装环境

你需要分别在远端机器和本地机器上进行安装和配置。

#### 远端机器

在远端机器上安装的环境与上述本地使用方式类似：

1. **安装Docker**：请确保您的机器上已安装Docker。您可以从[Docker官网](https://www.docker.com/get-started)下载并安装Docker。

2. **安装沙箱环境相关依赖**：在远端机器的Python环境中安装`ms-enclave`等包：

```bash
pip install evalscope[sandbox]
```

3. **启动沙箱服务器**：运行以下命令以启动沙箱服务器：

```bash
ms-enclave server --host 0.0.0.0 --port 1234
```

#### 本地机器

此时本地机器无需安装Docker，但需要安装EvalScope：

```bash
pip install evalscope[sandbox]
```

### 配置参数

在运行评测时，添加`use_sandbox`参数以自动启用沙箱环境，并在`sandbox_manager_config`中指定远端沙箱服务器的API地址：

完整示例代码如下：
```python
from dotenv import dotenv_values
env = dotenv_values('.env')
from evalscope import TaskConfig, run_task

task_config = TaskConfig(
    model='qwen-plus',
    datasets=['humaneval'],
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=env.get('DASHSCOPE_API_KEY'),
    eval_type='openai_api',
    eval_batch_size=5,
    limit=5,
    generation_config={
        'max_tokens': 4096,
        'temperature': 0.0,
        'seed': 42,
    },
    use_sandbox=True, # 启用沙箱
    sandbox_type='docker', # 指定沙箱类型
    sandbox_manager_config={
        'base_url': 'http://<remote_host>:1234'  # 远端沙箱管理器URL
    },
    judge_worker_num=5, # 指定评测时的沙箱工作进程数
)

run_task(task_config)
```

在模型评测过程中，EvalScope会通过API与远端沙箱服务器进行通信，确保代码在隔离的环境中运行。控制台有如下输出：
```text
[INFO:ms_enclave] HTTP sandbox manager started, connected to http://<remote_host>:1234
...
```
