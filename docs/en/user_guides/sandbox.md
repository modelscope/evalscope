# Sandbox Environment Usage

To complete LLM code capability evaluation, we need to set up an independent evaluation environment to avoid executing erroneous code in the development environment and causing unavoidable losses. Currently, EvalScope has integrated the [ms-enclave](https://github.com/modelscope/ms-enclave) sandbox environment, allowing users to evaluate model code capabilities in a controlled environment, such as using evaluation benchmarks like HumanEval and LiveCodeBench.

The following introduces two different sandbox usage methods:

- Local usage: Set up the sandbox environment on a local machine and conduct evaluation locally, requiring Docker support on the local machine;
- Remote usage: Set up the sandbox environment on a remote server and conduct evaluation through API interfaces, requiring Docker support on the remote machine.

```{tip}
- The default **sandbox environment configuration** for datasets can be viewed in the dataset configuration, such as [HumanEval](../get_started/supported_dataset/llm.md#humaneval).
- The **number of sandbox worker processes** needs to be adjusted according to the resource situation of the local machine. It is recommended to set it to no more than half of the local CPU cores. For example, if the local machine has 8 CPU cores, it is recommended to set it to 4. Especially for multi-language images (`volcengine/sandbox-fusion`), which consume more resources, it is recommended to appropriately reduce the number of worker processes.
```

## 1. Local Usage

Use Docker to set up a sandbox environment on a local machine and conduct evaluation locally, requiring Docker support on the local machine.

### Environment Setup

1. **Install Docker**: Please ensure Docker is installed on your machine. You can download and install Docker from the [Docker official website](https://www.docker.com/get-started).

2. **Install sandbox environment dependencies**: Install packages like `ms-enclave` in your local Python environment:

```bash
pip install evalscope[sandbox]
```

### Parameter Configuration
When running evaluations, add the `use_sandbox` and `sandbox_type` parameters to automatically enable the sandbox environment. Other parameters remain the same as regular evaluations:

Here's a complete example code for model evaluation on HumanEval:
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
    use_sandbox=True, # enable sandbox
    sandbox_type='docker', # specify sandbox type
    judge_worker_num=5, # specify number of sandbox workers during evaluation
)

run_task(task_config)
```

During model evaluation, EvalScope will automatically start and manage the sandbox environment, ensuring code runs in an isolated environment. The console will display output like:
```text
[INFO:ms_enclave] Local sandbox manager started
...
```

## 2. Remote Usage

Set up the sandbox environment on a remote server and conduct evaluation through API interfaces, requiring Docker support on the remote machine.

### Environment Setup

You need to install and configure separately on both the remote machine and local machine.

#### Remote Machine

The environment installation on the remote machine is similar to the local usage method described above:

1. **Install Docker**: Please ensure Docker is installed on your machine. You can download and install Docker from the [Docker official website](https://www.docker.com/get-started).

2. **Install sandbox environment dependencies**: Install packages like `ms-enclave` in remote Python environment:

```bash
pip install evalscope[sandbox]
```

3. **Start sandbox server**: Run the following command to start the sandbox server:

```bash
ms-enclave server --host 0.0.0.0 --port 1234
```

#### Local Machine

The local machine does not need Docker installation at this point, but needs to install EvalScope:

```bash
pip install evalscope[sandbox]
```

### Parameter Configuration

When running evaluations, add the `use_sandbox` parameter to automatically enable the sandbox environment, and specify the remote sandbox server's API address in `sandbox_manager_config`:

Complete example code is as follows:
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
    use_sandbox=True, # enable sandbox
    sandbox_type='docker', # specify sandbox type
    sandbox_manager_config={
        'base_url': 'http://<remote_host>:1234'  # remote sandbox manager URL
    },
    judge_worker_num=5, # specify number of sandbox workers during evaluation
)

run_task(task_config)
```

During model evaluation, EvalScope will communicate with the remote sandbox server through API, ensuring code runs in an isolated environment. The console will display output like:
```text
[INFO:ms_enclave] HTTP sandbox manager started, connected to http://<remote_host>:1234
...
```