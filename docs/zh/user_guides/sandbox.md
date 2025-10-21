# 沙箱环境使用

为了完成LLM代码能力评测，我们需要搭建一套独立的评测环境，避免在开发环境执行错误代码从而造成不可避免的损失。目前，EvalScope 接入了[ms-enclave](https://github.com/modelscope/ms-enclave) 沙箱环境，允许用户在受控的环境中评测模型的代码能力，例如使用HumanEval、LiveCodeBench等评测基准。

接下来介绍两种不同的沙箱使用方式：

- 本地使用：在本地机器上搭建沙箱环境，并在本地进行评测，需本地机器支持Docker；
- 异地使用：在远程服务器上搭建沙箱环境，并通过API接口进行评测，需远端机器支持Docker。

## 本地使用

使用Docker在本地机器上搭建沙箱环境，并在本地进行评测，需本地机器支持Docker

### 安装环境

1. **安装Docker**：请确保您的机器上已安装Docker。您可以从[Docker官网](https://www.docker.com/get-started)下载并安装Docker。

2. **安装沙箱环境相关依赖**：在本地使用的Python环境中安装`ms-enclave`等包：

```bash
pip install evalscope[sandbox]
```

### 配置参数
在运行评测时，添加以下参数以启用沙箱环境：

```python

```