# 安装

## 方式1. 使用 pip 安装（推荐）

我们推荐使用 conda 来管理环境，并使用 pip 安装依赖。

1. 创建 conda 环境（可选）
```shell
# 建议使用 python 3.10
conda create -n evalscope python=3.10
conda activate evalscope
```

2. 安装 EvalScope
```shell
pip install evalscope
```

3. 验证安装
```shell
evalscope --version
```

4. 安装额外功能（可选）

根据您的需求，安装相应的功能扩展：

| 功能 | 安装命令 |
|------|----------|
| 推理性能压测 | `pip install 'evalscope[perf]'` |
| 可视化服务 | `pip install 'evalscope[service]'` |
| AIGC 评测 | `pip install 'evalscope[aigc]'` |
| OpenCompass 后端 | `pip install 'evalscope[opencompass]'` |
| VLMEvalKit 后端 | `pip install 'evalscope[vlmeval]'` |
| RAG 评测 | `pip install 'evalscope[rag]'` |
| 全部安装 | `pip install 'evalscope[all]'` |

## 方式2. 使用源码安装

使用源码安装的方式，可以使用最新的代码，并方便地进行二次开发和调试。

1. 下载源码
```shell
git clone https://github.com/modelscope/evalscope.git
```

2. 安装依赖
```shell
cd evalscope/
pip install -e .
```

3. 安装额外依赖（可选）

| 功能 | 安装命令 |
|------|----------|
| 推理性能压测 | `pip install '.[perf]'` |
| 可视化服务 | `pip install '.[service]'` |
| AIGC 评测 | `pip install '.[aigc]'` |
| OpenCompass 后端 | `pip install '.[opencompass]'` |
| VLMEvalKit 后端 | `pip install '.[vlmeval]'` |
| RAG 评测 | `pip install '.[rag]'` |
| 全部安装 | `pip install '.[all]'` |

## 镜像

使用镜像可以查看 ModelScope 官方镜像，其中包含了 EvalScope 库，参考[这里](https://modelscope.cn/docs/intro/environment-setup#%E6%9C%80%E6%96%B0%E9%95%9C%E5%83%8F)

```{note}
本项目曾用名 `llmuses`。如果您需要使用 `v0.4.3` 或更早版本，请运行 `pip install llmuses<=0.4.3` 并使用 `from llmuses import ...` 导入。
```
