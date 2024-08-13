# 安装

## 使用pip安装
我们推荐使用conda来管理环境，并使用pip安装依赖:
1. 创建conda环境 (可选)
```shell
# 建议使用 python 3.10
conda create -n evalscope python=3.10

# 激活conda环境
conda activate evalscope
```
2. pip安装依赖
```shell
pip install evalscope                # 安装 Native backend (默认)
# 额外选项
pip install evalscope[opencompass]   # 安装 OpenCompass backend
pip install evalscope[vlmeval]       # 安装 VLMEvalKit backend
pip install evalscope[all]           # 安装所有 backends (Native, OpenCompass, VLMEvalKit)
```

```{warning}
版本废弃说明: 对于`v0.4.3`或更早版本，您可以使用以下命令安装：
```shell
pip install llmuses<=0.4.3

# Usage:
from llmuses.run import run_task
...
```


## 使用源码安装
1. 下载源码
```shell
git clone https://github.com/modelscope/evalscope.git
```
2. 安装依赖
```shell
cd evalscope/

pip install -e .                  # 安装 Native backend
# 额外选项
pip install -e '.[opencompass]'   # 安装 OpenCompass backend
pip install -e '.[vlmeval]'       # 安装 VLMEvalKit backend
pip install -e '.[all]'           # 安装所有 backends (Native, OpenCompass, VLMEvalKit)
```