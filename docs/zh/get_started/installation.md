# 安装

## 方式1. 使用pip安装
我们推荐使用conda来管理环境，并使用pip安装依赖，可以使用最新的evalscope pypi包。
1. 创建conda环境 (可选)
```shell
# 建议使用 python 3.10
conda create -n evalscope python=3.10

# 激活conda环境
conda activate evalscope
```
2. pip安装依赖
```shell
pip install evalscope
```
3. 安装额外依赖（可选）
  - 若要使用模型服务推理压测功能，需安装perf依赖：
    ```shell
    pip install 'evalscope[perf]'
    ```
  - 若要使用可视化功能，需安装app依赖：
    ```shell
    pip install 'evalscope[app]'
    ```
  - 若使用其他评测后端，可按需安装OpenCompass, VLMEvalKit, RAGEval：
    ```shell
    pip install 'evalscope[opencompass]'
    pip install 'evalscope[vlmeval]'
    pip install 'evalscope[rag]'
    ```
  - 安装所有依赖：
    ```shell
    pip install 'evalscope[all]'
    ```

````{note}
由于项目更名为`evalscope`，对于`v0.4.3`或更早版本，您可以使用以下安装：
```shell
 pip install llmuses<=0.4.3
```
使用`llmuses`导入相关依赖：
``` python
from llmuses import ...
```
````

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
3. 安装额外依赖
- 若要使用模型服务推理压测功能，需安装perf依赖：
   ```shell
   pip install '.[perf]'
   ```
 - 若要使用可视化功能，需安装app依赖：
   ```shell
   pip install '.[app]'
   ```
 - 若使用其他评测后端，可按需安装OpenCompass, VLMEvalKit, RAGEval：
   ```shell
   pip install '.[opencompass]'
   pip install '.[vlmeval]'
   pip install '.[rag]'
   ```
 - 安装所有依赖：
   ```shell
   pip install '.[all]'
   ```


## 镜像

使用镜像可以查看ModelScope官方镜像，其中包含了EvalScope库，参考[这里](https://modelscope.cn/docs/intro/environment-setup#%E6%9C%80%E6%96%B0%E9%95%9C%E5%83%8F)
