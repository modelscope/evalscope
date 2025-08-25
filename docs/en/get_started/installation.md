# Installation

## Method 1. Installing via pip
We recommend using conda to manage your environment and pip to install dependencies. You can install the latest evalscope package from PyPI.
1. Create a conda environment (optional)
```shell
# Python 3.10 is recommended
conda create -n evalscope python=3.10

# Activate the conda environment
conda activate evalscope
```
2. Install dependencies via pip
```shell
pip install evalscope
```
3. Install additional dependencies (optional)
  - To use the model service inference performance testing feature, install the perf dependency:
    ```shell
    pip install 'evalscope[perf]'
    ```
  - To use visualization features, install the app dependency:
    ```shell
    pip install 'evalscope[app]'
    ```
  - If you want to use other evaluation backends, you can optionally install OpenCompass, VLMEvalKit, and RAGEval:
    ```shell
    pip install 'evalscope[opencompass]'
    pip install 'evalscope[vlmeval]'
    pip install 'evalscope[rag]'
    ```
  - To install all dependencies:
    ```shell
    pip install 'evalscope[all]'
    ```

````{note}
Since the project has been renamed to `evalscope`, for version `v0.4.3` or earlier, you can install using:
```shell
 pip install llmuses<=0.4.3
```
To import dependencies from `llmuses`:
``` python
from llmuses import ...
```
````

## Method 2. Installing from source
By installing from source, you can use the latest code and conveniently perform secondary development and debugging.

1. Download the source code
```shell
git clone https://github.com/modelscope/evalscope.git
```
2. Install dependencies
```shell
cd evalscope/

pip install -e .
```
3. Install additional dependencies
- To use the model service inference performance testing feature, install the perf dependency:
   ```shell
   pip install '.[perf]'
   ```
 - To use visualization features, install the app dependency:
   ```shell
   pip install '.[app]'
   ```
 - If you want to use other evaluation backends, you can optionally install OpenCompass, VLMEvalKit, and RAGEval:
   ```shell
   pip install '.[opencompass]'
   pip install '.[vlmeval]'
   pip install '.[rag]'
   ```
 - To install all dependencies:
   ```shell
   pip install '.[all]'
   ```


## Docker Image

You can use the official ModelScope Docker images, which include the EvalScope library. Refer to [here](https://modelscope.cn/docs/intro/environment-setup#%E6%9C%80%E6%96%B0%E9%95%9C%E5%83%8F) for more information.