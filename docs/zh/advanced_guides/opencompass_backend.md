# OpenCompass 评测后端

为便于使用OpenCompass 评测后端，我们基于OpenCompass源码做了定制，命名为`ms-opencompass`，该版本在原版基础上对评估任务的配置和执行做了一些优化，并支持pypi安装方式，使得用户可以通过EvalScope发起轻量化的OpenCompass评估任务。同时，我们先期开放了基于OpenAI API格式的接口评估任务，您可以使用[ModelScope Swift](https://github.com/modelscope/swift) 部署模型服务，其中，[swift deploy](https://swift.readthedocs.io/zh-cn/latest/LLM/VLLM%E6%8E%A8%E7%90%86%E5%8A%A0%E9%80%9F%E4%B8%8E%E9%83%A8%E7%BD%B2.html#vllm)支持使用vLLM拉起模型推理服务。

##### 安装
```shell
# 安装额外选项
pip install evalscope[opencompass]
```

##### 数据准备
目前支持的数据集有：
```text
'obqa', 'AX_b', 'siqa', 'nq', 'mbpp', 'winogrande', 'mmlu', 'BoolQ', 'cluewsc', 'ocnli', 'lambada', 'CMRC', 'ceval', 'csl', 'cmnli', 'bbh', 'ReCoRD', 'math', 'humaneval', 'eprstmt', 'WSC', 'storycloze', 'MultiRC', 'RTE', 'chid', 'gsm8k', 'AX_g', 'bustm', 'afqmc', 'piqa', 'lcsts', 'strategyqa', 'Xsum', 'agieval', 'ocnli_fc', 'C3', 'tnews', 'race', 'triviaqa', 'CB', 'WiC', 'hellaswag', 'summedits', 'GaokaoBench', 'ARC_e', 'COPA', 'ARC_c', 'DRCD'
```
数据集的详细信息可以参考[OpenCompass数据集列表](https://hub.opencompass.org.cn/home)
您可以使用以下方式，来查看数据集的名称列表：
```python
from evalscope.backend.opencompass import OpenCompassBackendManager
print(f'** All datasets from OpenCompass backend: {OpenCompassBackendManager.list_datasets()}')
```

数据集下载方式：
- 方式1：使用ModelScope数据集下载
    ```shell
    git clone https://www.modelscope.cn/datasets/swift/evalscope_resource.git
    ```

- 方式2：使用github链接下载
    ```shell
    wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-complete-20240207.zip
    ```
总大小约1.7GB，下载并解压后，将数据集文件夹（即data文件夹）放置在当前工作路径下。后续我们也即将支持托管在ModelScope上的数据集按需加载方式。


##### 模型推理服务
我们使用ModelScope swift部署模型服务，具体可参考：[ModelScope Swift部署指南](https://swift.readthedocs.io/zh-cn/latest/LLM/VLLM%E6%8E%A8%E7%90%86%E5%8A%A0%E9%80%9F%E4%B8%8E%E9%83%A8%E7%BD%B2.html#vllm)
```shell
# 安装ms-swift
pip install ms-swift

# 部署模型服务
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type llama3-8b-instruct --port 8000
```


##### 模型评估

参考示例文件： [example_eval_swift_openai_api](examples/example_eval_swift_openai_api.py) 来配置评估任务
执行评估任务：
```shell
python examples/example_eval_swift_openai_api.py
```