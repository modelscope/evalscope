# VLMEvalKit 评测后端

为便于使用VLMEvalKit 评测后端，我们基于VLMEvalKit源码做了定制，命名为`ms-vlmeval`，该版本在原版基础上对评估任务的配置和执行进行了封装，并支持pypi安装方式，使得用户可以通过EvalScope发起轻量化的VLMEvalKit评估任务。同时，我们支持基于OpenAI API格式的接口评估任务，您可以使用ModelScope [swift](https://github.com/modelscope/swift) 部署多模态模型服务。

##### 安装
```shell
# 安装额外选项
pip install evalscope[vlmeval]
```

##### 数据准备
目前支持的数据集有：
```text
'COCO_VAL', 'MME', 'HallusionBench', 'POPE', 'MMBench_DEV_EN', 'MMBench_TEST_EN', 'MMBench_DEV_CN', 'MMBench_TEST_CN', 'MMBench', 'MMBench_CN', 'MMBench_DEV_EN_V11', 'MMBench_TEST_EN_V11', 'MMBench_DEV_CN_V11', 'MMBench_TEST_CN_V11', 'MMBench_V11', 'MMBench_CN_V11', 'SEEDBench_IMG', 'SEEDBench2', 'SEEDBench2_Plus', 'ScienceQA_VAL', 'ScienceQA_TEST', 'MMT-Bench_ALL_MI', 'MMT-Bench_ALL', 'MMT-Bench_VAL_MI', 'MMT-Bench_VAL', 'AesBench_VAL', 'AesBench_TEST', 'CCBench', 'AI2D_TEST', 'MMStar', 'RealWorldQA', 'MLLMGuard_DS', 'BLINK', 'OCRVQA_TEST', 'OCRVQA_TESTCORE', 'TextVQA_VAL', 'DocVQA_VAL', 'DocVQA_TEST', 'InfoVQA_VAL', 'InfoVQA_TEST', 'ChartQA_TEST', 'MathVision', 'MathVision_MINI', 'MMMU_DEV_VAL', 'MMMU_TEST', 'OCRBench', 'MathVista_MINI', 'LLaVABench', 'MMVet', 'MTVQA_TEST', 'MMLongBench_DOC', 'VCR_EN_EASY_500', 'VCR_EN_EASY_100', 'VCR_EN_EASY_ALL', 'VCR_EN_HARD_500', 'VCR_EN_HARD_100', 'VCR_EN_HARD_ALL', 'VCR_ZH_EASY_500', 'VCR_ZH_EASY_100', 'VCR_ZH_EASY_ALL', 'VCR_ZH_HARD_500', 'VCR_ZH_HARD_100', 'VCR_ZH_HARD_ALL', 'MMBench-Video', 'Video-MME', 'MMBench_DEV_EN', 'MMBench_TEST_EN', 'MMBench_DEV_CN', 'MMBench_TEST_CN', 'MMBench', 'MMBench_CN', 'MMBench_DEV_EN_V11', 'MMBench_TEST_EN_V11', 'MMBench_DEV_CN_V11', 'MMBench_TEST_CN_V11', 'MMBench_V11', 'MMBench_CN_V11', 'SEEDBench_IMG', 'SEEDBench2', 'SEEDBench2_Plus', 'ScienceQA_VAL', 'ScienceQA_TEST', 'MMT-Bench_ALL_MI', 'MMT-Bench_ALL', 'MMT-Bench_VAL_MI', 'MMT-Bench_VAL', 'AesBench_VAL', 'AesBench_TEST', 'CCBench', 'AI2D_TEST', 'MMStar', 'RealWorldQA', 'MLLMGuard_DS', 'BLINK'
```
数据集的详细信息可以参考[VLMEvalKit支持的图文多模态评测集](https://github.com/open-compass/VLMEvalKit/blob/main/docs/zh-CN/README_zh-CN.md#%E6%94%AF%E6%8C%81%E7%9A%84%E5%9B%BE%E6%96%87%E5%A4%9A%E6%A8%A1%E6%80%81%E8%AF%84%E6%B5%8B%E9%9B%86)
您可以使用以下方式，来查看数据集的名称列表：
```python
from evalscope.backend.vlm_eval_kit import VLMEvalKitBackendManager
print(f'** All models from VLMEvalKit backend: {VLMEvalKitBackendManager.list_supported_models().keys()}')

```

在加载数据集时，若本地不存在该数据集文件，将会自动下载数据集到 `~/LMUData/` 目录下。


##### 模型评估
模型评估有两种方式可以选择：

###### 1. ModelScope Swift部署模型服务评估

**模型部署**
使用ModelScope swift部署模型服务，具体可参考：[ModelScope Swift MLLM 部署指南](https://swift.readthedocs.io/zh-cn/latest/Multi-Modal/MLLM%E9%83%A8%E7%BD%B2%E6%96%87%E6%A1%A3.html)
```shell
# 安装ms-swift
pip install ms-swift

# 部署qwen-vl-chat多模态模型服务
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type qwen-vl-chat --model_id_or_path models/Qwen-VL-Chat
```

**模型评估**

参考示例文件： [example_eval_vlm_swift](examples/example_eval_vlm_swift.py) 来配置评估任务
执行评估任务：
```shell
python examples/example_eval_vlm_swift.py
```

###### 2. 本地模型推理评估

**模型推理评估**
不启动模型服务，直接在本地进行推理，参考示例文件： [example_eval_vlm_local](examples/example_eval_vlm_local.py) 来配置评估任务
执行评估任务：
```shell
python examples/example_eval_vlm_local.py
```


##### (可选) 部署裁判员模型
部署本地语言模型作为评判 / 选择提取器，同样使用ModelScope swift部署模型服务，具体可参考：[ModelScope Swift LLM 部署指南](https://swift.readthedocs.io/zh-cn/latest/LLM/VLLM%E6%8E%A8%E7%90%86%E5%8A%A0%E9%80%9F%E4%B8%8E%E9%83%A8%E7%BD%B2.html)
。在未部署裁判员模型模型时，将使用精确匹配。
```shell
# 部署qwen2-7b作为裁判员
CUDA_VISIBLE_DEVICES=1 swift deploy --model_type qwen2-7b-instruct --model_id_or_path models/Qwen2-7B-Instruct --port 8866
```
**必须配置裁判员模型环境变量才能正确调用模型**，需要配置的环境变量如下：
```
OPENAI_API_KEY=EMPTY
OPENAI_API_BASE=http://127.0.0.1:8866/v1/chat/completions # 裁判员模型的api_base
LOCAL_LLM=qwen2-7b-instruct #裁判员模型的 model_id
```