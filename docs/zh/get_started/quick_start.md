# 快速开始

### 简单评估
在指定的若干数据集上评估某个模型，流程如下：

如果使用git安装，可在任意路径下执行：
```shell
python -m evalscope.run --model ZhipuAI/chatglm3-6b --template-type chatglm3 --datasets arc --limit 100
```
如果使用源码安装，在evalscope路径下执行：
```shell
python evalscope/run.py --model ZhipuAI/chatglm3-6b --template-type chatglm3 --datasets mmlu ceval --limit 10
```
其中，--model参数指定了模型的ModelScope model id，模型链接：[ZhipuAI/chatglm3-6b](https://modelscope.cn/models/ZhipuAI/chatglm3-6b/summary)

### 带参数评估
```shell
python evalscope/run.py --model ZhipuAI/chatglm3-6b --template-type chatglm3 --model-args revision=v1.0.2,precision=torch.float16,device_map=auto --datasets mmlu ceval --use-cache true --limit 10
```
```shell
python evalscope/run.py --model qwen/Qwen-1_8B --generation-config do_sample=false,temperature=0.0 --datasets ceval --dataset-args '{"ceval": {"few_shot_num": 0, "few_shot_random": false}}' --limit 10
```
参数说明：
- --model-args: 模型参数，以逗号分隔，key=value形式
- --datasets: 数据集名称，支持输入多个数据集，使用空格分开，参考下文`数据集列表`章节
- --use-cache: 是否使用本地缓存，默认为`false`;如果为`true`，则已经评估过的模型和数据集组合将不会再次评估，直接从本地缓存读取
- --dataset-args: 数据集的evaluation settings，以json格式传入，key为数据集名称，value为参数，注意需要跟--datasets参数中的值一一对应
  - --few_shot_num: few-shot的数量
  - --few_shot_random: 是否随机采样few-shot数据，如果不设置，则默认为true
- --limit: 每个subset最大评估数据量
- --template-type: 需要手动指定该参数，使得evalscope能够正确识别模型的类型，用来设置model generation config。

关于--template-type，具体可参考：[模型类型列表](https://github.com/modelscope/swift/blob/main/docs/source/LLM/%E6%94%AF%E6%8C%81%E7%9A%84%E6%A8%A1%E5%9E%8B%E5%92%8C%E6%95%B0%E6%8D%AE%E9%9B%86.md)
在模型列表中的`Default Template`字段中找到合适的template；  
可以使用以下方式，来查看模型的template type list：
```shell
from evalscope.models.template import TemplateType
print(TemplateType.get_template_name_list())
```