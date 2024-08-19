# 基本使用

## 简单评估
在指定的若干数据集上使用默认配置评估某个模型，流程如下：

`````{tabs}
````{tab} 使用pip安装

可在任意路径下执行：
```bash
python -m evalscope.run \
 --model qwen/Qwen2-0.5B-Instruct \
 --template-type qwen \
 --datasets arc 
```
如遇到 `Do you wish to run the custom code? [y/N]` 请键入 `y`
````

````{tab} 使用源码安装

在`evalscope`路径下执行：
```bash
python evalscope/run.py \
 --model qwen/Qwen2-0.5B-Instruct \
 --template-type qwen \
 --datasets arc
```
如遇到 `Do you wish to run the custom code? [y/N]` 请键入 `y`
````
`````
### 基本参数说明
- `--model`: 指定了模型在[ModelScope](https://modelscope.cn/)中的`model_id`，可自动下载，例如[Qwen2-0.5B-Instruct模型链接](https://modelscope.cn/models/qwen/Qwen2-0.5B-Instruct/summary)；也可使用模型的本地路径，例如`/path/to/model`
- `--template-type`: 指定了模型对应的模板类型，参考[模板表格](https://swift.readthedocs.io/zh-cn/latest/LLM/%E6%94%AF%E6%8C%81%E7%9A%84%E6%A8%A1%E5%9E%8B%E5%92%8C%E6%95%B0%E6%8D%AE%E9%9B%86.html#id4)中的`Default Template`字段填写
    ````{note}
    也可以使用以下方式，来查看模型的`template_type`列表: 
    ``` python
    from evalscope.models.template import TemplateType
    print(TemplateType.get_template_name_list())
    ```
    ````
- `--datasets`: 数据集名称，支持输入多个数据集，使用空格分开，数据集将自动下载，支持的数据集参考[数据集列表](#支持的数据集列表)


## 带参数评估
若想进行更加自定义的评估，例如自定义模型参数，或者数据集参数，可以使用以下命令：

**示例1：**
```shell
python evalscope/run.py \
 --model qwen/Qwen2-0.5B-Instruct \
 --template-type qwen \
 --model-args revision=v1.0.2,precision=torch.float16,device_map=auto \
 --datasets mmlu ceval \
 --use-cache true \
 --limit 10
```

**示例2：**
```shell
python evalscope/run.py \ 
 --model qwen/Qwen2-0.5B-Instruct \
 --template-type qwen \
 --generation-config do_sample=false,temperature=0.0 \
 --datasets ceval \
 --dataset-args '{"ceval": {"few_shot_num": 0, "few_shot_random": false}}' \
 --limit 10
```

### 参数说明
除开上述三个[基本参数](#基本参数说明)，其他参数如下：
- `--model-args`: 模型加载参数，以逗号分隔，key=value形式
- `--generation-config`: 生成参数，以逗号分隔，key=value形式
  - `do_sample`: 是否使用采样，默认为`false`
  - `max_new_tokens`: 生成最大长度，默认为1024
  - `temperature`: 采样温度
  - `top_p`: 采样阈值
  - `top_k`: 采样阈值
- `--use-cache`: 是否使用本地缓存，默认为`false`；如果为`true`，则已经评估过的模型和数据集组合将不会再次评估，直接从本地缓存读取
- `--dataset-args`: 评估数据集的设置参数，以json格式传入，key为数据集名称，value为参数，注意需要跟`--datasets`参数中的值一一对应
  - `--few_shot_num`: few-shot的数量
  - `--few_shot_random`: 是否随机采样few-shot数据，如果不设置，则默认为`true`
- `--limit`: 每个数据集最大评估数据量，不填写则默认为全部评估，可用于快速验证


## 支持的数据集列表
```{note}
目前框架支持如下数据集，若您需要的数据集不在列表中，请提交issue，或者使用[OpenCompass backend](../user_guides/opencompass_backend.md)进行评估；或使用[VLMEvalKit backend](../user_guides/vlmevalkit_backend.md)进行多模态模型评估
```

| DatasetName        | Link                                                                                   | Status | Note |
|--------------------|----------------------------------------------------------------------------------------|--------|------|
| `mmlu`             | [mmlu](https://modelscope.cn/datasets/modelscope/mmlu/summary)                         | Active |      |
| `ceval`            | [ceval](https://modelscope.cn/datasets/modelscope/ceval-exam/summary)                  | Active |      |
| `gsm8k`            | [gsm8k](https://modelscope.cn/datasets/modelscope/gsm8k/summary)                       | Active |      |
| `arc`              | [arc](https://modelscope.cn/datasets/modelscope/ai2_arc/summary)                       | Active |      |
| `hellaswag`        | [hellaswag](https://modelscope.cn/datasets/modelscope/hellaswag/summary)               | Active |      |
| `truthful_qa`      | [truthful_qa](https://modelscope.cn/datasets/modelscope/truthful_qa/summary)           | Active |      |
| `competition_math` | [competition_math](https://modelscope.cn/datasets/modelscope/competition_math/summary) | Active |      |
| `humaneval`        | [humaneval](https://modelscope.cn/datasets/modelscope/humaneval/summary)               | Active |      |
| `bbh`              | [bbh](https://modelscope.cn/datasets/modelscope/bbh/summary)                           | Active |      |
| `race`             | [race](https://modelscope.cn/datasets/modelscope/race/summary)                         | Active |      |
| `trivia_qa`        | [trivia_qa](https://modelscope.cn/datasets/modelscope/trivia_qa/summary)               | To be intergrated |      |

