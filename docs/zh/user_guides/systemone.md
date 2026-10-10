# System One 评测

使用 `eval_type='systemone_api'` 评测只返回选择结果的文本决策模型。EvalScope 将原始题目、上下文、示例和完整选项转换为 System One 请求，再以返回标签计算准确率。

## 百炼示例

```python
import os

from evalscope import TaskConfig, run_task

run_task(TaskConfig(
    model='decision-model-preview',
    eval_type='systemone_api',
    api_url='https://trial.cn-beijing.maas.aliyuncs.com/compatible-mode/v1',
    api_key=os.environ['DASHSCOPE_API_KEY'],
    datasets=['ceval', 'banking77'],
    dataset_args={
        'ceval': {
            'subset_list': ['computer_network'],
            'few_shot_num': 5,
            'system_prompt': '请根据题目和示例选择最合适的答案。',
        },
    },
    generation_config={'timeout': 60, 'retries': 3, 'retry_interval': 2},
    eval_batch_size=8,
    limit=5,
))
```

`api_url` 必须以 `/v1` 结尾，实际请求追加 `/systemone`。百炼也可使用业务空间的域名；以当前[百炼 API 文档](https://help.aliyun.com/zh/model-studio/decision-model-api)为准。TypeSafe Jev 使用相同评测类型，配置其 API 基地址、API key 和固定模型版本。

## TypeSafe Jev 示例

在环境中设置 `TYPESAFE_API_KEY` 后运行：

```python
from evalscope import TaskConfig, run_task

run_task(TaskConfig(
    model='jev-1.13.0',
    eval_type='systemone_api',
    api_url='https://api.typesafe.ai/v1',
    datasets=['boolq'],
    limit=2,
    eval_batch_size=1,
))
```

显式 `api_key` 优先。未指定、设置为 `None`、空字符串或默认占位值 `EMPTY` 时，仅对官方 HTTPS 域名 `api.typesafe.ai` 读取 `TYPESAFE_API_KEY`。百炼和自定义服务地址仍需显式配置 key，TypeSafe 环境 key 不会转发给它们。参见 [TypeSafe API 文档](https://docs.typesafe.ai/api)。

## 提示词与示例

- `system_prompt` 合并到任务指令，协议没有原生 system 消息角色及其优先级。
- `few_shot_num` 保持 benchmark 自身的默认值、取样方式和允许数量；示例放入 `state.examples`。GPQA、SuperGPQA 保留固定示例。
- `dataset_args.<benchmark>.choice_instructions` 可以替换默认的选择任务说明。使用反引号引用 `question`、`examples` 等状态字段。
- 默认生成模板中的输出格式和推理要求不发送给模型。显式生成模板、输出过滤器和 `use_cot` 不兼容此模式。
- 仅配置 timeout、重试和并发等传输参数；temperature、max_tokens、stream 和生成专用参数不适用。

## 支持的 benchmark

`mmlu`、`ceval`、`cmmlu`、`mmlu_pro`、`arc`、`gpqa_diamond`、`hellaswag`、`winogrande`、`general_mcq`、`musr`、`super_gpqa`、`anli`、`boolq`、`banking77`、`contract_nli`、`reward_bench`。

- ARC-C 使用 `arc` 的 `ARC-Challenge` 子集。
- ANLI 的子集为 `r1`、`r2`、`r3`；默认使用各轮 test，示例使用对应 train。
- BoolQ 使用原版 validation，评测两选一 Choice。
- BANKING77 保留全部 77 类；不进行候选项裁剪。
- ContractNLI 使用完整合同的 `contractnli_b`，仅评三分类，不评证据定位。
- RewardBench 使用 v1 filtered 数据，确定性打乱回答位置。总体分数是样本加权 accuracy，不是官方类别加权榜单分数。

新增的五个 benchmark 均默认从 ModelScope 下载，ContractNLI 使用 `evalscope/contract-nli` 的 `contractnli_b` 配置。可通过各自的 `local_path` 和 `dataset_revision` 配置本地数据或固定数据版本。图片、音频、视频、多选题、工具调用和 LLM judge 不支持。

## 结果与复现

System One 沿用 `Model.generate(messages)` 调用入口。MCQ adapter 在现有 `ChatMessage.internal` 中附带原题、候选项和示例等转换信息，`SystemOneAPI.generate()` 将消息转换为协议请求。无需新增样本或模型输出字段；没有 MCQ 转换信息的任意聊天文本不能直接转换。

返回标签转换为 benchmark 原有答案格式，例如 `ANSWER: A`、`答案：A` 或 `ANSWER: 51`，再通过原有答案提取、评分、缓存和报告流程。

预测 JSONL 的 `model_output.metadata.choice_response.answers.answer` 保存原始标签、概率与厂商置信度；`choice_request` 保存完整实际请求。厂商将概率保留到两位小数时，校验允许对应舍入误差，保存的概率不重新归一化。消息文本展示题目和选项，system 消息及 `internal` 保留转换所需的指令、上下文与示例。

配置和原始请求保留 Choice 指令及 few-shot 设置；执行统计区分成功、失败和未完成。无效响应不计为错误答案，默认终止运行；`ignore_errors=True` 时排除失败样本并报告覆盖情况。

固定模型版本、数据版本、split、示例数量和任务指令后再比较成绩。缓存复用和 `rerun_review` 沿用 EvalScope 现有规则；复用预测时，实际模型输入以保存的原始请求为准。模型别名不能保证远端模型版本稳定。

参考：[TypeSafe API](https://docs.typesafe.ai/api)、[公开 Jev 评测代码](https://github.com/AppliedMachineLearning-Lab/jev-benchmarking)。不同 split、shots、推理模式或汇总指标的榜单分数不能直接对照。
