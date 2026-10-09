# System One Choice 决策模型

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

新增的五个 benchmark 默认从 Hugging Face 下载；可通过各自的 `dataset_args.dataset_hub`、`local_path` 和 `dataset_revision` 配置数据源。图片、音频、视频、多选题、工具调用和 LLM judge 不支持。

## 结果与复现

预测 JSONL 中的 `model_output.choice_result` 保存标签、概率与厂商置信度；`model_output.metadata` 保存完整请求、响应和请求哈希。厂商将概率保留到两位小数时，校验允许对应舍入误差，保存的概率不重新归一化。

配置和报告记录 Choice 协议与实际 shots；执行统计区分成功、失败和未完成。无效响应不计为错误答案，默认终止运行；`ignore_errors=True` 时排除失败样本并报告覆盖情况。

固定模型版本、数据版本、split、示例数量和任务指令后再比较成绩。普通缓存只复用请求一致的预测；`rerun_review=True` 遇到请求变化会拒绝复用。模型别名不能保证远端模型版本稳定。

参考：[TypeSafe API](https://docs.typesafe.ai/api)、[公开 Jev 评测代码](https://github.com/AppliedMachineLearning-Lab/jev-benchmarking)。不同 split、shots、推理模式或汇总指标的榜单分数不能直接对照。
