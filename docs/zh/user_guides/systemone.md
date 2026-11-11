# System One 评测

System One 是决策模型使用的一种 API 接口。模型接收待判断的内容和判断规则，直接返回选项、是非概率或等级评分。EvalScope 通过 `eval_type='systemone_api'` 连接这类服务，完成数据加载、请求发送、评分和报告生成。

本页介绍如何连接百炼或 TypeSafe Jev、评测已有选择题基准，以及查看结果。Choice（多选一）、Noul（是非判断）、Score（等级评分）的应用示例、自定义数据和 Clef-Flash 部署，见[决策模型评测最佳实践](../best_practice/decision_evaluation.md)。

## 快速开始：百炼

安装评测和结果浏览组件，并设置百炼 API key：

```bash
pip install -U 'evalscope[service]'
export DASHSCOPE_API_KEY='<your-bailian-api-key>'
```

下面使用 `decision-model-preview` 评测 C-Eval 的计算机网络子集和 BANKING77，各取 5 个样本：

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

`limit=5` 限制每个评测子集的样本数，`few_shot_num=5` 则为每道 C-Eval 题提供 5 个示例，两者作用不同。上面的少量样本用于检查流程，不能代表完整基准的成绩。

`api_url` 填写以 `/v1` 结尾的服务基地址，EvalScope 会自动追加 `/systemone`，不要填写完整请求路径。示例使用试用地址；如果账号使用业务空间域名，请按[百炼 API 文档](https://help.aliyun.com/zh/model-studio/decision-model-api)替换地址，并使用对应的密钥。

## 使用 TypeSafe Jev

TypeSafe Jev 使用相同的评测类型，只需更换模型、地址和密钥。先设置 TypeSafe API key：

```bash
export TYPESAFE_API_KEY='<your-typesafe-api-key>'
```

下面在 BoolQ 上评测两个样本：

```python
import os

from evalscope import TaskConfig, run_task

run_task(TaskConfig(
    model='jev-1.13.0',
    eval_type='systemone_api',
    api_url='https://api.typesafe.ai/v1',
    api_key=os.environ['TYPESAFE_API_KEY'],
    datasets=['boolq'],
    limit=2,
    eval_batch_size=1,
))
```

模型名请使用账号可调用的版本，参见 [TypeSafe API 文档](https://docs.typesafe.ai/api)。比较多次评测时，应使用固定版本，而不是随时可能更新的模型别名。

连接官方 HTTPS 地址 `api.typesafe.ai` 时，也可以省略 `api_key`，由 EvalScope 读取 `TYPESAFE_API_KEY`。百炼和自定义服务地址需要显式传入密钥。

## 配置任务指令和请求参数

已有选择题会被转换为题干、候选项和示例，再发送给模型。可通过 `dataset_args.<基准名>` 调整任务说明：

| 参数 | 作用 |
| --- | --- |
| `system_prompt` | 补充任务指令；会与选择题指令合并，不具有聊天接口中 system 角色的优先级 |
| `few_shot_num` | 设置每道题的示例数，取样方式和允许数量遵循对应基准；不设置时保留基准默认值 |
| `choice_instructions` | 替换默认的选择题指令，可用反引号引用 `question`、`examples` 等输入字段 |

此模式不要求模型生成自然语言答案，默认生成模板中的输出格式和推理要求不会发送给模型。自定义任务说明请使用 `choice_instructions`；不要设置生成用的 `prompt_template`、`few_shot_prompt_template`、`query_template` 或输出 `filters`，也不要启用 `use_cot`。

请求超时和重试放在 `generation_config` 中，并发通过 `eval_batch_size` 控制。上例的 `timeout=60` 表示单次请求超时 60 秒，`retries=3` 表示最多尝试 3 次，`retry_interval=2` 表示两次尝试间隔 2 秒。不支持 `temperature`、`max_tokens` 或流式生成。

## 选择评测基准

以下已有基准支持文本单选题的 Choice 转换：

| 任务 | 基准名称 |
| --- | --- |
| 知识与考试 | `mmlu`、`ceval`、`cmmlu`、`mmlu_pro`、`arc`、`gpqa_diamond`、`super_gpqa` |
| 常识与推理 | `hellaswag`、`winogrande`、`musr` |
| 阅读理解与文本推断 | `anli`、`boolq` |
| 文本分类 | `banking77`、`contract_nli` |
| 回答偏好 | `reward_bench` |
| 自定义选择题 | `general_mcq` |

其中 BoolQ 默认也是两选一的 Choice。如果希望直接评测“是”的概率，可在 `dataset_args` 中为它设置 `extra_params={'decision_kind': 'noul'}`；同一 `extra_params` 中的 `decision_threshold` 控制是非判断阈值，默认 0.5。

原生决策题基准则保留各自的题型和评分规则，使用零样本评测：

| 基准 | 适用场景 |
| --- | --- |
| [OpenDecisions](../benchmarks/open_decisions.md) | 同时评测 Choice、Noul、Score，检查判断能力与概率表现 |
| [JevBench 公开子集](../benchmarks/jevbench.md) | 按公开题及其规则评分，不等同于完整 JevBench 榜单 |
| [STS-B Decision](../benchmarks/stsb.md) | 用 Score 评测句子相似度，保留连续标准分数 |
| [General Decision](../advanced_guides/custom_dataset/decision.md) | 使用自己的 Choice、Noul 或 Score 数据 |

公共数据默认从 ModelScope 下载；可通过各基准的 `local_path` 指定本地数据，或通过 `dataset_revision` 固定数据版本。当前 EvalScope 的 System One 接入仅支持文本决策，不支持图片、音频、视频、多选题、Agent、工具调用或 LLM judge。

```{dropdown} 部分基准的评测口径
- ARC-C 对应 `arc` 的 `ARC-Challenge` 子集。
- ANLI 包含 `r1`、`r2`、`r3`，各轮使用 test 评测、对应 train 提供示例。
- BoolQ 使用原版 validation 数据。
- BANKING77 保留全部 77 类，不裁剪候选项。
- ContractNLI 使用 `evalscope/contract-nli` 的 `contractnli_b` 配置及完整合同，仅评三分类，不评证据定位。
- RewardBench 使用 v1 filtered 数据，确定性打乱回答位置；总体准确率按样本数加权，不是官方榜单的类别加权分数。
- GPQA 和 SuperGPQA 按各自基准规则保留固定示例。
```

## 查看结果

结果默认保存在 `outputs/<时间戳>/`，其中 `reports/` 是汇总报告，`predictions/` 保存模型输出，`reviews/` 保存逐题评分。常见指标包括：

| 指标 | 如何理解 |
| --- | --- |
| `acc` | 答对的比例，越高越好 |
| `nll`、`brier` | 概率预测与标准答案之间的损失，越低越好 |
| `ece` | 按概率分组后，预测概率与对应事件实际发生比例的偏差，越低越好 |
| `mae` | 按概率计算的 Score 分数与标准分数的平均绝对误差，越低越好 |

具体指标取决于题型和基准。例如 BANKING77 还报告 `macro_f1`，STS-B 以连续分数的 MAE 为主；OpenDecisions 的综合指标说明见[最佳实践](../best_practice/decision_evaluation.md)。

启动 Web UI，打开 `http://127.0.0.1:9000`，在“评估”页面选择报告，再从“预测浏览”逐题查看预测、标准答案和概率分布：

```bash
evalscope service
```

选项概率描述模型对各答案的预测分布，`confidence` 则是厂商定义的置信度，含义可能不同。查看概率时，界面可以切换原始分布和评分所用分布；后者可能因舍入误差而做了归一化，原始返回值仍会保留。

```{dropdown} 原始请求和概率保存在哪里
- 预测 JSONL 的 `model_output.metadata.choice_request` 保存实际发送的请求。
- `model_output.metadata.choice_response.answers.answer` 保存服务返回的答案，包括对应题型的概率和可选的厂商置信度。
- 逐题 review 的 `score.metadata.decision` 保存评分所用分布；`score.metadata.probabilities_normalized_for_scoring` 标记是否做了归一化。

需要复现模型输入时，请以保存的实际请求为准，而不是只看界面中的题目文本。
```

## 处理失败与复现评测

请求失败或返回无效响应时，评测默认报错并停止。显式设置 `ignore_errors=True` 后可跳过失败样本继续运行；报告会区分成功、失败和未完成，比较分数时也要检查覆盖情况。失败样本通常不纳入普通指标，但 JevBench 的 `public_accuracy` 会将缺失预测计为错误，具体以基准说明为准。

修复问题后，可将 `use_cache` 指向原输出目录以复用已有结果；只需重新评分时使用 `rerun_review`，详见[参数说明](../get_started/parameters.md)。

比较多次评测时，请固定模型版本、数据版本、评测数据划分、示例数和任务指令。使用不同题目划分、示例数或汇总方式得到的分数，不能直接与公开榜单对照。
