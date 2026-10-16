# 从一次错判开始，用 EvalScope 评测决策模型

在一次真实的 BoolQ 试跑中，决策模型把 Yes 的概率给到了 **86%**。

标准答案却是 No。

这条结果很适合放在评测的起点。如果只看选项，我们知道它答错了一道题；再看概率，就能发现，这还是一次相当有把握的错判。对于准备根据概率自动执行下一步的应用，后者同样值得关心。

把这个问题放回 Agent 的工作流，就更容易理解了。下一步该搜索还是直接回答？材料够不够？任务完成到什么程度？这些判断通常只需要一个选项、一个是非结论，或者一个分数，却会影响后面的行动。

Jev、Clef-Flash 这类决策模型把判断连同概率一起返回。评测也就多了一项任务，既要检查答案，还要检查这些概率是否靠得住。等请求多起来，还得知道服务能否及时完成判断。

这篇实践就沿着这三个问题展开。**用 EvalScope 跑一次评测，翻开一条错题，再测一轮并发。**公开数据和开源模型从 ModelScope 获取，结果可以在 EvalScope Web UI 中交互查看。

有可用的决策 API，就从下一节开始；想自己部署开源模型，可以先跳到[第 5 节](#5-用-vllm-部署-modelscope-上的-clef-flash)。全文沿用 System One 一种接口，更换服务时只需调整连接参数。

> 配图占位 1｜ModelScope 提供模型和数据，推理服务提供决策结果，EvalScope 完成效果评测、报告与性能测试。突出生态中各环节的分工。

## 1. 先让三类判断跑起来

先把要测的东西说清楚。以一个需要查询当天信息的助手为例，三类判断可以放在同一段工作流里理解。

| 工作中的判断 | 题型 | 模型返回什么 |
|---|---|---|
| 下一步搜索、计算，还是直接回答 | Choice，选择题 | 各选项的概率 |
| 当前是否已经拿到所需信息 | Noul，是非题 | 条件成立的概率 |
| 已完成了多少项要求 | Score，分级评分 | 各等级的概率，以及由此得到的期望分数 |

Score 的分数可以是小数。比如模型把概率分配在两个相邻等级上，最后的期望分数就可能落在它们之间。

下面用三组公开数据把这三条路径一起跑通。这里评估的是单次文本判断，完整 Agent 的工具执行和任务成功率需要另行评测。

### 安装 EvalScope

在包含本页决策评测功能的 [EvalScope 源码](https://github.com/modelscope/evalscope)根目录执行以下命令。后文的新终端也进入同一目录并激活环境。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[perf,service]'
```

### 连接一个决策服务

准备好模型名、服务地址和 API Key，就能开始。下面以 DashScope 托管的 `decision-model-preview` 为例，地域、业务空间和调用权限见[服务文档](https://help.aliyun.com/zh/model-studio/decision-model-api)。已有其他 System One 服务，替换这三项即可。

```bash
export DECISION_MODEL='decision-model-preview'
export DECISION_API_URL='https://trial.cn-beijing.maas.aliyuncs.com/compatible-mode/v1'
export DECISION_API_KEY='替换为所用服务的 API Key'
```

将下面代码保存为 `evaluate.py`，执行 `python evaluate.py`。

```python
import os

from evalscope import TaskConfig, run_task

run_task(TaskConfig(
    model=os.environ['DECISION_MODEL'],
    eval_type='systemone_api',
    api_url=os.environ['DECISION_API_URL'],
    api_key=os.environ['DECISION_API_KEY'],
    datasets=['arc', 'boolq', 'stsb'],
    dataset_args={
        'arc': {'subset_list': ['ARC-Challenge']},
        'boolq': {'extra_params': {'decision_kind': 'noul'}},
    },
    generation_config={'timeout': 300, 'retries': 1},
    eval_batch_size=1,
    limit=5,
))
```

运行后，EvalScope 会从 ModelScope 加载数据，向服务发送问题，再用标准答案评分。标准答案不会发给模型。

这段配置中，`datasets` 选定要测的能力，`eval_type='systemone_api'` 选定接口。ARC-Challenge 对应 Choice，BoolQ 显式切换到 Noul，STS-B 对应 0–5 分的 Score。以后更换模型时，可以保留这套题和评分方式。

先保留 `limit=5`，确认三类请求都能得到有效结果，再删掉限制跑完整评测。这个限制按每个子集生效，选择多个子集时，请求数可能超过 5。`retries=1` 表示每条样本只尝试一次；不用额外添加 `temperature`、`max_tokens` 或流式生成参数。

## 2. 报告出来了，先看哪几项

评测结束后，在 EvalScope Web UI 里查看结果。从源码安装时，首次使用需先安装 Node.js 并构建页面；在源码根目录执行：

```bash
make web-build
evalscope service --host 127.0.0.1 --port 9000 --outputs ./outputs
```

打开 `http://127.0.0.1:9000`，进入「评估」，点击模型报告。先检查请求成功数、有效评分数量和覆盖情况，确认题目确实测完了，再读分数。

用前面的配置进行一次真实试跑，得到过下面这组结果。模型为 `decision-model-preview`，每组只有 5 道题，适合说明报告怎么看，不能据此判断模型的整体水平；重新运行也可能得到不同数值。

| 数据集 | 有效评分数量 | 主指标 | 读法 |
|---|---|---|---|
| ARC-Challenge | 5 | Accuracy 80% | 5 道选择题答对 4 道 |
| BoolQ，Noul 模式 | 5 | Accuracy 80% | 5 道是非题答对 4 道 |
| STS-B | 5 | MAE 0.564 | 在 0–5 分量表上，平均相差 0.564 分 |

你可以在「预测浏览」中切换数据集、筛选错题，检查自己的结果。以这次 BoolQ 试跑为例，单题面板里 Yes 是 86%，No 是 14%，标准答案是 No。概率条让这个错误比一个总分更直观。

下方「评估结果」给出单题得分。需要追查请求或概率处理时，再展开「原始评分数据」，或切换原始概率与评分用概率。输出目录中的 `predictions/`、`reviews/` 和 HTML 汇总也会保留，方便复查。

> 配图占位 2｜真实试跑的报告与 BoolQ 错题，标出有效评分数量、Yes 86% 和标准答案 No。统一补图时使用对应实测记录。

翻过这条错题，再看几个指标，就容易理解它们各自解决什么问题了。

**Accuracy 记录答案是否正确。** 开头那道题无论给 Yes 51% 还是 86%，都会被记为答错。准确率适合看总体答题表现，却无法区分这两种概率。

**NLL、Brier 和 ECE 帮助检查概率。** 对同一道标准答案为 No 的题，把 Yes 的概率从 51% 提到 86%，NLL 和 Brier 都会变差。它们会惩罚自信的错误，越低越好。ECE 则检查概率和实际频率是否接近。Choice 给所选答案八成概率时，看这组答案是否约八成正确；Noul 给 Yes 八成概率时，看这组题的条件是否约八成成立。单次错判并不证明概率失准，需要足够多的样本才能看出规律，这 5 道题还不够。

**MAE 衡量评分偏差。** STS-B 的标准答案本身可以是小数。EvalScope 用模型的期望分数与它比较，保留这种差别。上表的 0.564 就是平均误差 0.564 分，越小越好。

再往业务里走一步。你可能愿意自动接受部分判断，把剩余情况交给人工。「详情」中的风险—覆盖率交互可以帮助探索这件事，选择题型和概率来源，再调整接收覆盖范围，观察留下的预测有多少错误。AURC 汇总这条曲线，通常越低越好；Score 的风险使用按量表范围归一化的评分误差。

选项概率与服务另行返回的 `confidence` 可能含义不同，报告会分开展示。真正设定业务门槛时，还需要用有代表性的独立数据验证，不能只凭这次小样本试跑。

> 配图占位 3｜准确率相近、概率表现不同的示意图，旁边放风险—覆盖率示意。明确标注示意数据，避免与实测混淆。

需要时，再查这些辅助指标。

| 指标 | 帮你看什么 | 更好的方向 |
|---|---|---|
| macro-F1 | 各固定类别的综合表现，每类同等计权 | 越高越好 |
| Precision / Recall / F1 | Noul 肯定判断的准确程度、正例覆盖和两者的平衡 | 越高越好 |
| AUROC / AUPRC | Noul 对正负样本的区分能力 | 越高越好；跨数据集慎比 |
| Spearman / Pearson | 评分排序和分数变化是否与标准答案一致 | 越接近 1 越好 |
| nRPS | 离散 Score 的概率是否集中在正确等级附近 | 越低越好 |

样本太少、只有一个类别或评分没有变化时，有些指标无法计算。先看报告给出的原因，缺项不等于零分。二分类和多分类 Brier、不同 Score 档位的误差也不宜混在一起比较。

## 3. 从 ModelScope 选择评测数据

三类请求都跑通之后，可以开始选题了。关心科学推理、意图识别，还是更综合的决策能力，选用的数据会不同。

下面这些 benchmark 已经接入 EvalScope。填写左侧的名称，框架就会加载对应数据、构造问题并计算指标，无需自己逐一编写请求。

| Benchmark | 测什么 |
|---|---|
| `arc` | 科学知识和推理，本文使用 ARC-Challenge 选择题 |
| `boolq` | 阅读短文后的是非判断，本文使用 Noul |
| `stsb` | 两句话的语义相似度，按 0–5 分评价 |
| `jevbench` | Choice、Noul、Score 三类综合决策，共 231 道公开题 |
| `open_decisions` | 三类综合决策，共 2,352 道题，评估 Intelligence、Calibration 和 Capability |
| `banking77` | 77 类银行业务意图识别 |
| `reward_bench` | 比较两个回答，选出更好的一个 |
| `general_decision` | 自己业务中的选择、是非判断和分级评分 |

[JevBench](https://modelscope.cn/datasets/evalscope/jevbench)和[OpenDecisions](https://modelscope.cn/datasets/evalscope/OpenDecisions)已在 ModelScope 提供。数据来源、评分定义及服务能力要求，可在文末的详细使用文档中查阅。

在已跑通的 `evaluate.py` 中，将数据设置替换为下面两行，就能先试跑综合决策题。

```python
datasets=['jevbench', 'open_decisions'],
dataset_args={},
```

保留 `limit=5` 完成试跑，再删除限制。公共题测的是决策能力的一部分，不能直接当作 Agent 完整任务的成功率。RewardBench 的样本加权成绩也不等于官方分类别加权榜单成绩。

## 4. 把判断换成自己的业务

公共题给了我们一把共同的尺子。要判断模型是否适合自己的应用，还得把它每天遇到的问题放进来。

就用前面那个助手的工作流。选择下一步能力、判断信息是否已经取得、检查答案覆盖了几项要求，都可以写成一道有明确参考答案的题。

创建目录 `decision-data`，将下面三个示意样本保存为 `decision-data/default_test.jsonl`，每行一道题。

```json
{"id":"route","state":"用户要查询今天的天气，当前还没有联网查询结果。","question":{"type":"choice","instructions":"下一步使用哪类能力？","criteria":{"search":"查询外部信息","calculate":"进行数值计算","answer":"根据已有信息回答"}},"expected":"search"}
{"id":"ready","state":"用户要查询今天的天气，当前还没有联网查询结果。","question":{"type":"noul","instructions":"当前是否已经取得今天的天气信息？"},"expected":"no"}
{"id":"complete","state":"任务要求列出价格、库存和购买链接。当前答案只给出了价格。","question":{"type":"score","instructions":"按已经覆盖的要求数量评价完成度。","criteria":["未覆盖任何要求","覆盖一项","覆盖两项","三项都覆盖"]},"expected":"1"}
```

每行只有三个核心部分。`state` 给模型看材料，`question` 说明要做什么判断，`expected` 留给评测器核对答案，不会发给模型。

Choice 的答案填选项 ID，Noul 填 `yes` / `no`。Score 填从 0 开始的等级编号，上例只覆盖了一项要求，所以标准答案是 `1`。把标准写清楚，后续换模型、调提示或更新服务，才有同一把尺子可以复测。

在同一个评测脚本中替换数据设置即可，本地文件无需上传。

```python
datasets=['general_decision'],
dataset_args={'general_decision': {'local_path': './decision-data'}},
```

这三个样本用来检查格式和流程，不能据此判断模型好坏。正式评测需要代表真实业务的独立样本，保留容易混淆和信息不足的情况。

完整字段、多子集和连续标准分数的写法，见[自定义决策数据集](../advanced_guides/custom_dataset/decision.md)。

## 5. 用 vLLM 部署 ModelScope 上的 Clef-Flash

前面的评测不要求自己部署模型。如果你希望把服务放在自己的机器上，可以使用 ModelScope 上的 [Cloudflare/clef-flash](https://modelscope.cn/models/Cloudflare/clef-flash)。它基于 Qwen3.5-9B，约 90 亿参数，支持本文的三类决策；这里使用文本输入。

部署完成后，仍然运行同一个 `evaluate.py`。这样对比线上服务和自己的模型时，数据与评分方式能够保持一致。

以下部署面向 Linux / NVIDIA GPU，需要足够的显存。目前 [vLLM 最新开发文档](https://docs.vllm.ai/en/latest/models/structured_decisions_models/)已列出 Clef-Flash 的原生 System One 支持。请安装包含该实现的开发版本；旧版仅能加载底座，并不代表已经加载了决策头。

```bash
pip install -U modelscope
pip install -U --pre vllm --extra-index-url https://wheels.vllm.ai/nightly
VLLM_USE_MODELSCOPE=true vllm serve Cloudflare/clef-flash \
  --served-model-name clef-flash \
  --max-model-len 16384 --port 8000
```

`VLLM_USE_MODELSCOPE=true` 会让 vLLM 自动从 ModelScope 下载模型，不必另写下载脚本。保持服务终端运行，在另一个已激活环境的终端切换连接参数。

```bash
export DECISION_MODEL='clef-flash'
export DECISION_API_URL='http://127.0.0.1:8000/v1'
export DECISION_API_KEY='EMPTY'
```

先检查服务能否完成一个判断。

```bash
curl "$DECISION_API_URL/systemone" \
  -H 'Content-Type: application/json' \
  -d '{"model":"clef-flash","state":"The sky is blue.","questions":{"answer":{"type":"choice","instructions":"What color is the sky?","criteria":{"blue":"Blue","red":"Red"}}}}'
```

返回应包含 `answers.answer.choice` 和按选项 ID 对应的 `probabilities`。数值以实际响应为准。

再执行 `python evaluate.py`。模型名、地址和 key 改了，EvalScope 的 `eval_type='systemone_api'` 与前面的评测配置可以继续使用。`EMPTY` 表示这个本地示例未配置鉴权。

**部署验证范围**：上述命令依据 vLLM 当前开发文档整理，尚未在 NVIDIA GPU 上完成本项目的部署验收；此前 Clef-Flash 的真实效果验证使用官方 PyTorch 推理。二者不能混为同一次验证。正式比较时记录模型 revision、vLLM 版本、启动参数，并核对长输入是否被截断。

## 6. 请求多起来以后，还能及时完成判断吗

一条请求能及时返回，不代表十几条一起到来时也一样。如果判断放在 Agent 的必经步骤上，排队时间就会直接进入用户的等待时间。

继续用 EvalScope 测两件事，一次判断要等多久，服务每秒能完成多少次判断。这里统计完整的决策请求。

先停止同一服务上的效果评测，固定模型配置和测试输入。下面的命令继续使用当前终端的 `DECISION_*` 参数，适用于兼容的线上或本地推理服务。线上测试需遵守服务配额，先从小规模开始。

导出 ARC-Challenge Choice、BoolQ Noul 和 STS-B Score 三组请求。每组默认 20 个不同输入，压测时循环取用。

```bash
python examples/decision/prepare_perf_requests.py \
  --api systemone --model "$DECISION_MODEL" \
  --choice-benchmark arc --output ./perf-input/systemone
evalscope perf --api systemone --model "$DECISION_MODEL" \
  --url "$DECISION_API_URL" --api-key "$DECISION_API_KEY" \
  --dataset line_by_line --dataset-path ./perf-input/systemone/choice.jsonl \
  --parallel 1 --number 20 --warmup-num 2 --no-stream \
  --read-timeout 300 --outputs-dir ./perf-results/smoke
```

成功后再分别测试三类任务和不同并发。

```bash
for kind in choice noul score; do
  evalscope perf --api systemone --model "$DECISION_MODEL" \
    --url "$DECISION_API_URL" --api-key "$DECISION_API_KEY" \
    --dataset line_by_line --dataset-path "./perf-input/systemone/$kind.jsonl" \
    --parallel 1 4 8 16 --number 1000 1000 1000 1000 \
    --warmup-num 20 --no-stream --read-timeout 300 \
    --outputs-dir "./perf-results/$kind"
done
```

每种题型分别在 1、4、8、16 并发下，先预热 20 次，再正式测 1,000 次。切换模型后重新导出请求，避免文件中仍保留旧模型名。这是一套可复制的测试配置，实际性能需要在自己的服务上测量。

在 Web UI 的「性能」页选择 `perf-results` 目录，打开运行记录，就能比较不同并发的图表，也能翻看逐请求记录；输出目录还会保留 `perf_report.html`。

先看成功率和失败数。请求失败很多时，即使返回很快，也不能说明服务可用。接着看 p50／p90／p95／p99 延迟和成功 req/s。例如，p95 为 0.3 秒，表示约 95% 的成功请求在 0.3 秒内完成；这里每请求一道题，成功 req/s 也就是每秒成功完成的决策数。

并发升高时，如果吞吐几乎不变，延迟却持续变长，通常是在排队。比较线上和本地结果时，还要留意网络、请求长度、缓存和测量位置，不能只拿一个吞吐数字排名。

> 配图占位 4｜同一模型在不同并发下的成功吞吐与 p95 延迟曲线，附输入长度、缓存和运行配置，实测后补图。

HTTP 200 但决策结果非法，也会记为失败。请求延迟包含客户端 HTTP 请求中的网络、排队和服务执行，不是单独的模型计算时间。本文使用非流式决策，不看逐字生成的 TTFT、TPOT、ITL。更多参数见[性能评测快速开始](../user_guides/stress_test/quick_start.md)。

## 7. 需要业务定制时，接上 ms-swift

当错题开始集中在某一类业务判断上，这批评测数据也能帮助确定训练方向。完成训练后，保留同一套独立测试集再跑一次，检查改进是否落到了原来的问题上。

[ms-swift](https://swift.readthedocs.io/zh-cn/latest/BestPractices/Decision.html)支持决策模型训练，已有 JEV-27B-VL、Cloudflare/clef、OmniJev 和通用 CausalLM 路径。训练后的模型可通过 Transformers 加载并推理，再适配成兼容决策服务，交给 EvalScope 复评。

这里的 `Cloudflare/clef` 与 `clef-flash` 是不同模型。ms-swift 训练 checkpoint 到 vLLM System One 的导出和部署，需要单独适配与验收，本文不把它描述为已经跑通的一键流程。训练、校准和评测数据也应分开，避免把记住的样本当成泛化能力。

## 遇到问题时

| 现象 | 先检查什么 |
|---|---|
| `401` / `403` 或模型不可用 | API Key、地域、业务空间及模型调用权限 |
| `404` / `501` | 是否使用 `/v1/systemone`，vLLM 是否包含该模型的决策策略 |
| Choice 选项超过限制 | 服务真实上限；完整 BANKING77 需要 77 项，不裁剪题目 |
| Score 档位不支持 | 保留原始等级，选择兼容服务 |
| 有分数但评分数量不足 | 请求失败、非法响应和覆盖率，查看 predictions／reviews |
| 第一次只想确认能跑 | 质量评测保留 `limit=5`；perf 先用 20 次请求 |

## 接口补充

[TypeSafe Jev](https://api.typesafe.ai/redoc)同样使用 `systemone_api`。服务地址设为 `https://api.typesafe.ai/v1`，模型名从账号的 `/v1/models` 查询，API Key 换成该服务的 key。EvalScope 的 benchmark、评分和性能测试用法保持一致；不同模型的能力和使用限制仍需分别确认。

AutoTrust JEV 的 `/v1/decide` 对应 `autotrust_api`，示例服务见 `examples/decision/serve_autotrust.py`。其 Score 固定六档，不能直接评完整 JevBench／OpenDecisions。普通 chat 模型使用 `openai_api`，仍可运行 ARC、BANKING77、默认 Choice 的 BoolQ、RewardBench 来测答案正确率；聊天中自报的把握不能替代候选概率。

## 继续使用 EvalScope

做到这里，你已经可以检查一条判断的答案和概率，换上自己业务的题，再比较请求增多后的响应情况。回到开头那道 Yes 概率 86% 的错题，我们现在能把它找出来、看清楚，也能在下一次模型更新后用同一套数据复测。

继续扩展时，按需要查阅下面的文档。

- [EvalScope 使用文档](https://evalscope.readthedocs.io/zh-cn/latest/)
- [决策模型 API 与配置](../user_guides/systemone.md)
- [评测数据集](../get_started/supported_dataset/llm.md)
- [Web 服务与结果查看](../user_guides/service.md)
- [性能评测快速开始](../user_guides/stress_test/quick_start.md)
