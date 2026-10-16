# 用 EvalScope 评测决策模型，从效果到性能

一个 Agent 准备结束任务时，需要做一个很小、却很关键的判断。

任务真的完成了吗？

同样的判断还会出现在它选择搜索工具、检查信息是否充分、给结果打分的时候。这些步骤往往只需要一个选项、一个判断或一个分数。执行得是否准确、概率能否信任、请求多起来会不会排队，都直接影响后面的流程。

Jev 带动了大家对决策模型的关注。拿到一个模型之后，开发者还需要回答几个实际问题：它擅长哪些判断，给出的概率能否作为依据，以及业务请求多起来以后要等多久。

这篇实践围绕 **EvalScope 的使用流程**展开。模型和公开数据从 ModelScope 准备，EvalScope 负责组织请求、评分、生成报告和性能测试。你可以连接已有决策服务，也可以部署开源模型 Clef-Flash；更换服务时沿用同一套 System One 评测配置。

> 配图占位 1｜ModelScope 提供模型和数据，推理服务提供决策结果，EvalScope 完成效果评测、报告与性能测试。突出生态中各环节的分工。

## 1. 先让三类判断跑起来

假设 Agent 收到一条请求，需要查找今天的信息。它要选择下一步调用的工具，这是 **Choice**。检查当前材料是否已经足够，是 **Noul**。按事先约定的等级评价结果完成度，是 **Score**。

Choice 返回各候选项的概率，Noul 返回条件成立的概率，Score 返回各等级的概率。Score 的预测分数由等级概率加权得到，可以落在两个等级之间。

EvalScope 会保留这些结果，评估答案和概率。这里测的是单次文本决策；完整 Agent 的工具执行和任务成功率需要另行评测。

### 安装 EvalScope

在包含本页决策评测功能的 [EvalScope 源码](https://github.com/modelscope/evalscope)根目录执行以下命令。后文的新终端也进入同一目录并激活环境。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[perf,service]'
```

### 连接一个决策服务

EvalScope 需要模型名、服务地址和 API Key。下面以百炼托管的 `decision-model-preview` 为连接示例；使用自己的可用模型和接入地址即可，具体地域、业务空间和权限见[服务文档](https://help.aliyun.com/zh/model-studio/decision-model-api)。已有兼容服务可直接替换这三项；需要部署开源模型时，先按第 3 节准备服务。

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

这一次运行就覆盖三类题型。ARC-Challenge 测选择，BoolQ 测是非判断，STS-B 测 0–5 分的句子相似度。数据默认从 ModelScope 加载，标准答案只用于评分，不会发给模型。

`limit=5` 用于确认流程。跑通后删掉它，再进行完整评测。它按每个子集生效，多子集 benchmark 的请求数可能超过 5。

`retries=1` 表示每条样本只尝试一次。决策配置中不用添加 `temperature`、`max_tokens` 或流式生成参数。

## 2. 报告出来了，先看哪几项

评测结束后，在 EvalScope Web UI 里查看结果。从源码安装时，首次使用需先安装 Node.js 并构建页面；在源码根目录执行：

```bash
make web-build
evalscope service --host 127.0.0.1 --port 9000 --outputs ./outputs
```

打开 `http://127.0.0.1:9000`，进入「评估」，点击模型报告。「详情」可查看各指标，并选择题型、概率来源和接收覆盖范围，查看对应风险；「预测浏览」支持切换数据集、筛选样本和逐条翻看概率分布。原始概率与评分用概率可以切换查看。

终端打印的输出目录还保留了 `predictions/`、`reviews/` 和 HTML 汇总，方便保存和复查。

先检查请求成功数、有效评分数量和覆盖情况，再读分数。只完成一部分题的高分，不能代表整套测试。

> 配图占位 2｜真实报告截图，标出主指标、有效评分数量和样本详情入口。截图在实际运行后补充。

初次阅读可以先抓住三组指标。

**答案有没有选对。** Choice、Noul 看 Accuracy。比如一百道题答对八十道，准确率就是 80%。固定类别任务还可以结合 macro-F1，观察少数类别是否被忽略。

**给出的概率是否可靠。** NLL、Brier 会惩罚自信的错误，越低越好。ECE 则按概率分组，检查给出八成概率的一组预测，实际是否接近八成正确。它反映一组样本的统计表现，不能保证某一次判断正确，也需要和准确率一起看。

**评分离标准答案多远。** Score 看 MAE。STS-B 使用 0–5 分，MAE 为 0.6 表示预测的期望分数平均离标准分数约 0.6 分。它保留连续标准答案，不强行取整计算分类准确率。

如果你的流程会优先接受有把握的答案，再转人工处理其他情况，可以继续看 AURC 和风险—覆盖率数据。它们帮助你观察提高接收门槛之后，剩余预测的错误风险和覆盖范围怎样变化。选项概率与服务额外返回的 `confidence` 可能含义不同，报告分别标记这两种依据。

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

## 3. 从 ModelScope 下载开源模型并部署

需要自己部署时，可以从 ModelScope 下载 [Cloudflare/clef-flash](https://modelscope.cn/models/Cloudflare/clef-flash)。设置 `VLLM_USE_MODELSCOPE=true` 后，vLLM 会自动从 ModelScope 下载并加载模型，再交给 EvalScope 评测。

Clef-Flash 是基于 Qwen3.5-9B 的约 90 亿参数模型，使用联合决策头输出候选概率。本文使用文本输入。

以下部署面向 Linux / NVIDIA GPU，需要足够的显存。目前 [vLLM 最新开发文档](https://docs.vllm.ai/en/latest/models/structured_decisions_models/)已列出 Clef-Flash 的原生 System One 支持。请安装包含该实现的开发版本；旧版仅能加载底座，并不代表已经加载了决策头。

```bash
pip install -U modelscope
pip install -U --pre vllm --extra-index-url https://wheels.vllm.ai/nightly
VLLM_USE_MODELSCOPE=true vllm serve Cloudflare/clef-flash \
  --served-model-name clef-flash \
  --max-model-len 16384 --port 8000
```

保持这个终端运行，在另一个已激活环境的终端切换连接参数。

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

## 4. 从 ModelScope 选择评测数据

前面的三组数据适合入门。EvalScope 将 benchmark 名称对应到 ModelScope 上的数据，并按任务要求构造请求、读取标准答案和计算指标。扩展评测时，按目标调整数据配置即可。

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

## 5. 把判断换成自己的业务

回到开头的 Agent。公共测试跑通之后，更有用的一步，是把它真正遇到的选择、完成度判断和评分标准整理成数据。

创建目录 `decision-data`，将下面三个示意样本保存为 `decision-data/default_test.jsonl`，每行一道题。

```json
{"id":"route","state":"用户要查询今天的天气，当前还没有联网查询结果。","question":{"type":"choice","instructions":"下一步使用哪类能力？","criteria":{"search":"查询外部信息","calculate":"进行数值计算","answer":"根据已有信息回答"}},"expected":"search"}
{"id":"ready","state":"用户要查询今天的天气，当前还没有联网查询结果。","question":{"type":"noul","instructions":"当前是否已经取得今天的天气信息？"},"expected":"no"}
{"id":"complete","state":"任务要求列出价格、库存和购买链接。当前答案只给出了价格。","question":{"type":"score","instructions":"按已经覆盖的要求数量评价完成度。","criteria":["未覆盖任何要求","覆盖一项","覆盖两项","三项都覆盖"]},"expected":"1"}
```

`state` 是材料，`question` 定义问题，`expected` 是只用于评分的标准答案。Choice 填选项 ID，Noul 填 `yes` / `no`，Score 填从 0 开始的等级编号。给真实业务准备数据时，先把判定标准写清楚，再由人工或可靠规则给出参考答案。

在同一个评测脚本中替换数据设置即可，本地文件无需上传。

```python
datasets=['general_decision'],
dataset_args={'general_decision': {'local_path': './decision-data'}},
```

这三个样本用来检查格式和流程，不能据此判断模型好坏。正式评测需要代表真实业务的独立样本，保留容易混淆和信息不足的情况。

完整字段、多子集和连续标准分数的写法，见[自定义决策数据集](../advanced_guides/custom_dataset/decision.md)。

## 6. 请求多起来以后，还能及时完成判断吗

效果评测回答选得怎么样，性能测试回答一次判断要等多久、服务每秒能完成多少次判断。

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

每种题型分别在 1、4、8、16 并发下，先预热 20 次，再正式测 1,000 次。切换模型后重新导出请求，避免文件中仍保留旧模型名。

在 Web UI 的「性能」页选择 `perf-results` 目录，打开运行记录，可切换图表、查看不同并发结果和逐请求记录；输出目录也保留 `perf_report.html`。先确认成功率和失败数，再看 p50／p90／p95／p99 延迟，以及成功 req/s。p95 为 0.3 秒，表示约 95% 的成功请求在 0.3 秒内完成；这里每请求一道题，成功 req/s 也就是成功 decisions/s。

并发升高时，如果吞吐几乎不变，延迟却持续变长，通常是在排队。比较线上和本地结果时，还要留意网络、请求长度、缓存和测量位置，不能只拿一个吞吐数字排名。

> 配图占位 4｜同一模型在不同并发下的成功吞吐与 p95 延迟曲线，附输入长度、缓存和运行配置，实测后补图。

HTTP 200 但决策结果非法，也会记为失败。请求延迟包含客户端 HTTP 请求中的网络、排队和服务执行，不是单独的模型计算时间。本文使用非流式决策，不看逐字生成的 TTFT、TPOT、ITL。更多参数见[性能评测快速开始](../user_guides/stress_test/quick_start.md)。

## 7. 需要业务定制时，接上 ms-swift

[ms-swift](https://swift.readthedocs.io/zh-cn/latest/BestPractices/Decision.html)支持决策模型训练，已有 JEV-27B-VL、Cloudflare/clef、OmniJev 和通用 CausalLM 路径。训练后的模型可通过 Transformers 加载并推理，再适配成兼容决策服务，交给 EvalScope 复评。

这里的 `Cloudflare/clef` 与 `clef-flash` 是不同模型。ms-swift 训练 checkpoint 到 vLLM System One 的导出和部署，需要单独适配与验收，本文不把它描述为已经跑通的一键流程。训练、校准和评测数据也应分开，避免把记住的样本当成泛化能力。

最开始，我们只想知道 Agent 能否选对下一步。现在可以把这个判断拆开检查，答案是否正确、概率是否可靠，以及在业务流量下要等多久。把这三项留在报告里，下一次更换模型、修改规则或继续训练时，就有了可以复核的依据。

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

更多参数和完整说明，可按需要查阅：

- [EvalScope 使用文档](https://evalscope.readthedocs.io/zh-cn/latest/)
- [决策模型 API 与配置](../user_guides/systemone.md)
- [评测数据集](../get_started/supported_dataset/llm.md)
- [Web 服务与结果查看](../user_guides/service.md)
- [性能评测快速开始](../user_guides/stress_test/quick_start.md)
