# 让 AI 做判断：用 EvalScope 检验决策模型

一条按规则该保留的评论，模型给了 82% 的折叠概率，预测和正确答案正好相反。

这是我们最近在测的一类模型干出来的事。普通大模型接到问题，往往先写三段分析，再委婉地给出结论；这类模型不写文章，只表态：返回一个类别、一个分数，或一句“是 / 否”，再附上每个选项的概率。这类模型叫决策模型，[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)、[Clef-Flash](https://developers.cloudflare.com/changelog/post/2026-10-01-clef-workers-ai/) 就是代表。需要解释和展开讨论的内容，再交给 Chat 模型不迟。

不过 82% 这个数也提醒了一件事：概率写得信誓旦旦，答案照样可能是错的，拿它干活之前得先验一验。本文用 [**EvalScope**](https://github.com/modelscope/evalscope) 给决策模型出题、收答案、判分，一路检查三件事：题答得对不对、概率能不能用、响应够不够快。先用一套公开题跑通流程，扩展部分再讲测自己的数据和自部署。

## 1. 决策模型能答哪三种题

筛选评论只是决策模型的一种用法。消息分类、内容评分、判断两段文字是否相关，也都可以交给它。这些活儿归到题目上，常见的是三种。

先看一道真实的中文评论分类题，评论节选如下。

> 在家动动手指日赚500+，刷单返现，宝妈学生都能做，名额有限先到先得！

标准答案留给 EvalScope 核对，不会发给模型。下图是这道题的完整题面（节选自 [OpenDecisions](https://modelscope.cn/datasets/evalscope/OpenDecisions)）：

![评论分类题，包含 A、B、C、D 四个选项](https://sail-moe.oss-cn-hangzhou.aliyuncs.com/yunlin/images/evalscope/doc/decision_evaluation/public-question.png)

从广告推广、正常评价、辱骂和闲聊里选一个。这种“多选一”叫 **Choice**，模型返回每个选项的概率，适合消息分类、请求分流。

还有两种同样常用：**Noul** 属于是非判断，只问某个条件成不成立，返回“成立”的概率，比如“这条评论是否该折叠”；**Score** 是分级评分，在几个有高低顺序的等级里打分，适合完成度、干扰程度这类有轻重之分的判断。

出题的关键，是把判断对象、规则和可选答案说清楚：分类型用 Choice，判断条件是否成立用 Noul，按程度打分用 Score。规则要具体：“讨论商品价格可以保留，商家招揽客户应折叠”，比一句“过滤不好的内容”清楚得多。

## 2. 一条命令，跑通公开题评测

这次实测使用 `decision-model-preview`，通过百炼 System One 接口（决策模型通用的请求格式）传入题目、接收判断和概率。兼容这个接口的其他服务也能用同样的命令，模型名、API 地址和 Key 换成自己的即可。

依次执行下面的命令。公开题用 [OpenDecisions](https://modelscope.cn/datasets/evalscope/OpenDecisions)，两千多道题、三种题型都覆盖，EvalScope 会自己下载数据、出题、评分：

```bash
# 安装依赖（含压测和 Web UI 组件）
pip install 'evalscope[perf,service]'

# 连接参数：换成所用服务的模型名、地址和 Key
export DECISION_MODEL='decision-model-preview'
export DECISION_API_URL='https://trial.cn-beijing.maas.aliyuncs.com/compatible-mode/v1'
export DECISION_API_KEY='替换为所用服务的 API Key'

# 跑公开题评测
evalscope eval \
  --model "$DECISION_MODEL" \
  --eval-type systemone_api \
  --api-url "$DECISION_API_URL" \
  --api-key "$DECISION_API_KEY" \
  --datasets open_decisions
```

运行结束后，终端会打印各题型的得分和综合指标。我们从全量题里挑了 59 道内容审核题做实测，下面摘出这份报告里的几项；上面的命令默认评测全量 2352 道题。

| 题型 | 样本数 | 准确率 ↑ | 平均误差 MAE ↓ |
| --- | ---: | ---: | ---: |
| Choice | 22 | 63.6% | — |
| Noul | 19 | 89.5% | — |
| Score（3 档，0–2） | 2 | 100% | 0.015 |
| Score（4 档，0–3） | 16 | 56.3% | 0.6462 |

Choice 取最高概率的选项，Noul 默认以 0.5 为是非分界，准确率就是答对的比例。Score 的准确率看最高概率的等级有没有选对，MAE 则看按概率算出的分数平均偏离标准等级多少，越低越好。不同等级尺度的评分题会分开列出。

同一批 59 道题的综合分如下，三项都是越高越好：

| 综合指标 | 得分 ↑ |
| --- | ---: |
| Intelligence（判断能力） | 39.1% |
| Calibration（概率可靠程度） | 53.9% |
| Capability（综合表现） | 46.5% |

这三个综合指标不能当成准确率读。**Intelligence（判断能力）** 衡量模型比基线好多少：Choice 和 Noul 的基线是随机猜，Score 通常以打中间分为参照。例如，是非题随机猜有 50% 的准确率，做到 80%，换算后的能力分就是 `(80% - 50%) / (100% - 50%) = 60%`。各类题按难度加权，三种题型的能力分取平均，得到本次的 **39.1%**。

**Calibration（概率可靠程度）** 看概率和实际表现是否接近。例如，一批预测都认为某个选项有 80% 的概率成立，实际也应有约 80% 成立。计算时按概率分组，统计这种偏差，偏差越小，得分越高；Choice 使用 hard 难度题，Score 还会考虑整组概率与标准等级的距离。各题型的得分取平均，本次为 **53.9%**。

**Capability（综合表现）** 就是两者的平均值，这次约为 `(39.1% + 53.9%) / 2 = 46.5%`。它是综合表现分，不是答对题目的比例。

想先快速跑通，可以加 `--limit 50` 只取前 50 道。评测结果默认写在 `./outputs`，下面的 Web UI 会直接读到。

## 3. 在 Web UI 里读懂结果

评测完成后启动 Web UI，在 `http://127.0.0.1:9000` 的“评估”页面打开模型报告：

```bash
evalscope service
```

报告页先看整体：Choice 和 Noul 题看准确率，Score 题看平均误差（MAE）。整体没问题，再逐题核对细节。

在“预测浏览”中打开一道题，可以对照模型预测和标准答案，还能查看每个选项的概率。下面就是前面那道中文评论题的真实结果，模型选中了广告推广，界面也标出了对应的标准答案。

![Web UI 中公开评论题的预测、标准答案和四个选项的概率分布](https://sail-moe.oss-cn-hangzhou.aliyuncs.com/yunlin/images/evalscope/doc/decision_evaluation/prediction-probabilities.jpg)

图中的 `spam_or_scam_ad` 对应广告推广，其余三项分别是正常评价、辱骂和闲聊。除了看有没有选对，这组概率也能帮助你观察模型是否在几个选项之间犹豫。需要检查错误时，可以筛出错题，对照题目和概率一起看。

## 4. 响应速度同样重要

自动分类和筛选要反复调用模型，等待时间同样影响体验。压测也不用自己整理数据，还是刚才那套公开题：数据集填 `open_decisions` 就会自动加载题目、逐条发给模型：

```bash
evalscope perf --api systemone --model "$DECISION_MODEL" \
  --url "$DECISION_API_URL" --api-key "$DECISION_API_KEY" \
  --dataset open_decisions \
  --parallel 1 4 --number 24 24 --warmup-num 2 --no-stream
```

这条命令按并发 1 和 4 各发 24 个请求，正式统计前会先热身 2 个。下面的图来自一组中文评论请求的短时压测；你跑上面的命令，就是用公开题复现同一流程。那次测试里，并发从 1 提到 4，吞吐升到三倍多，p95 延迟只略微上涨。左侧是实际报告，右侧是两档并发的对比；p95 表示 95% 请求的延迟水平。

![本次性能报告，以及两档并发的成功吞吐和 p95 延迟](https://sail-moe.oss-cn-hangzhou.aliyuncs.com/yunlin/images/evalscope/doc/decision_evaluation/performance-comparison.png)

读压测报告先看成功率，失败率高的“快”不算数，再看延迟和吞吐。完整记录可以在 Web UI 的“性能”页直接查看，压测结果默认也写在 `./outputs`，不用切路径。接入自己的应用前，记得加大请求量看持续表现；换模型时沿用相同题目和测试方式，比较也更方便。

## 5. 扩展：换成你自己的数据

公开题给的是一个公共参照，合不合你的用途，还得看你自己的数据。数据不用多，把你日常会遇到的典型情况各写几条，就有参考价值。把要判断的内容整理成[自定义数据格式](../advanced_guides/custom_dataset/decision.md)，存到本地，再用 `general_decision` 指过去跑一遍即可：

```bash
evalscope eval \
  --model "$DECISION_MODEL" \
  --eval-type systemone_api \
  --api-url "$DECISION_API_URL" \
  --api-key "$DECISION_API_KEY" \
  --datasets general_decision \
  --dataset-args '{"general_decision":{"local_path":"./decision-data"}}'
```

每条数据一行 JSON，题面、规则和期望答案都在一行里：

```json
{"id": "fold-1", "state": "评论原文", "question": {"type": "noul", "instructions": "这条评论是否该折叠"}, "expected": "no"}
```

跑完照旧打开 Web UI，在“预测浏览”里逐题翻看，其中错题尤其值得停下来细看。开头那个 82% 的误判，就是从自己的数据里翻出来的：这条评论按规则应该保留，模型的预测却与标准答案刚好相反。

> 有人在评论里留了商家电话，我已经举报，这条评论不重复那些联系方式。

![Web UI 中一条中文误判的预测、标准答案和 yes/no 概率](https://sail-moe.oss-cn-hangzhou.aliyuncs.com/yunlin/images/evalscope/doc/decision_evaluation/decision-error-probabilities.jpg)

高概率也可能判断错。折叠门槛调高，通常能减少误伤，也可能漏掉更多推广，具体取舍要放回自己的数据里看。如果错误集中在某一类内容，可以补充这类样本、调整规则，或用 ms-swift [训练自己的决策模型](https://swift.readthedocs.io/en/latest/BestPractices/Decision.html)，再用同一套题检查变化。更多概率指标见[决策模型评测说明](../user_guides/systemone.md)。

## 6. 扩展：在自己机器上部署

不想用托管服务，也可以在自己的机器上跑开源的 [Cloudflare/clef-flash](https://modelscope.cn/models/Cloudflare/clef-flash)（约 9B，基于 Qwen3.5-9B，支持三类题型）。需要一个带 System One 决策头支持的 vLLM，这个支持比较新，所以要装 nightly 版；装好后起服务、换连接参数、跑评测，命令都在下面这一个块里：

```bash
pip install -U modelscope
pip install -U --pre vllm --extra-index-url https://wheels.vllm.ai/nightly
VLLM_USE_MODELSCOPE=true vllm serve Cloudflare/clef-flash \
  --served-model-name clef-flash \
  --max-model-len 16384 --port 8000

# 服务起来后，另开一个终端，换连接参数并评测
export DECISION_MODEL='clef-flash'
export DECISION_API_URL='http://127.0.0.1:8000/v1'
export DECISION_API_KEY='EMPTY'
evalscope eval --model "$DECISION_MODEL" --eval-type systemone_api --api-url "$DECISION_API_URL" --api-key "$DECISION_API_KEY" --datasets open_decisions
```

之后的流程和第 2 到 4 节完全一样：看报告、翻预测、压测，只是模型换成了你自己机器上的这个。

## 总结

本文沿着一条线走完：用 EvalScope 检验决策模型，先用公开题检查答得对不对，再在 Web UI 里逐题看概率靠不靠谱、压测看响应快不快，最后换成自己的数据、自己部署的模型，用同一套流程再验一遍。

合不合适，看三件事：规则写没写清楚、错误能不能接受、调用速度够不够。公开题给的只是参照，真正的答案在你自己的数据里。写下一组你认可的答案，跑一遍评测，看看模型在哪里和你判断不同，这份报告比任何公开榜单都更能回答“这个模型用起来合不合适”。想换题型，[数据集列表](../get_started/supported_dataset/llm.md)里还有更多现成的；要接别的 System One 服务，[决策模型 API 配置](../user_guides/systemone.md)都写好了。挑一条顺手的，开始就行。
