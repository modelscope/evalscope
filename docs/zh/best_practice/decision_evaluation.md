# 使用 EvalScope 评测决策模型

本文带你完成四件事：**选 benchmark → 部署模型 → 查看效果 → 测试性能**。先用少量样本跑通，再进行完整评测。所有命令均在 EvalScope 源码根目录执行，模型和数据从 ModelScope 下载。

## 1. 先确认自己的模型属于哪一类

- **Chat 模型**：通过 `/v1/chat/completions` 生成答案，例如输出 `ANSWER: A`。选择 `eval_type='openai_api'`。
- **决策模型**：直接返回选项概率、判断概率或评分分布。根据服务接口选择 `autotrust_api` 或 `systemone_api`。

决策模型有三类题型：Choice 是“从几个选项中选一个”；Noul 是“这件事成立的概率是多少”；Score 是“在有顺序的几个等级中评分”。例如，Score 的六档可以是 0–5 分，模型返回每档概率，EvalScope 用这些概率计算平均分。

先按下表选择 benchmark。普通 chat 接口不能直接替代决策接口。

| Benchmark | 用来测什么 | 普通 chat | System One 决策接口 | 本文的 AutoTrust JEV 示例 |
|---|---|---|---|---|
| `arc` | 科学选择题，本文选 ARC-Challenge | 支持 | 支持 | 支持 |
| `banking77` | 银行业务意图分类，每题保留 77 类 | 支持 | 支持 | 支持 |
| `boolq` | 根据短文回答是非题 | 支持 Choice | 支持 Choice / Noul | 支持 Choice / Noul |
| `reward_bench` | 从两个回答中选出更好的回答 | 支持 | 支持 | 支持 |
| `stsb` | 两句话有多相似，0–5 分 | 不支持本文适配器 | 支持 | 支持六档 Score |
| `jevbench` | 231 道公开决策题，包含三类题型 | 不支持 | 需支持题目原有等级 | 不能完整运行：包含四档、五档 Score |
| `open_decisions` | 2,352 道三类决策题 | 不支持 | 需支持题目原有等级 | 不能完整运行：包含三至六档 Score |
| `general_decision` | 你自己的决策题 | 不支持 | 按题型和等级支持 | Score 必须是六档 |

“支持普通 chat”表示可以测答案正确率。只有返回完整概率的决策接口，才能进一步计算本文的概率指标。不要让 chat 模型随口给一个“置信度”，再把它当作模型的选项概率。

这里的 [JevBench](https://modelscope.cn/datasets/evalscope/jevbench) 是 [Benchmark Heaven 的公开子集](https://github.com/fstandhartinger/jevbench)，与其他同名 benchmark 不同；它不提供完整榜单的 Composite 成绩。[OpenDecisions](https://modelscope.cn/datasets/evalscope/OpenDecisions) 使用发布包中的 `train` 划分做评测，不会训练模型。

## 2. 用 vLLM 部署模型

下面以 Linux / NVIDIA GPU 为例，先创建独立环境并安装 EvalScope。每个新终端都先进入同一仓库，执行 `source .venv/bin/activate`：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[perf]'
pip install -U uv modelscope fastapi uvicorn httpx
```

下面两条路径任选其一。已有可用模型服务的用户可以直接跳到第 3 节。

### 路径 A：普通 chat 模型

以一个小模型为例；你可以换成自己的 ModelScope 模型 ID，或者直接使用已下载的本地模型目录：

```bash
uv pip install -U vllm --torch-backend auto
modelscope download --model Qwen/Qwen2.5-0.5B-Instruct --local_dir ./models/my-chat-model
vllm serve ./models/my-chat-model \
  --served-model-name my-chat-model --port 8000 --max-model-len 8192
```

保留这个终端运行，在另一个终端检查服务：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"my-chat-model","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```

能收到回答，就可以使用第 3 节的 chat 配置。后续的 `model` 必须与这里的 `--served-model-name` 一致。

### 路径 B：AutoTrust 决策模型

以下使用 `autotrust/JEV-27B`。它需要比上面的小模型更多显存；如果单卡放不下，可按 vLLM 的要求使用多卡。该模型使用较新的架构，这里按 [vLLM 官方方法](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)安装开发版；下载脚本会准备完整模型并校验文件：

```bash
uv pip install -U vllm --torch-backend auto \
  --extra-index-url https://wheels.vllm.ai/nightly
python examples/decision/download_models.py \
  --model autotrust/JEV-27B --output ./models/JEV-27B
vllm serve ./models/JEV-27B \
  --served-model-name autotrust/JEV-27B \
  --enable-lora --max-lora-rank 32 \
  --lora-modules jev-decision=./models/JEV-27B/adapter_vllm \
  --max-model-len 16384 --max-logprobs 256 \
  --logprobs-mode processed_logprobs --port 18132
```

然后在另一个终端启动决策接口：

```bash
python examples/decision/serve_autotrust.py \
  --model-path ./models/JEV-27B \
  --upstream http://127.0.0.1:18132/v1 --port 18133
```

这里有两个端口：**18132 运行 vLLM，18133 是给 EvalScope 使用的决策接口**。模型的决策能力需要完整 adapter/head；仅启动底座 chat 服务不够。请使用支持该模型架构和输出 head LoRA 的 vLLM 版本，模型的[官方部署说明](https://modelscope.cn/models/autotrust/JEV-27B)列出了要求。

检查一次决策请求：

```bash
curl http://127.0.0.1:18133/v1/decide \
  -H 'Content-Type: application/json' \
  -d '{"kind":"choice","state":"The sky is blue.","question":"What color is the sky?","options":["Blue","Red"],"thinking":"off"}'
```

响应应包含 `options` 和一一对应的 `probabilities`，例如 `[0.97, 0.03]`。实际数字以服务返回为准。

如果使用自己的决策模型，需要让它提供相应的 `/v1/decide` 或 `/v1/systemone` 接口。上面的示例服务针对 AutoTrust 发布格式；其他模型不能只改模型名字就套用。标准 vLLM 的聊天接口不会自动变成这两种决策接口。

## 3. 先跑少量样本，再扩大评测

将下面适合自己的配置保存为 `evaluate.py`，执行 `python evaluate.py`。

### 普通 chat：测四个现有 benchmark

```python
from evalscope import TaskConfig, run_task

run_task(TaskConfig(
    model='my-chat-model',
    eval_type='openai_api',
    api_url='http://127.0.0.1:8000/v1',
    api_key='EMPTY',
    datasets=['arc', 'banking77', 'boolq', 'reward_bench'],
    dataset_args={'arc': {'subset_list': ['ARC-Challenge']}},
    generation_config={'temperature': 0, 'max_tokens': 256},
    limit=5,
))
```

这里的 BoolQ 使用默认 Choice。不要在 chat 配置中打开 Noul，也不要加入表中标为不支持 chat 的决策 benchmark。

### AutoTrust：同时测 Choice、Noul、Score

```python
from evalscope import TaskConfig, run_task

run_task(TaskConfig(
    model='autotrust/JEV-27B',
    eval_type='autotrust_api',
    api_url='http://127.0.0.1:18133/v1',
    datasets=['arc', 'banking77', 'boolq', 'reward_bench', 'stsb'],
    dataset_args={
        'arc': {'subset_list': ['ARC-Challenge']},
        'boolq': {'extra_params': {'decision_kind': 'noul'}},
    },
    eval_batch_size=4,
    generation_config={'timeout': 300, 'retries': 1},
    limit=5,
))
```

这个配置用 ARC / BANKING77 / RewardBench 测 Choice，用 BoolQ 测 Noul，用 STS-B 测六档 Score。决策接口不使用 `temperature`、`max_tokens` 或流式输出；`retries=1` 表示每题只请求一次。

### System One：测完整 JevBench / OpenDecisions

已经有支持原始 Score 等级的 System One 服务时，使用：

```python
from evalscope import TaskConfig, run_task

run_task(TaskConfig(
    model='your-decision-model',
    eval_type='systemone_api',
    api_url='http://127.0.0.1:9000/v1',
    datasets=['jevbench', 'open_decisions'],
    eval_batch_size=4,
    generation_config={'timeout': 300, 'retries': 1},
    limit=5,
))
```

将模型名和地址替换为实际服务；该地址必须能处理 `/v1/systemone` 请求。不能把 AutoTrust 的六档 Score 改写成四档、五档来凑完整评测。

**跑通后删除 `limit=5`，即可做完整评测。** `limit` 按每个子集生效，因此有多个子集的 benchmark 可能不止请求 5 题。数据会自动从 ModelScope 加载，标准答案不会发给模型。

## 4. 怎么看结果

终端会显示报告目录，通常位于 `outputs/<时间>/reports/`；打开其中的 `report.html` 看汇总，`predictions/` 和 `reviews/` 中可以追查具体样本。

先看**请求是否全部成功、有效评分数量是否完整**，再看分数。只有少数题成功的高分，不能代表整套 benchmark。

| 指标 | 大白话解释 | 哪个方向更好 |
|---|---|---|
| Accuracy | 有多少题选对了 | 越高越好 |
| macro-F1 | 每个类别都同样重要，避免只看常见类别 | 越高越好 |
| Precision / Recall / F1 | Noul 中，肯定回答有多准 / 找出了多少正例 / 两者的平衡 | 越高越好 |
| NLL / Brier | 给正确答案的概率是否合理；自信地答错会被惩罚 | 越低越好 |
| ECE | 说自己有八成把握的答案，实际是否大约八成正确 | 越低越好；也要同时看准确率 |
| AUROC / AUPRC | Noul 能否把“成立”和“不成立”的题分开 | 越高越好；不同数据集间慎比 |
| MAE | 预测分数平均离标准分数多远 | 越低越好 |
| Spearman / Pearson | 预测分数与标准分数的排序 / 变化趋势是否一致 | 越接近 1 越好 |
| nRPS | Score 的概率是否集中在正确等级附近 | 越低越好 |
| AURC | 优先回答把握大的题时，错误风险能降到多低 | 越低越好 |

Score 的 MAE 看期望分数；离散等级的 Accuracy 看概率最高的一档，两者含义不同。例如，STS-B 使用 0–5 分，MAE 为 0.6 表示平均相差约 0.6 分。它保留连续标准分数，不强行取整来计算分类准确率或 nRPS。有些指标需要足够多的不同样本才能计算；报告缺项时先看原因，不要把缺项当作零分。

JevBench 的主指标是公开子集准确率，并另报 family-macro；OpenDecisions 的主指标是 Capability，同时报告 Intelligence（任务能力）和 Calibration（概率校准）。它们使用各自的官方口径，与普通 Accuracy 不完全相同。JSON 中的比率 `0.82` 对应 `82%`；MAE 仍按分数单位读。只跑 `limit` 样本得到的是试跑成绩，不能当作完整榜单成绩。RewardBench 这里使用 v1 的成对选择和样本加权汇总，也不是官方分类别加权榜单成绩。

## 5. 用自己的数据

创建目录 `decision-data`，将下面内容保存为 `decision-data/default_test.jsonl`，**每行一道题**：

```json
{"id":"c1","state":"The customer cannot find their card.","question":{"type":"choice","instructions":"What should support help with?","criteria":{"find":"Find the card","close":"Close the account"}},"expected":"find"}
{"id":"n1","state":"The sky is blue.","question":{"type":"noul","instructions":"Is the sky blue?"},"expected":"yes"}
{"id":"s1","state":"Excellent service.","question":{"type":"score","instructions":"Rate satisfaction.","criteria":["Very poor","Poor","Fair","Good","Very good","Excellent"]},"expected":"5"}
```

`state` 是给模型看的材料，`question` 是问题，`expected` 是只用于评分的标准答案。Choice 的标准答案必须是选项 ID；Noul 是 `yes` / `no`；Score 是从 0 开始的等级编号。

在已经跑通的决策配置中替换：

```python
datasets=['general_decision'],
dataset_args={'general_decision': {'local_path': './decision-data'}},
```

其余模型和 API 参数保持原来的配置。使用 AutoTrust 示例时，Score 保留六档；其他服务按其实际支持的等级填写。

## 6. 怎么测性能

先停止同一个服务上的效果评测，避免两种流量互相影响。下面接着使用 AutoTrust 决策服务；普通 chat 模型的压测见[性能评测快速开始](../user_guides/stress_test/quick_start.md)。

先导出三组固定请求。Choice 使用完整 77 类的 BANKING77，Noul 使用 BoolQ，Score 使用 STS-B：

```bash
python examples/decision/prepare_perf_requests.py \
  --api autotrust --model autotrust/JEV-27B --output ./perf-input/autotrust
```

再分别测三种题型：

```bash
for kind in choice noul score; do
  evalscope perf --api autotrust --model autotrust/JEV-27B \
    --url http://127.0.0.1:18133/v1 --dataset line_by_line \
    --dataset-path "./perf-input/autotrust/$kind.jsonl" \
    --parallel 1 4 8 16 --number 1000 1000 1000 1000 \
    --warmup-num 20 --no-stream --read-timeout 300 \
    --outputs-dir "./perf-results/$kind"
done
```

这表示每种题型分别用 1、4、8、16 并发测试，每档预热 20 次，再正式测 1,000 次。第一次试跑可以缩减为 `--parallel 1 --number 20 --warmup-num 2`，确认成功后再跑完整配置。

如果测的是 System One 服务，导出和压测时都改为 `--api systemone`，并替换模型名、服务地址和输入目录。导出的文件对应具体协议，不能只切换压测命令中的 API 名。

在输出目录打开 `perf_report.html`，按这个顺序读：

1. **成功率、失败和超时**：先确认服务正确处理了请求；HTTP 200 但答案格式非法也算失败。
2. **p50 / p90 / p95 / p99 延迟**：分别表示约 50% / 90% / 95% / 99% 的请求能在这个时间内完成，越低越快。
3. **成功 req/s**：每秒完成多少次决策，越高越好。本文一请求一题，因此也等于 decisions/s。
4. **并发增加后的变化**：吞吐提高多少，延迟是否明显变长；如果吞吐几乎不变而延迟变长，往往是在排队。

决策接口不逐字输出，因此不看 TTFT、TPOT、ITL 等生成速度指标。请求延迟包含网络和排队；比较两次结果时，应使用同一组输入、相同模型配置和相同测量位置。

## 常见问题

| 问题 | 处理方法 |
|---|---|
| `404 Not Found` | 核对 API 类型和端口；普通 vLLM chat 端口没有原生 `/decide` / `/systemone` |
| vLLM 提示不支持模型或 `lm_head` LoRA | 使用模型官方要求的兼容 vLLM 版本，保留完整 adapter/head |
| 提示 Score 必须是六档 | AutoTrust 示例的限制；换用支持原始等级的服务，不要改 benchmark 的等级 |
| Chat 配置运行新决策 benchmark 时报错 | 按第 1 节的兼容表更换 benchmark 或 API |
| 只想先确认能跑 | 评测加 `limit=5`，压测先用 20 次请求 |
| 想复核个别答案 | 查看对应的 predictions 和 reviews，先确认题目、原选项、答案和评分 |
