# 决策模型

使用 `general_decision`，可以评测自己业务中的选择、是非判断和分级评分。数据保存在本地 JSONL 文件中，无需上传数据集。

这种格式需要 `systemone_api` 或 `autotrust_api` 决策服务。模型部署和连接参数见[决策模型评测最佳实践](../../best_practice/decision_evaluation.md)；普通聊天模型的问答和选择题见[大模型数据格式](llm.md)。

## 数据准备

在运行评测的目录下创建以下文件：

```text
decision-data/
└── default_test.jsonl
```

将下面三个样本保存到 `decision-data/default_test.jsonl`，使用 UTF-8 编码，每行一个完整 JSON 对象。它们分别对应 Choice、Noul 和四档 Score：

```json
{"id":"route","state":"用户要查询今天的天气，当前还没有联网查询结果。","question":{"type":"choice","instructions":"下一步使用哪类能力？","criteria":{"search":"查询外部信息","calculate":"进行数值计算","answer":"根据已有信息回答"}},"expected":"search"}
{"id":"ready","state":"用户要查询今天的天气，当前还没有联网查询结果。","question":{"type":"noul","instructions":"当前是否已经取得今天的天气信息？"},"expected":"no"}
{"id":"complete","state":"任务要求列出价格、库存和购买链接。当前答案只给出了价格。","question":{"type":"score","instructions":"按已经覆盖的要求数量评价完成度。","criteria":["未覆盖任何要求","覆盖一项","覆盖两项","三项都覆盖"]},"expected":"1"}
```

这些样本用于说明格式。正式比较模型时，需要使用独立、能代表真实业务的数据，并提供可靠的标准答案。

### 必需字段

| 字段 | 类型 | 含义 |
|---|---|---|
| `state` | JSON 值，通常为字符串或对象 | 作出判断时，模型能够看到的材料 |
| `question` | 对象 | 一道题；每条记录暂不支持多道题 |
| `question.type` | 字符串 | `choice`、`noul` 或 `score` |
| `question.instructions` | 非空字符串 | 问题以及判断标准 |
| `question.criteria` | 对象或数组 | Choice、Score 必填；Noul 可省略，具体见下表 |
| `expected` | 标准答案 | 只用于评分，不同题型的含义见下表 |

`state` 和 `question` 会发送给模型服务。不要把标准答案、正确答案的解释或教师模型预测放进这两个字段。

### 三类题型分别怎么填

| 题型 | `criteria` | `expected` | 怎么评 |
|---|---|---|---|
| Choice | 按所需顺序填写“选项 ID → 描述”的对象 | 选项 ID，如 `"search"`，不是选项描述 | 比较最高概率选项与标准答案；概率并列时按选项 ID 确定结果 |
| Noul | 通常省略；也可用对象描述 `true`、`false` 中的一个或两个条件 | `"yes"` / `"no"`；也接受布尔值及 `"true"` / `"false"` | 比较 `P(yes)` 与分类阈值，默认阈值为 0.5 |
| Score | 从低到高排列的等级描述数组 | 从 0 开始的等级编号，如 `"1"` 或整数 `1` | 最大概率档用于等级准确率；完整分布还用于计算概率加权分数与评分误差 |

上例的四个等级依次编号为 `0`、`1`、`2`、`3`，`expected="1"` 表示“覆盖一项”，不是从 1 开始编号。对 normalized RPS 来说，错到很远的等级，比错到相邻等级带来的误差更大。

EvalScope 接受 2–255 个 Choice 选项、2–10 个 Score 等级，模型服务可能有更小的上限。AutoTrust 原生 `/decide` **只支持六档 Score**，不要给四档任务凑等级或改变原意。

### 可选字段

| 字段 | 默认值 | 用途 |
|---|---|---|
| `id` | 无 | 在样本元数据中保留源 ID；建议使用唯一字符串，方便追查 |
| `continuous` | `false` | 仅限 Score：将 `expected` 作为 `0` 至“等级数量减 1”之间的连续分数，见下文 |
| `fixed_classes` | `false` | 固定分类任务额外计算 macro-F1；所有样本的选项 ID 须保持同一套类别含义 |
| `family` | 空 | 对离散标准答案额外计算 `family_macro`：先算每个家族的准确率，再对家族等权平均 |
| `group`、`tier`、`language` | 无／空 | 保留来源信息；不会自动生成子集或新增指标 |

### 文件命名与子集

`local_path` 指向目录，而非 JSONL 文件。默认子集为 `default`、评测划分为 `test`，所以文件名是 `default_test.jsonl`。如果改用 `routing_test.jsonl` 和 `quality_test.jsonl`，在该 benchmark 的数据参数中设置 `subset_list=['routing', 'quality']`；每个文件都可以包含三类题型。

### 连续标准分数（可选）

如果标准答案本身是连续评分，使用 `continuous=true`。例如将下面的样本单独保存为 `decision-data/continuous_test.jsonl`：

```json
{"id":"similarity","state":{"sentence1":"A girl is styling her hair.","sentence2":"A girl is brushing her hair."},"question":{"type":"score","instructions":"Rate the semantic similarity of the two sentences.","criteria":["Unrelated","Some shared meaning","Mostly similar","Equivalent"]},"expected":1.5,"continuous":true}
```

## 配置任务

启动本地 System One 服务后，将以下代码保存为 `evaluate_custom.py`，执行 `python evaluate_custom.py`。使用已有服务时，设置 `DECISION_MODEL`、`DECISION_API_URL`、`DECISION_API_KEY` 即可，数据文件不用改变。

```python
import os

from evalscope import TaskConfig, run_task

run_task(TaskConfig(
    model=os.getenv('DECISION_MODEL', 'clef-flash'),
    eval_type='systemone_api',
    api_url=os.getenv('DECISION_API_URL', 'http://127.0.0.1:8000/v1'),
    api_key=os.getenv('DECISION_API_KEY', 'EMPTY'),
    datasets=['general_decision'],
    dataset_args={
        'general_decision': {
            'local_path': './decision-data',
        },
    },
    generation_config={'timeout': 300, 'retries': 1},
    eval_batch_size=1,
))
```

这里不需要聊天生成参数，也不需要 LLM Judge，EvalScope 直接根据决策结果和标准答案评分。

### 连续评分的配置

只评测这个四档连续 Score 子集时，将脚本中的 `dataset_args` 替换为：

```python
dataset_args={
    'general_decision': {
        'local_path': './decision-data',
        'subset_list': ['continuous'],
        'metric_list': ['mae'],
        'primary_metric': 'mae',
    },
},
```

此时主指标为 MAE。连续标准答案不会被取整计算 Accuracy、NLL、Brier、ECE 或 RPS；满足样本条件时，仍可计算 MAE、相关系数和风险—覆盖率。如果业务分数使用其他量程，先将标准答案转换到声明的从 0 开始的等级尺度。

## 评测指标与结果

默认主指标为**准确率**，汇总有离散标准答案的样本。其他指标按题型分开，Score 还会按等级数量分开，不把不同题型和尺度的误差混在一起平均。

| 题型 | 指标 | 解读 |
|---|---|---|
| Choice | Accuracy；NLL、多分类 Brier、ECE；固定分类任务还有 macro-F1 | 准确率、F1 越高越好；概率误差越低越好 |
| Noul | Accuracy、Precision、Recall、F1、AUROC、AUPRC；NLL、二分类 Brier、ECE | 分类与排序指标越高越好；概率误差越低越好 |
| 离散 Score | Accuracy、MAE、normalized RPS；NLL、多分类 Brier、ECE；Spearman、Pearson | MAE 的单位是等级；误差越低越好，相关系数越接近 1 越好 |
| 所有题型 | AURC 与保存的风险—覆盖率数据 | AURC 越低越好，同时看清置信度来源和风险定义 |

上例只有三个样本，AUROC 可能无法计算，Score 的相关系数也无法计算。报告会保留原因，缺项不等于零分。比较分数前，先检查执行覆盖情况和有效样本数量。

可通过 [EvalScope Web UI](../../best_practice/decision_evaluation.md#2-报告出来了先看哪几项)查看，也可检查输出目录的 `predictions/`、`reviews/` 和 `reports/`。概率面板可切换原始概率和评分用概率；如果评分时对舍入后的概率进行了归一化，review 中会记录这一处理。

更多配置见[决策模型 API](../../user_guides/systemone.md)和[评测参数](../../get_started/parameters.md)。
