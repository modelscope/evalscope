# CLUEWSC


## 概述

CLUEWSC 是 [FewCLUE](https://github.com/CLUEbenchmark/FewCLUE) 中的中文 Winograd 模式挑战（Winograd Schema Challenge）任务。每道题判断句子中的代词是否指代给定的名词短语，需要依靠常识推理而非表面线索。

## 任务描述

- **任务类型**：二分类指代消解，以单项选择形式评测（中文）
- **输入**：一个中文句子、其中的代词和一个候选指代对象
- **输出**：A（是，指代该对象）或 B（否，不指代）
- **领域**：中文常识推理

## 主要特点

- 使用 FewCLUE 官方划分：`test_public`（976 条，标签均衡）、`dev_few_all`（159 条）和 `train_few_all`（160 条）
- 问题措辞与 OpenCompass `FewCLUE_cluewsc_gen` 配置一致，分数可对比

## 评估说明

- 默认在 `test_public` 上进行 **0-shot** 评估；设置 `eval_split='dev_few_all'` 可评估开发集。
- 设置 `few_shot_num` 可从 `train_few_all` 中抽取示例。
- 答案应遵循格式："答案：[LETTER]"


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `cluewsc` |
| **数据集ID** | [evalscope/cluewsc](https://modelscope.cn/datasets/evalscope/cluewsc/summary) |
| **论文** | [论文](https://arxiv.org/abs/2107.07498) |
| **标签** | `Chinese`, `MCQ`, `Reasoning` |
| **指标** | `accuracy` |
| **默认样本数（Shots）** | 0-shot |
| **评估分割** | `test_public` |
| **训练分割** | `train_few_all` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 976 |
| 提示词长度（平均） | 182.99 字符 |
| 提示词长度（最小/最大） | 135 / 271 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "e440caf5",
      "content": "回答下面的单项选择题，请选出其中的正确答案。你的回答的全部内容应该是这样的格式：\"答案：[LETTER]\"（不带引号），其中 [LETTER] 是 A,B 中的一个。\n\n问题：不过，在面子上，毛豆还下不来，一半是因为他确实很生气；另一半也是因为，他毛豆怎么能与他们做一路人。所以，他必须生气。有几次大王问他累不累，要不要喝水，后面的人立即送上矿泉水瓶子，他不理睬。\n此处，“他”是否指代“毛豆”？\n选项：\nA) 是\nB) 否\n"
    }
  ],
  "choices": [
    "是",
    "否"
  ],
  "target": "A",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "idx": 1180
  }
}
```

## 提示模板

**提示模板：**
```text
回答下面的单项选择题，请选出其中的正确答案。你的回答的全部内容应该是这样的格式："答案：[LETTER]"（不带引号），其中 [LETTER] 是 {letters} 中的一个。

问题：{question}
选项：
{choices}

```

<details>
<summary>少样本模板</summary>

```text
以下是一些示例问题：

{fewshot}

回答下面的单项选择题，请选出其中的正确答案。你的回答的全部内容应该是这样的格式："答案：[LETTER]"（不带引号），其中 [LETTER] 是 {letters} 中的一个。

问题：{question}
选项：
{choices}

```

</details>

## 使用方法

### 使用命令行（CLI）

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets cluewsc \
    --limit 10  # 正式评估时请删除此行
```

### 使用 Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['cluewsc'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
