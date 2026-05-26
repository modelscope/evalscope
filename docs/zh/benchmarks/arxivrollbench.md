# ArxivRollBench

## 概述

ArxivRollBench 是一个基于近期 arXiv 论文构建的滚动基准测试。它通过三种任务形式（排序、完形填空和下一段预测）评估大语言模型是否能够对最新的科学文本进行推理。

## 任务描述

- **任务类型**：多项选择科学文本推理
- **输入**：近期 arXiv 文本片段，附带四个选项
- **输出**：单个正确答案字母（A、B、C 或 D）
- **领域**：计算机科学、定量金融、数学、物理学、统计学、定量生物学、经济学以及电气工程/系统科学
- **版本**：2024b、2025a 和 2026a 的滚动快照

## 主要特点

- 时间感知的基准快照可减少因数据污染导致的性能高估
- 覆盖多个 arXiv 领域及不同科学写作风格
- 在 SCP 框架下包含排序（sequencing）、完形填空（cloze）和预测（prediction）三种格式
- 紧凑型 `-50` 子集适用于成本可控的 API 评估
- 完整子集可通过 `arxivrollbench_full` 获取

## 评估说明

- 默认配置使用 **0-shot** 评估
- 默认的 `arxivrollbench` 基准使用紧凑型 `-50` 数据集
- 如需完整公开子集，请使用 `arxivrollbench_full`
- 每个子集均从 `liangzid` 命名空间下的公开 ModelScope 镜像加载
- 答案统一归一化为 A-D，并以准确率（accuracy）进行评估

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `arxivrollbench` |
| **数据集ID** | [liangzid/arxivrollbench](https://modelscope.cn/datasets/liangzid/arxivrollbench/summary) |
| **论文** | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/41098) |
| **标签** | `Knowledge`, `MCQ`, `Reasoning` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估分割** | `train` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 3,254 |
| 提示词长度（平均） | 1514.19 字符 |
| 提示词长度（最小/最大） | 307 / 14112 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示平均长度 | 提示最小长度 | 提示最大长度 |
|--------|---------|-------------|------------|------------|
| `2024b_cs_s` | 42 | 949.6 | 590 | 1805 |
| `2024b_cs_c` | 31 | 307 | 307 | 307 |
| `2024b_cs_p` | 50 | 2617.32 | 922 | 7512 |
| `2024b_q_fin_s` | 49 | 1042.31 | 586 | 2329 |
| `2024b_q_fin_c` | 44 | 307 | 307 | 307 |
| `2024b_q_fin_p` | 50 | 3430.52 | 872 | 9106 |
| `2024b_math_s` | 34 | 829.85 | 593 | 2115 |
| `2024b_math_c` | 15 | 307 | 307 | 307 |
| `2024b_math_p` | 51 | 1957.24 | 869 | 6260 |
| `2024b_physics_s` | 45 | 957.11 | 576 | 4402 |
| `2024b_physics_c` | 28 | 307 | 307 | 307 |
| `2024b_physics_p` | 51 | 2948.1 | 885 | 13643 |
| `2024b_stat_s` | 45 | 936.4 | 582 | 1678 |
| `2024b_stat_c` | 33 | 307 | 307 | 307 |
| `2024b_stat_p` | 50 | 2946.44 | 861 | 7026 |
| `2024b_q_bio_s` | 43 | 975 | 583 | 2555 |
| `2024b_q_bio_c` | 34 | 307 | 307 | 307 |
| `2024b_q_bio_p` | 49 | 3354.53 | 883 | 8867 |
| `2024b_econ_s` | 48 | 1021.58 | 586 | 2070 |
| `2024b_econ_c` | 43 | 307 | 307 | 307 |
| `2024b_econ_p` | 50 | 3257.76 | 846 | 8967 |
| `2024b_eess_s` | 48 | 1034.56 | 574 | 2922 |
| `2024b_eess_c` | 42 | 307 | 307 | 307 |
| `2024b_eess_p` | 51 | 2612.69 | 882 | 8609 |
| `2025a_cs_s` | 50 | 921.2 | 592 | 1632 |
| `2025a_cs_c` | 44 | 307 | 307 | 307 |
| `2025a_cs_p` | 51 | 2895.02 | 942 | 6540 |
| `2025a_q_fin_s` | 50 | 931.08 | 589 | 2202 |
| `2025a_q_fin_c` | 43 | 307 | 307 | 307 |
| `2025a_q_fin_p` | 51 | 2837.86 | 793 | 7577 |
| `2025a_math_s` | 42 | 852.52 | 580 | 1595 |
| `2025a_math_c` | 28 | 307 | 307 | 307 |
| `2025a_math_p` | 51 | 2449.49 | 889 | 6893 |
| `2025a_physics_s` | 44 | 939.32 | 587 | 1874 |
| `2025a_physics_c` | 34 | 307 | 307 | 307 |
| `2025a_physics_p` | 49 | 3568.29 | 1001 | 9325 |
| `2025a_stat_s` | 48 | 932.81 | 600 | 2063 |
| `2025a_stat_c` | 42 | 307 | 307 | 307 |
| `2025a_stat_p` | 50 | 3115.36 | 822 | 7349 |
| `2025a_q_bio_s` | 49 | 1074.12 | 591 | 1810 |
| `2025a_q_bio_c` | 49 | 307 | 307 | 307 |
| `2025a_q_bio_p` | 50 | 3639.26 | 1038 | 8890 |
| `2025a_econ_s` | 48 | 982.19 | 591 | 2322 |
| `2025a_econ_c` | 45 | 307 | 307 | 307 |
| `2025a_econ_p` | 51 | 2860.9 | 884 | 6494 |
| `2025a_eess_s` | 46 | 1017.35 | 588 | 1807 |
| `2025a_eess_c` | 42 | 307 | 307 | 307 |
| `2025a_eess_p` | 50 | 3541.1 | 943 | 14112 |
| `2026a_cs_s` | 51 | 944.12 | 584 | 1795 |
| `2026a_cs_c` | 38 | 307 | 307 | 307 |
| `2026a_cs_p` | 51 | 2629.06 | 919 | 5234 |
| `2026a_q_fin_s` | 48 | 1025.44 | 608 | 2320 |
| `2026a_q_fin_c` | 45 | 307 | 307 | 307 |
| `2026a_q_fin_p` | 51 | 3094.78 | 872 | 6644 |
| `2026a_math_s` | 44 | 844.05 | 575 | 1381 |
| `2026a_math_c` | 30 | 307 | 307 | 307 |
| `2026a_math_p` | 51 | 2160.27 | 860 | 12385 |
| `2026a_physics_s` | 47 | 1082.04 | 599 | 2522 |
| `2026a_physics_c` | 41 | 307 | 307 | 307 |
| `2026a_physics_p` | 50 | 3420.58 | 894 | 8788 |
| `2026a_stat_s` | 49 | 1013.47 | 575 | 2482 |
| `2026a_stat_c` | 46 | 307 | 307 | 307 |
| `2026a_stat_p` | 51 | 2564.47 | 955 | 6387 |
| `2026a_q_bio_s` | 47 | 1019.7 | 584 | 1707 |
| `2026a_q_bio_c` | 40 | 307 | 307 | 307 |
| `2026a_q_bio_p` | 48 | 3030.71 | 954 | 6468 |
| `2026a_econ_s` | 48 | 989.67 | 580 | 2320 |
| `2026a_econ_c` | 47 | 307 | 307 | 307 |
| `2026a_econ_p` | 51 | 2920.76 | 885 | 7061 |
| `2026a_eess_s` | 51 | 988.14 | 579 | 2231 |
| `2026a_eess_c` | 45 | 307 | 307 | 307 |
| `2026a_eess_p` | 51 | 2812.61 | 922 | 5589 |

## 样例示例

**子集**: `2024b_cs_s`

```json
{
  "input": [
    {
      "id": "7b220fd6",
      "content": "Answer the following ArxivRollBench multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D.\n\nSelect the option that correctly compl ... [TRUNCATED 381 chars] ... m a diagonal matrix into the identity, allows us to write the input matrix as a product of transvections. **C**: Note that row and column operations are effected by left- and right multiplications by transvections\n\nA) BAC\nB) ABC\nC) ACB\nD) BCA"
    }
  ],
  "choices": [
    "BAC",
    "ABC",
    "ACB",
    "BCA"
  ],
  "target": "C",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "original_label": "Selection 3",
    "task_type": "s/c"
  }
}
```

## 提示模板

**提示模板：**
```text
Answer the following ArxivRollBench multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}.

{question}

{choices}
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets arxivrollbench \
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
    datasets=['arxivrollbench'],
    dataset_args={
        'arxivrollbench': {
            # subset_list: ['2024b_cs_s', '2024b_cs_c', '2024b_cs_p']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```