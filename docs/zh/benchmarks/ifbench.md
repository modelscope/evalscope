# IFBench

## 概述

IFBench 是一个用于评估 AI 模型在遵循新颖、具有挑战性且多样化的可验证指令方面可靠性的基准测试，特别强调**领域外泛化能力**。该基准由 AllenAI 开发，旨在解决现有基准中存在的过拟合和数据污染问题。

## 任务描述

- **任务类型**：指令遵循评估（Instruction Following Evaluation）
- **输入**：包含可验证约束的提示（prompts）
- **输出**：必须满足特定约束的响应
- **重点**：精确的指令遵循能力

## 核心特性

- 包含 58 个人工精心整理的可验证约束
- 约束类别涵盖：计数、格式、词汇使用等
- 重点关注领域外泛化能力
- 支持对约束满足情况进行程序化验证
- 有效缓解数据污染问题

## 评估说明

- 默认配置采用 **0-shot** 评估方式
- 评估指标包括：`prompt_level_strict`、`inst_level_strict`、`prompt_level_loose`、`inst_level_loose`
- 需要安装 `emoji`、`syllapy` 和 `spacy` 等依赖包
- 同时评估严格和宽松条件下的约束满足情况

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `ifbench` |
| **数据集ID** | [allenai/IFBench_test](https://modelscope.cn/datasets/allenai/IFBench_test/summary) |
| **论文** | N/A |
| **标签** | `InstructionFollowing` |
| **指标** | `prompt_level_strict`, `inst_level_strict`, `prompt_level_loose`, `inst_level_loose` |
| **默认示例数量** | 0-shot |
| **评估划分** | `train` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 300 |
| 提示词长度（平均） | 343.41 字符 |
| 提示词长度（最小/最大） | 50 / 904 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "9e0a5835",
      "content": "What should the world's smartest man, surrounded by corruption, greed, inequity, madness, inequality, an establishment who preached conspiracy theories and wild speculations over truth and an equally evil resistance funded by the mega rich, a ... [TRUNCATED] ... ad here. Include keyword kaleidoscope once in your response, keyword nebula twice in your response, keyword whisper three times in your response, keyword labyrinth five times in your response, and keyword paradox seven times in your response."
    }
  ],
  "target": "",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "key": "0",
    "prompt": "What should the world's smartest man, surrounded by corruption, greed, inequity, madness, inequality, an establishment who preached conspiracy theories and wild speculations over truth and an equally evil resistance funded by the mega rich, a ... [TRUNCATED] ... ad here. Include keyword kaleidoscope once in your response, keyword nebula twice in your response, keyword whisper three times in your response, keyword labyrinth five times in your response, and keyword paradox seven times in your response.",
    "instruction_id_list": [
      "count:keywords_multiple"
    ],
    "kwargs": [
      {
        "N": null,
        "capital_frequency": null,
        "capital_relation": null,
        "end_phrase": null,
        "first_word": null,
        "forbidden_words": null,
        "frequency": null,
        "keyword": null,
        "keyword1": "kaleidoscope",
        "keyword2": "nebula",
        "keyword3": "whisper",
        "keyword4": "labyrinth",
        "keyword5": "paradox",
        "keywords": null,
        "language": null,
        "let_frequency": null,
        "let_relation": null,
        "letter": null,
        "m": null,
        "max_words": null,
        "min_words": null,
        "n": null,
        "n_end": null,
        "n_start": null,
        "nth_paragraph": null,
        "num_bullets": null,
        "num_highlights": null,
        "num_paragraphs": null,
        "num_placeholders": null,
        "num_sections": null,
        "num_sentences": null,
        "num_words": null,
        "options": null,
        "percentage": null,
        "postscript_marker": null,
        "prompt_to_repeat": null,
        "reference_text": null,
        "relation": null,
        "section_spliter": null,
        "sep": null,
        "small_n": null,
        "word": null
      }
    ]
  }
}
```

*注：部分内容因展示需要已被截断。*

## 提示模板

*未定义提示模板。*

## 使用方法

### 通过命令行（CLI）

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets ifbench \
    --limit 10  # 正式评估时请删除此行
```

### 通过 Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['ifbench'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```