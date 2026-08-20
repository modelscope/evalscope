# HellaSwag-Hindi


## 概述

HellaSwag-Hindi 是 HellaSwag 常识句子补全基准测试完整验证集的印地语翻译版本。其中上下文主干保持英文不变，四个候选续写选项被翻译为印地语，因此模型需要将一个英文场景与其最合理的印地语结尾相匹配。该数据集源自 `ai4bharat/hellaswag-translated`（规范名称；旧 ID `ai4bharat/hellaswag-hi` 会重定向至此），与 lighteval 的 `community_hellaswag_hin` 任务所使用的数据集相同。

## 任务描述

- **任务类型**：常识句子补全（混合语言）
- **输入**：一个英文上下文句子，附带四个印地语候选续写选项
- **输出**：正确答案的字母
- **覆盖范围**：完整的 HellaSwag 验证集（10,042 个样例）

## 主要特性

- 完整的 HellaSwag 验证集：包含 10,042 个带有标准答案标签的样例
- 英文上下文主干与印地语翻译的候选结尾配对（混合语言设置）
- 与 lighteval 的 `community_hellaswag_hin` 任务套件使用相同的数据集

## 评估说明

- 默认配置采用 **0-shot** 评估（使用验证集，这是唯一提供标准答案标签的划分 —— HellaSwag 的 `test` 划分不包含标准答案）
- 默认从 ModelScope 加载（镜像为 `ai4bharat/hellaswag-translated`），无需访问令牌

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `hellaswag_hi` |
| **数据集ID** | [ai4bharat/hellaswag-translated](https://modelscope.cn/datasets/ai4bharat/hellaswag-translated/summary) |
| **论文** | 无 |
| **标签** | `MCQ`, `Reasoning` |
| **指标** | `accuracy` |
| **默认示例数** | 0-shot |
| **评估划分** | `validation` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 10,042 |
| 提示词长度（平均） | 1021.77 字符 |
| 提示词长度（最小/最大） | 367 / 1977 字符 |

## 样例示例

**子集**: `hi`

```json
{
  "input": [
    {
      "id": "f96132ec",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D.\n\nA man is sitting on a roof. he\n\nA) वह स्की की एक जोड़ी को लपेटने के लिए रैप का उपयोग कर रहा है।\nB) यह स्तर की टाइलों को चीर रहा है।\nC) वह एक रूबिक क्यूब पकड़े हुए है।\nD) एक छत पर छत खींचना शुरू करता है।"
    }
  ],
  "choices": [
    "वह स्की की एक जोड़ी को लपेटने के लिए रैप का उपयोग कर रहा है।",
    "यह स्तर की टाइलों को चीर रहा है।",
    "वह एक रूबिक क्यूब पकड़े हुए है।",
    "एक छत पर छत खींचना शुरू करता है।"
  ],
  "target": "D",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "activity_label": "Roof shingle removal"
  }
}
```

## 提示模板

**提示模板：**
```text
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}.

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
    --datasets hellaswag_hi \
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
    datasets=['hellaswag_hi'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
