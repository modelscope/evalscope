# MaritimeBench

## 概述

MaritimeBench 是一个用于评估 AI 模型在中文海事相关多项选择题上表现的基准测试。该基准包含与海事知识、航海、船舶工程及海上作业相关的专业问题。

## 任务描述

- **任务类型**：海事知识多项选择问答（中文）
- **输入**：一道包含 4 个选项（A-D）的海事相关问题
- **输出**：正确答案的字母，格式为方括号内 [A/B/C/D]
- **语言**：中文

## 主要特点

- 面向海事领域的专业问题
- 中文语言评估
- 包含多个覆盖不同海事主题的子集
- 考察专业的海事知识
- 采用标准化的中文考试格式

## 评估说明

- 默认配置使用 **0-shot** 评估
- 在测试集（test split）上进行评估
- 使用简单准确率（accuracy）作为评估指标
- 通过正则表达式提取方括号格式的答案 [A/B/C/D]

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `maritime_bench` |
| **数据集 ID** | [HiDolphin/MaritimeBench](https://modelscope.cn/datasets/HiDolphin/MaritimeBench/summary) |
| **论文** | N/A |
| **标签** | `Chinese`, `Knowledge`, `MCQ` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,888 |
| 提示词长度（平均） | 259.04 字符 |
| 提示词长度（最小/最大） | 173 / 717 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "0b8c5f63",
      "content": "请回答单选题。要求只输出选项，不输出解释，将选项放在[]里，直接输出答案。示例：\n\n题目：在船舶主推进动力装置中，传动轴系在运转中承受以下复杂的应力和负荷，但不包括______。\n选项：\nA. 电磁力\nB. 压拉应力\nC. 弯曲应力\nD. 扭应力\n答：[A]\n 当前题目\n 船舶电力系统供电网络中，放射形网络的特点是______。①发散形传输②环形传输③缺乏冗余④冗余性能好\n选项：\nA. ②③\nB. ①③\nC. ②④\nD. ①④"
    }
  ],
  "choices": [
    "②③",
    "①③",
    "②④",
    "①④"
  ],
  "target": "B",
  "id": 0,
  "group_id": 0
}
```

## 提示模板

**提示模板：**
```text
请回答单选题。要求只输出选项，不输出解释，将选项放在[]里，直接输出答案。示例：

题目：在船舶主推进动力装置中，传动轴系在运转中承受以下复杂的应力和负荷，但不包括______。
选项：
A. 电磁力
B. 压拉应力
C. 弯曲应力
D. 扭应力
答：[A]
 当前题目
 {question}
选项：
{choices}
```

## 使用方法

### 使用命令行（CLI）

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets maritime_bench \
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
    datasets=['maritime_bench'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```