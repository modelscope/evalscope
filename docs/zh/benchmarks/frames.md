# FRAMES


## 概述

FRAMES 是一个全面的评估数据集，旨在测试检索增强生成（Retrieval-Augmented Generation, RAG）系统的能力。它在长上下文场景中评估事实性、检索准确性和推理能力。

## 任务描述

- **任务类型**：RAG 评估 / 长上下文问答（Long-Context QA）
- **输入**：维基百科上下文文档 + 问题
- **输出**：指定格式的事实性答案
- **领域**：事实性、检索、多跳推理

## 主要特点

- 测试 RAG 的核心能力：事实性、检索、推理
- 提供源自维基百科的上下文文档
- 问题需要综合多个来源的信息
- 同时评估检索质量和答案生成效果
- 支持精确匹配（exact match）和大语言模型（LLM）裁判两种评估方式

## 评估说明

- 默认使用 **test** 数据划分进行评估
- 主要指标：**准确率（Accuracy）**，结合精确匹配和 LLM 裁判
- 回答格式：`"Therefore, the answer is (answer here)"`
- 精确匹配采用标准化后的答案比较
- LLM 裁判提供灵活的语义匹配能力

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `frames` |
| **数据集ID** | [iic/frames](https://modelscope.cn/datasets/iic/frames/summary) |
| **论文** | N/A |
| **标签** | `LongContext`, `Reasoning` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 824 |
| 提示词长度（平均） | 67976.16 字符 |
| 提示词长度（最小/最大） | 235 / 557427 字符 |

## 样例示例

**子集**: `test`

```json
{
  "input": [
    {
      "id": "6845108b",
      "content": "Please read the following text and answer the question below.\n\n<text>\nPresident_of_the_United_States\nThe president of the United States (POTUS) is the head of state and head of government of the United States of America. The president directs ... [TRUNCATED] ... irst lady of the United States' mother and her surname is the same as the second assassinated president's mother's maiden name, what is my future wife's name? \n\nFormat your response as follows: \"Therefore, the answer is (insert answer here)\"."
    }
  ],
  "target": "Jane Ballou",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "context": "President_of_the_United_States\nThe president of the United States (POTUS) is the head of state and head of government of the United States of America. The president directs the executive branch of the federal government and is the commander-i ... [TRUNCATED] ... Garfield from the U.S. National Library of Medicine. Contains medical bulletins issued by attending physicians D. Hayes Agnes, J.K. Barnes, D. W. Bliss, Frank H. Hamilton, Robert Reyburn, and J.J. Woodward between July 6 – September 19, 1881.",
    "wiki_items": [
      {
        "title": "President_of_the_United_States",
        "link": "https://en.wikipedia.org/wiki/President_of_the_United_States",
        "frames_prompt_id": "[0, 261]",
        "original_link": "['https://en.wikipedia.org/wiki/President_of_the_United_States', 'https://en.wikipedia.org/wiki/President_of_the_United_States#History_and_development']",
        "text": "The president of the United States (POTUS) is the head of state and head of government of the United States of America. The president directs the executive branch of the federal government and is the commander-in-chief of the United States Ar ... [TRUNCATED] ... assuming office.\n\nSee also\nOutline of American politics\n\nNotes\nReferences\nFurther reading\nExternal links\n\nWhite House homepage\nUnited States Presidents Collection. General Collection, Beinecke Rare Book and Manuscript Library, Yale University"
      },
      {
        "title": "James_Buchanan",
        "link": "https://en.wikipedia.org/wiki/James_Buchanan",
        "frames_prompt_id": "[0]",
        "original_link": "['https://en.wikipedia.org/wiki/James_Buchanan']",
        "text": "James Buchanan Jr. ( bew-KAN-ən; April 23, 1791 – June 1, 1868) was the 15th president of the United States, serving from 1857 to 1861. Buchanan also served as the secretary of state from 1845 to 1849 and represented Pennsylvania in both hous ... [TRUNCATED] ... Letters Shapell Manuscript Foundation\nMr. Buchanans Administration on the Eve of the Rebellion. President Buchanans memoirs.\nInaugural Address Archived August 9, 2020, at the Wayback Machine\nFourth Annual Message to Congress, December 3, 1860"
      },
      {
        "title": "Harriet_Lane",
        "link": "https://en.wikipedia.org/wiki/Harriet_Lane",
        "frames_prompt_id": "[0]",
        "original_link": "['https://en.wikipedia.org/wiki/Harriet_Lane']",
        "text": "Harriet Rebecca Lane Johnston (May 9, 1830 – July 3, 1903) acted as first lady of the United States during the administration of her uncle, lifelong bachelor president James Buchanan, from 1857 to 1861. She has been described as the first of  ... [TRUNCATED] ... nan Dying (play). (Ms. Johnston is a character in Updike's fictional play about President Buchanan.)\n\nExternal links\nWorks by or about Harriet Lane at the Internet Archive\n\"Harriet Lane\". First Ladies: Influence & Image. firstladies.org. CNN."
      },
      {
        "title": "List_of_presidents_of_the_United_States_who_died_in_office",
        "link": "https://en.wikipedia.org/wiki/List_of_presidents_of_the_United_States_who_died_in_office",
        "frames_prompt_id": "[0]",
        "original_link": "['https://en.wikipedia.org/wiki/List_of_presidents_of_the_United_States_who_died_in_office']",
        "text": "Since the office was established in 1789, 45 persons have served as president of the United States. Of these, eight have died in office: four were assassinated, and four died of natural causes. In each of these instances, the vice president h ... [TRUNCATED] ... te University Press. ISBN 0-87338-210-2.\nVowell, Sarah (2005). Assassination Vacation. Simon and Schuster. ISBN 0-7432-6003-1.\n\nExternal links\nThe Mortal Presidency Archived June 3, 2015, at the Wayback Machine (Shapell Manuscript Foundation)"
      },
      {
        "title": "James_A._Garfield",
        "link": "https://en.wikipedia.org/wiki/James_A._Garfield",
        "frames_prompt_id": "[0]",
        "original_link": "['https://en.wikipedia.org/wiki/James_A._Garfield']",
        "text": "James Abram Garfield (November 19, 1831 – September 19, 1881) was the 20th president of the United States, serving from March 1881 until his assassination in September that year. A preacher, lawyer, and Civil War general, Garfield served nine ... [TRUNCATED] ... Garfield from the U.S. National Library of Medicine. Contains medical bulletins issued by attending physicians D. Hayes Agnes, J.K. Barnes, D. W. Bliss, Frank H. Hamilton, Robert Reyburn, and J.J. Woodward between July 6 – September 19, 1881."
      }
    ]
  }
}
```

*注：部分内容为显示目的已截断。*

## 提示模板

**提示模板：**
```text
Please read the following text and answer the question below.

<text>
{context}
</text>

{question}

Format your response as follows: "Therefore, the answer is (insert answer here)".
```

## 使用方法

### 使用命令行（CLI）

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets frames \
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
    datasets=['frames'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```