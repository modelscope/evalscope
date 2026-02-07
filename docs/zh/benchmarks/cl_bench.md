# CL-bench

## 概述

CL-bench 旨在推动构建具备上下文学习（Context Learning）这一基础能力的大语言模型，使其更加智能，并促进其在现实场景中的部署。该基准测试专门用于评估模型能否直接从上下文中学习特定的、通常是新颖或长尾的知识，并将其应用于解决问题，从而模拟真实世界中的学习过程。

**资源：**  
[主页](https://github.com/Tencent-Hunyuan/CL-bench) | [数据集](https://huggingface.co/datasets/tencent/CL-bench)

## 任务描述

- **任务类型**：上下文学习与推理（上下文相关的问答）
- **输入**：包含新规则、虚构信息或特定逻辑的上下文，后接一个查询
- **输出**：严格基于所提供上下文得出的答案（而非依赖预训练知识）
- **难度**：多样，要求理解提示中定义的新颖概念

## 核心特性

- **无污染性**：使用合成数据或高度特定的数据（例如虚构法律、新编程语法），确保模型无法依赖记忆中的训练数据
- **真实场景模拟**：模拟人类通过阅读说明或文档来学习新任务的情境
- **多领域覆盖**：涵盖在给定上下文内的逻辑推理、规则遵循、语言理解和谜题求解
- **适应性评估**：衡量模型的“学习”能力，而不仅仅是“检索”或“记忆”能力

## 评估说明

- 重点关注模型是否能严格遵循上下文中提供的指令
- 评估通常检查推理过程是否利用了提示中给出的独特信息
- 答案通常根据上下文中定义的具体真实规则进行评判
- 对评估模型在无需微调的情况下适应私有数据或动态环境的能力至关重要

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `cl_bench` |
| **数据集ID** | [tencent-community/CL-bench](https://modelscope.cn/datasets/tencent-community/CL-bench/summary) |
| **论文** | 无 |
| **标签** | `InstructionFollowing`, `Reasoning` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估分割** | `train` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,899 |
| 提示词长度（平均） | 42491.73 字符 |
| 提示词长度（最小/最大） | 6595 / 320655 字符 |

## 样例示例

**子集**：`default`

```json
{
  "input": [
    {
      "id": "b0a2c972",
      "content": "You are an AI designed to play the Twisted Cryptids board Game developed by Unstable Unicorns. Your purpose is to teach rules, host/referee a session, simulate AI opponents if requested, track game state, and provide strategy advice for the g ... [TRUNCATED 861 chars] ...  brief rationale, then a single short quip).\n\nYou take the role of an eccentric conspiracy theorist when talking to the user, but constrain your tone to brief, optional quips that never obscure or delay rules, turn summaries, or move prompts.",
      "role": "system"
    },
    {
      "id": "109b41de",
      "content": "RULE BOOK\nTM\nTM\nIt’s so hard to feel seen as a Cryptid these days. In this easy-to-learn strategy game,\nyou’ll navigate your love-hate relationship with humans and outwit your fellow Cryptids\nto earn your status as a true legend. As hikers, h ... [TRUNCATED 158305 chars] ...  game\n02:23:53.880 designers what the gloi cryp it is to be fair we still have a dark horse because\n02:24:00.560 uh the worm is still underway yeah and who knows how gloopy they will\n\nThis is my first time playing, what do Sightings Cards do?",
      "role": "user"
    }
  ],
  "target": [
    "The response should define what a Sighting card is and its role when playing Twisted Cryptids. For example, Sighting cards are revealed to award/penalize Myth and trigger effects that may move Humans.",
    "The response should name the four different sighting card types, namely: Decoys, Hoaxes, Silhouettes, and Real Deals.",
    "The response should state the requirements for when sighting cards can be revealed: for example, during the dusk phase of each round, during an encounter at a site in the wilderness, when a player has at least 1 hiding spot at the encounter site.",
    "The response should explain how sighting cards are scored once revealed. For example, by gaining or losing myth that depends on the distribution of human types present at the site of the encounter, as specified on the sighting card.",
    "The response should explain the following gameplay sequence in order, which happens when a sighting is going to be revealed: 1. reveal the top face down sighting card in the corresponding stack; 2. apply myth gain/loss shown on the card; 3. r ... [TRUNCATED 161 chars] ... lace it on its corresponding stack (or if that sighting reveal emptied the stack, return the hiding spot token to the box for the rest of the game); 6. the next player who has a hiding spot at the site of the encounter reveals their sighting.",
    "The response should state that players can look at their own sighting cards at any time, but other player's sightings remain hidden until revealed.",
    "The response should state that a player can gain 7 Myth from any of their real deal sighting cards that have not been revealed by the end of the game (after 5 rounds).",
    "The response should explain how the four different sighting card types work. For example,\r\n1. Decoys: Are objects arranged to look just like you! Use them to trick Hunters and earn Myth.\r\n2. Hoaxes: Enlist the help of woodland creatures to de ... [TRUNCATED 239 chars] ... lly you! Expose yourself attacking, sleeping, or engaged in your favorite activity to the right Humans and get rewarded with Myth for your bravery. But beware, you could lose a lot of Myth if the wrong Humans are at the Site of the Encounter.",
    "The response should state that each player has nine sighting cards in Twisted Cryptids.",
    "The response should state that at the start of the game each player forms 3 stacks of sighting cards that correspond to each player's 3 hiding spot tokens they placed on the board.",
    "... [TRUNCATED 4 more items] ..."
  ],
  "id": 0,
  "group_id": 0,
  "metadata": {
    "task_id": "71a2cd92-6978-4ea8-a37f-d99728129d89",
    "context_category": "Rule System Application",
    "sub_category": "Game Mechanics"
  }
}
```

## 提示模板

*未定义提示模板。*

## 使用方法

### 使用命令行接口（CLI）

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets cl_bench \
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
    datasets=['cl_bench'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```