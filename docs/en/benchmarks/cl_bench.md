# CL-bench


## Overview

CL-bench represents a step towards building LMs with this fundamental capability (Context Learning), making them more intelligent and advancing their deployment in real-world scenarios. This benchmark is specifically designed to evaluate a model's ability to learn specific, often novel or long-tail knowledge directly from the context and apply it to solve problems, simulating real-world learning processes.

**Resources:**
[Homepage](https://github.com/Tencent-Hunyuan/CL-bench) | [Dataset](https://huggingface.co/datasets/tencent/CL-bench)

## Task Description

- **Task Type**: In-Context Learning & Reasoning (Context-dependent QA)
- **Input**: A context containing new rules, fictional information, or specific logic, followed by a query
- **Output**: A solution derived strictly from the provided context (not pre-trained knowledge)
- **Difficulty**: Varied, requiring understanding of novel concepts defined in the prompt

## Key Features

- **Contamination-Free**: Uses synthetic or highly specific data (e.g., fictional laws, new programming syntax) to ensure models cannot rely on memorized training data
- **Real-world Simulation**: Mimics scenarios where humans learn new tasks by reading instructions or documentation
- **Diverse Domains**: Covers logic reasoning, rule following, language understanding, and puzzle solving within a given context
- **Evaluation of Adaptability**: Measures the "learning" capability rather than just "retrieval" or "memory"

## Evaluation Notes

- Focuses on the model's ability to follow strict instructions provided in the context
- Evaluation typically checks if the reasoning process utilizes the unique information given in the prompt
- Answers are often evaluated against specific ground-truth rules defined in the context
- Crucial for assessing how well models can adapt to private data or dynamic environments without fine-tuning


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `cl_bench` |
| **Dataset ID** | [tencent-community/CL-bench](https://modelscope.cn/datasets/tencent-community/CL-bench/summary) |
| **Paper** | N/A |
| **Tags** | `InstructionFollowing`, `Reasoning` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1,899 |
| Prompt Length (Mean) | 42491.73 chars |
| Prompt Length (Min/Max) | 6595 / 320655 chars |

## Sample Example

**Subset**: `default`

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

## Prompt Template

*No prompt template defined.*

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets cl_bench \
    --limit 10  # Remove this line for formal evaluation
```

### Using Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['cl_bench'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


