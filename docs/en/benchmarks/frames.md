# FRAMES


## Overview

FRAMES is a comprehensive evaluation dataset designed to test the capabilities of Retrieval-Augmented Generation (RAG) systems. It evaluates factuality, retrieval accuracy, and reasoning abilities in long-context scenarios.

## Task Description

- **Task Type**: RAG Evaluation / Long-Context QA
- **Input**: Wikipedia context documents + question
- **Output**: Factual answer in specified format
- **Domains**: Factuality, retrieval, multi-hop reasoning

## Key Features

- Tests core RAG capabilities: factuality, retrieval, reasoning
- Provides Wikipedia-sourced context documents
- Questions require synthesizing information from multiple sources
- Evaluates both retrieval quality and answer generation
- Supports both exact match and LLM judge evaluation

## Evaluation Notes

- Default evaluation uses the **test** split
- Primary metric: **Accuracy** with both exact match and LLM judge
- Response format: "Therefore, the answer is (answer here)"
- Uses normalized answer comparison for exact matching
- LLM judge provides flexible semantic matching


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `frames` |
| **Dataset ID** | [iic/frames](https://modelscope.cn/datasets/iic/frames/summary) |
| **Paper** | N/A |
| **Tags** | `LongContext`, `Reasoning` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 824 |
| Prompt Length (Mean) | 67976.16 chars |
| Prompt Length (Min/Max) | 235 / 557427 chars |

## Sample Example

**Subset**: `test`

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

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
Please read the following text and answer the question below.

<text>
{context}
</text>

{question}

Format your response as follows: "Therefore, the answer is (insert answer here)".
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets frames \
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
    datasets=['frames'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


