# OpenAI MRCR


## Overview

MRCR (Memory-Recall with Contextual Retrieval) is OpenAI's benchmark for evaluating retrieval and recall capabilities in long-context scenarios. It tests whether models can correctly extract and use specific information (needles) embedded in long prompts.

## Task Description

- **Task Type**: Long-Context Memory and Retrieval
- **Input**: Long chat conversation with 2, 4, or 8 embedded needles
- **Output**: Exact retrieval of needle content
- **Domains**: Long-context understanding, information retrieval

## Key Features

- Tests retrieval with varying needle counts (2, 4, or 8)
- Evaluates across multiple context length bins (4K to 1M tokens)
- Measures exact recall of embedded information
- Uses sequence ratio scoring for precise matching
- Supports configurable context size filtering

## Evaluation Notes

- Default evaluation uses the **train** split
- Primary metric: **MRCR Score** (sequence ratio)
- Results bucketed by token count:
  - 4K-8K, 8K-16K, 16K-32K, 32K-64K
  - 64K-128K, 128K-256K, 256K-512K, 512K-1M
- Configurable via extra_params: max/min_context_size, needle_count, tokenizer
- Uses tiktoken for accurate token counting


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `openai_mrcr` |
| **Dataset ID** | [openai-mirror/mrcr](https://modelscope.cn/datasets/openai-mirror/mrcr/summary) |
| **Paper** | N/A |
| **Tags** | `LongContext`, `Retrieval` |
| **Metrics** | `mrcr_score` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 2,400 |
| Prompt Length (Mean) | 944363.11 chars |
| Prompt Length (Min/Max) | 15459 / 5244934 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "d0537a08",
      "content": "Here are some examples of conversations succeeded by a follow-up question answered correctly:\n\n======EXAMPLE======\nUser: write a short essay about education\nAssistant: Education is a cornerstone of society, serving as the foundation upon whic ... [TRUNCATED 14670 chars] ... etween creators and audiences, one carefully edited word at a time.\n\n**Contact Information:**\n\nFor more information on the role of editors in publishing, please contact [Your Contact Information].\n\n**End of Article**\n\n======END EXAMPLE======\n"
    },
    {
      "id": "0a4d44d6",
      "content": "write a email about horses"
    },
    {
      "id": "6b6a9ec1",
      "content": "Subject: Let's Talk About Horses!\n\nHi [Recipient's Name],\n\nI hope this email finds you well! I wanted to share some thoughts and insights about a subject that I find truly fascinating: horses.\n\nHorses have been integral to human history for t ... [TRUNCATED 943 chars] ...  volunteering at equestrian centers.\n\nPlease feel free to share your experiences or any questions you might have about horses. I’d love to hear what interests you most about them!\n\nLooking forward to your thoughts.\n\nBest regards,\n\n[Your Name]"
    },
    {
      "id": "09e6442d",
      "content": "write a social media post about records"
    },
    {
      "id": "d045617f",
      "content": "🌟✨ Dive into the World of Records! 🎶📀\n\nHey, music lovers! 🎧 Did you know that records are making a major comeback? From their rich sound quality to their iconic artwork, vinyl is not just about music; it's an experience! 🎵📚\n\nWhether you're a  ... [TRUNCATED 230 chars] ... pins with us! 📸🤩 Let's celebrate the timeless magic of vinyl together. Tag a friend who needs to know about this revival! 🎤💫\n\n#VinylRevival #RecordLovers #SpinThatVinyl #MusicMagic #AnalogVibes\n\n🎶 Drop your favorite record in the comments! 🎤👇"
    },
    {
      "id": "7e20ad5e",
      "content": "write a email about departments"
    },
    {
      "id": "4b9162de",
      "content": "Sure, I can help you draft an email about departments. Could you please provide some details about the specific purpose of the email? Are you introducing new department structures, outlining changes, or simply describing different departments within an organization? Knowing this will help tailor the email to your needs."
    },
    {
      "id": "1a818762",
      "content": "write a poem about fields"
    },
    {
      "id": "f6dcf345",
      "content": "In the heart of the gentle rolling plains,  \nWhere the horizon meets the sky,  \nFields stretch wide, like endless lanes,  \nBeneath the sun's watchful eye.\n\nGolden grains sway with the breeze,  \nA whispering sea of amber hues,  \nDancing with a ... [TRUNCATED 546 chars] ... s and blessings dry,  \nIn every furrow, in every knell,  \nEcho tales of ages gone by.\n\nSo let us wander through these lands,  \nWhere nature’s beauty softly sings,  \nAnd feel the earth beneath our hands,  \nIn fields where life forever springs."
    },
    {
      "id": "a0d56b6e",
      "content": "write a social media post about force"
    },
    "... [TRUNCATED 762 more items] ..."
  ],
  "target": "mWEa9DrPT3**Verse 1**  \nIn a world so vast and wide,  \nWe walk on paths unknown.  \nTwo souls meeting eye to eye,  \nSeeds of trust are sown.  \nWhispers of understanding,  \nIn the gentle breeze they glide,  \nPromises of commitment,  \nAs we stan ... [TRUNCATED 1435 chars] ...  \nSo here’s to the agreements,  \nThe vows that bind us tight.  \nFor in our shared convictions,  \nWe'll march into the night.  \nHand in hand we journey,  \nOn a path we've made our own.  \nWith every heart’s agreement,  \nTogether, we have grown.",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "random_string_to_prepend": "mWEa9DrPT3",
    "n_needles": 2,
    "desired_msg_index": 721,
    "total_messages": 772,
    "n_chars": 708925,
    "raw_input_tok_cnt": 146252,
    "bin_index": 5
  }
}
```

## Prompt Template

*No prompt template defined.*

## Extra Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_context_size` | `int | null` | `None` | Maximum context tokens; samples exceeding are skipped. Defaults to None (no limit). |
| `min_context_size` | `int | null` | `None` | Minimum context tokens; samples below are skipped. Defaults to None (no limit). |
| `needle_count` | `list[int] | null` | `None` | Needle count filter (allowed: 2,4,8). Must be a list, e.g., [2], [4], or [2, 4, 8].  None keeps all. |
| `tik_enc` | `str` | `o200k_base` | tiktoken encoding name used for token counting. |
| `prefix_filter` | `str | null` | `None` | Regex pattern to filter answers. Defaults to None (no filtering). |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets openai_mrcr \
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
    datasets=['openai_mrcr'],
    dataset_args={
        'openai_mrcr': {
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


