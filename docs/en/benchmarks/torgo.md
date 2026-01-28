# TORGO


## Overview

TORGO is a specialized database of dysarthric speech designed for evaluating ASR systems on speakers with motor speech disorders. It contains aligned acoustic and articulatory data from speakers with cerebral palsy (CP) or amyotrophic lateral sclerosis (ALS).

## Task Description

- **Task Type**: Dysarthric Speech Recognition
- **Input**: Audio recordings from speakers with speech disorders
- **Output**: Transcribed text
- **Focus**: Accessibility and inclusive ASR evaluation

## Key Features

- Specialized dataset for dysarthric speech
- Speakers with cerebral palsy (CP) or ALS
- Intelligibility-based subsets (mild, moderate, severe)
- 3D articulatory feature alignment
- Important for accessibility research

## Evaluation Notes

- Default configuration uses **test** split
- Subsets by intelligibility: **mild**, **moderate**, **severe**
- Metrics: **CER** (Character Error Rate), **WER** (Word Error Rate), **SemScore**
- Requires `jiwer` package for CER/WER metrics
- Requires `jellyfish` package for SemScore metric
- Supports batch scoring for efficiency


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `torgo` |
| **Dataset ID** | [extraordinarylab/torgo](https://modelscope.cn/datasets/extraordinarylab/torgo/summary) |
| **Paper** | N/A |
| **Tags** | `Audio`, `SpeechRecognition` |
| **Metrics** | `cer`, `wer`, `sem_score` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 5,553 |
| Prompt Length (Mean) | 67 chars |
| Prompt Length (Min/Max) | 67 / 67 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `mild` | 1,479 | 67 | 67 | 67 |
| `moderate` | 1,666 | 67 | 67 | 67 |
| `severe` | 2,408 | 67 | 67 | 67 |

**Audio Statistics:**

| Metric | Value |
|--------|-------|
| Total Audio Files | 5,553 |
| Audio per Sample | min: 1, max: 1, mean: 1 |
| Formats | wav |


## Sample Example

**Subset**: `mild`

```json
{
  "input": [
    {
      "id": "1220f252",
      "content": [
        {
          "text": "Please recognize the speech and only output the recognized content:"
        },
        {
          "audio": "[BASE64_AUDIO: wav, ~89.1KB]",
          "format": "wav"
        }
      ]
    }
  ],
  "target": "FEE",
  "id": 0,
  "group_id": 0,
  "subset_key": "mild",
  "metadata": {
    "transcript": "FEE",
    "intelligibility": "mild",
    "duration": 2.8499999046325684
  }
}
```

## Prompt Template

**Prompt Template:**
```text
Please recognize the speech and only output the recognized content:
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets torgo \
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
    datasets=['torgo'],
    dataset_args={
        'torgo': {
            # subset_list: ['mild', 'moderate', 'severe']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


