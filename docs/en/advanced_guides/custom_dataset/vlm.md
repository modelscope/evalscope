# Multimodal Large Models

This framework supports two custom multimodal evaluation methods:

- **General-VQA Format**: Suitable for Q&A-based multimodal evaluation tasks. Supports two input styles: **OpenAI Messages Data**, **MMMU-style Data with Media Placeholders**.
- **General-VMCQ Format**: Suitable for multiple-choice multimodal evaluation tasks. Uses the [media placeholders][mp-feature] to embed images, videos, and audio in questions and options, similar to MMMU format.

## General-VQA Format

General-VQA supports **two input styles**:

1. **OpenAI Messages Data** — full structured content with explicit media parts (images, audio, video) in the OpenAI message schema. Supports multi-turn conversations, system prompts, and fine-grained control over each content part.
2. **MMMU-style Data with Media Placeholders** — a simpler approach where the user message is a plain-text string containing `<image N>`, `<video N>`, or `<audio N>` placeholders, and media files are supplied via separate indexed columns (see [Media Placeholder Mechanism][mp-feature]).

Both formats support **JSONL** or **TSV** files.

### OpenAI Messages Data

In this format, each record contains a `messages` array following the OpenAI chat completion schema. Media (images, audio, video) are embedded directly as structured content parts within user messages.

**JSONL Example** (`example_openai.jsonl`):
```json
{"messages": [{"role": "user", "content": [{"type": "text", "text": "What animal is this?"}, {"type": "image_url", "image_url": {"url": "custom_eval/multimodal/images/dog.jpg"}}]}], "answer": "Dog"}
{"messages": [{"role": "user", "content": [{"type": "text", "text": "What building is this?"}, {"type": "image_url", "image_url": {"url": "custom_eval/multimodal/images/AMNH.jpg"}}]}], "answer": "Museum"}
```

**TSV Example** (`example_openai.tsv`):
```text
messages	answer
[{"role": "user", "content": [{"type": "text", "text": "What animal is this?"}, {"type": "image_url", "image_url": {"url": "custom_eval/multimodal/images/dog.jpg"}}]}]	Dog
[{"role": "user", "content": [{"type": "text", "text": "What building is this?"}, {"type": "image_url", "image_url": {"url": "custom_eval/multimodal/images/AMNH.jpg"}}]}]	Museum
```

**Field Descriptions**:
- `messages`: OpenAI format message array, supporting:
  - Text content: `{"type": "text", "text": "question text"}`
  - Image URL: `{"type": "image_url", "image_url": {"url": "path or base64"}}`
  - Audio input: `{"type": "input_audio", "input_audio": {"data": "path or base64", "format": "wav"}}`
  - Video URL: `{"type": "video_url", "video_url": {"url": "path or base64"}}`
  - System message: `{"role": "system", "content": "system prompt"}`
- `answer`: Reference answer (optional, used to calculate BLEU and Rouge scores)

**Supported Image Formats**:
- Local path: `"url": "custom_eval/multimodal/images/dog.jpg"`
- HTTP URL: `"url": "https://example.com/image.jpg"` (requires model service support)
- Base64 encoding: `"url": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."`

**Supported Audio Formats**:
- Local path: `"data": "custom_eval/multimodal/audio/sample.wav"`
- Base64 encoding: `"data": "data:audio/wav;base64,UklGRiQ..."`
- Audio format (`format` field): supports `"wav"` and `"mp3"`

**Supported Video Formats**:
- Local path: `"url": "custom_eval/multimodal/videos/sample.mp4"`
- HTTP URL: `"url": "https://example.com/video.mp4"` (requires model service support)
- Base64 encoding: `"url": "data:video/mp4;base64,AAAAIGZ0eX..."`
- Video format is inferred from the path, URL, or data URI; supported formats are `"mp4"`, `"mpeg"`, `"mov"`, and `"avi"`.

**Multi-image Input**

Supports using multiple images in one question:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Compare these two images:"},
        {"type": "image_url", "image_url": {"url": "image1.jpg"}},
        {"type": "text", "text": "and"},
        {"type": "image_url", "image_url": {"url": "image2.jpg"}},
        {"type": "text", "text": "What are the differences?"}
      ]
    }
  ],
  "answer": "The main differences are..."
}
```

**System Prompt**

You can add system messages to set the evaluation context:

```json
{
  "messages": [
    {"role": "system", "content": "You are a medical AI assistant."},
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Analyze this X-ray:"},
        {"type": "image_url", "image_url": {"url": "xray.jpg", "detail": "high"}}
      ]
    }
  ],
  "answer": "The X-ray shows..."
}
```

**Base64 Images**

Supports directly using base64 encoded images:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What's in this image?"},
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD..."
          }
        }
      ]
    }
  ],
  "answer": "A beautiful landscape"
}
```

**Audio Input**

Supports audio content input using OpenAI `input_audio` format. The `data` field accepts either a local file path or base64-encoded data. The `format` field supports `"wav"` and `"mp3"`:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe the content of this audio clip."},
        {
          "type": "input_audio",
          "input_audio": {
            "data": "custom_eval/multimodal/audio/sample.wav",
            "format": "wav"
          }
        }
      ]
    }
  ],
  "answer": "A piano music performance."
}
```

You can also use base64-encoded audio data:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What is being said in this audio?"},
        {
          "type": "input_audio",
          "input_audio": {
            "data": "UklGRiQAAABXQVZFZm10IBAAAA...",
            "format": "wav"
          }
        }
      ]
    }
  ],
  "answer": "Hello, world."
}
```

**Video Input**

Supports video content input using OpenAI-compatible `video_url` format. The `url` field accepts either a local file path, HTTP URL, or base64-encoded data URL:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe the content of this video clip."},
        {
          "type": "video_url",
          "video_url": {
            "url": "custom_eval/multimodal/videos/sample.mp4"
          }
        }
      ]
    }
  ],
  "answer": "A short video clip."
}
```

### MMMU-style Data with Media Placeholders

In this format, the `messages` field keeps the user message as a **plain-text string** containing placeholders like `<image 1>`, `<video 1>`, or `<audio 1>` and filled in-place via separate indexed columns (`image_1`, `video_1`, `audio_1`, etc.). For each media type, use either indexed columns or its plural list column, not both. See the [Media Placeholder Mechanism][mp-feature] section below for full details on how placeholders are resolved, supported column names, and media types.

**JSONL Example** (`example_placeholder.jsonl`):
```json
{"messages": [{"role": "user", "content": "What animal is this?<image 1>"}], "image_1": "custom_eval/multimodal/images/dog.jpg", "answer": "Dog"}
{"messages": [{"role": "user", "content": "What building is this?<image 1>"}], "image_1": "custom_eval/multimodal/images/AMNH.jpg", "answer": "Museum"}
{"messages": [{"role": "user", "content": "Which city's skyline is this?<image 1>"}], "image_1": "custom_eval/multimodal/images/tokyo.jpg", "answer": "Tokyo"}
{"messages": [{"role": "user", "content": "What is the brand of this car?<image 1>"}], "image_1": "custom_eval/multimodal/images/tesla.jpg", "answer": "Tesla"}
{"messages": [{"role": "user", "content": "What is the person in the picture doing?<image 1>"}], "image_1": "custom_eval/multimodal/images/running.jpg", "answer": "Running"}
```

**Mixed Media Example**:
```json
{"messages": [{"role": "user",
               "content": "<image 1> Watch <video 1> and describe both."}],
 "answer": "A sunny beach and a wave video.",
 "image_1": "custom_eval/multimodal/images/beach.jpg",
 "video_1": "custom_eval/multimodal/videos/wave.mp4"}
```

**Note**: Only user messages (`"role": "user"`) with plain-text `content` are scanned for placeholders. Messages that already have structured content (a list of content parts) or messages with other roles (system, assistant, tool) are left untouched.


### 2. Configure Evaluation Task

Evaluate using Python API or CLI:

**Python API**:
```python
from evalscope.run import run_task
from evalscope.config import TaskConfig
from os import environ as env

task_cfg = TaskConfig(
    model='qwen-vl-plus',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=env.get('DASHSCOPE_API_KEY'),
    eval_type='openai_api',
    datasets=['general_vqa'],
    dataset_args={
        'general_vqa': {
            'local_path': 'custom_eval/multimodal/vqa',  # Dataset directory
            'subset_list': ['example_openai'],  # Filename (without extension)
        }
    },
    limit=5,  # Optional: limit number of evaluation samples
)

result = run_task(task_cfg=task_cfg)
```

**CLI**:
```bash
evalscope eval \
    --model qwen-vl-plus \
    --api-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
    --api-key "$DASHSCOPE_API_KEY" \
    --eval-type openai_api \
    --datasets general_vqa \
    --dataset-args '{"general_vqa": {"local_path": "custom_eval/multimodal/vqa", "subset_list": ["example_openai"]}}' \
    --limit 5
```

Evaluation will output BLEU and Rouge metrics:
```text
+--------------+-------------+----------------+----------------+-------+---------+---------+
| Model        | Dataset     | Metric         | Subset         |   Num |   Score | Cat.0   |
+==============+=============+================+================+=======+=========+=========+
| qwen-vl-plus | General-VQA | BLEU ↑ · 1              | example_openai |     5 |    0.7% | default |
+--------------+-------------+----------------+----------------+-------+---------+---------+
| qwen-vl-plus | General-VQA | BLEU ↑ · 2              | example_openai |     5 |      0% | default |
+--------------+-------------+----------------+----------------+-------+---------+---------+
| qwen-vl-plus | General-VQA | BLEU ↑ · 3              | example_openai |     5 |      0% | default |
+--------------+-------------+----------------+----------------+-------+---------+---------+
| qwen-vl-plus | General-VQA | BLEU ↑ · 4              | example_openai |     5 |      0% | default |
+--------------+-------------+----------------+----------------+-------+---------+---------+
| qwen-vl-plus | General-VQA | ROUGE ↑ · 1 · Recall    | example_openai |     5 |     40% | default |
+--------------+-------------+----------------+----------------+-------+---------+---------+
| qwen-vl-plus | General-VQA | ROUGE ↑ · 1 · Precision | example_openai |     5 |    0.6% | default |
+--------------+-------------+----------------+----------------+-------+---------+---------+
| qwen-vl-plus | General-VQA | ROUGE ↑ · 1 · F1        | example_openai |     5 |    1.2% | default |
+--------------+-------------+----------------+----------------+-------+---------+---------+
| qwen-vl-plus | General-VQA | ROUGE ↑ · 2 · Recall    | example_openai |     5 |      0% | default |
+--------------+-------------+----------------+----------------+-------+---------+---------+
| qwen-vl-plus | General-VQA | ROUGE ↑ · 2 · Precision | example_openai |     5 |      0% | default |
+--------------+-------------+----------------+----------------+-------+---------+---------+
| qwen-vl-plus | General-VQA | ROUGE ↑ · 2 · F1        | example_openai |     5 |      0% | default |
+--------------+-------------+----------------+----------------+-------+---------+---------+
| qwen-vl-plus | General-VQA | ROUGE ↑ · L · Recall    | example_openai |     5 |     40% | default |
+--------------+-------------+----------------+----------------+-------+---------+---------+
| qwen-vl-plus | General-VQA | ROUGE ↑ · L · Precision | example_openai |     5 |    0.5% | default |
+--------------+-------------+----------------+----------------+-------+---------+---------+
| qwen-vl-plus | General-VQA | ROUGE ↑ · L · F1        | example_openai |     5 |    0.9% | default |
+--------------+-------------+----------------+----------------+-------+---------+---------+
```

### 3. Configure Judge Model

You can specify a judge model through the `judge_model` parameter to generate reference answers for evaluation, which will obtain accuracy metrics:

```python
from evalscope.run import run_task
from evalscope.constants import EvalType
from os import environ as env

task_cfg = TaskConfig(
    model='qwen-vl-plus',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=env.get('DASHSCOPE_API_KEY'),
    eval_type='openai_api',
    datasets=['general_vqa'],
    dataset_args={
        'general_vqa': {
            'local_path': 'custom_eval/multimodal/vqa',
            'subset_list': ['example_openai'],
        }
    },
    limit=5,
    judge={
        'strategy': 'llm',
        'models': {
            'model_id': 'qwen-plus',  # Does not need to be a multimodal model
            'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
            'api_key': env.get('DASHSCOPE_API_KEY'),
            'generation_config': {'temperature': 0.0, 'max_tokens': 4096},
        },
    },
    eval_batch_size=5,
)
result = run_task(task_cfg=task_cfg)
```

**CLI** (equivalent):
```bash
evalscope eval \
  --model qwen-vl-plus \
  --api-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --api-key "$DASHSCOPE_API_KEY" \
  --eval-type openai_api \
  --datasets general_vqa \
  --dataset-args '{"general_vqa": {"local_path": "custom_eval/multimodal/vqa", "subset_list": ["example_openai"]}}' \
  --limit 5 \
  --eval-batch-size 5 \
  --judge '{"strategy": "llm", "models": {"model_id": "qwen-plus", "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "api_key": "$DASHSCOPE_API_KEY", "generation_config": {"temperature": 0.0, "max_tokens": 4096}}}'
```

Evaluation will output accuracy metrics:
```text
+--------------+-------------+----------+----------------+-------+---------+---------+
| Model        | Dataset     | Metric   | Subset         |   Num |   Score | Cat.0   |
+==============+=============+==========+================+=======+=========+=========+
| qwen-vl-plus | general_vqa | Accuracy ↑ | example_openai |     5 |    100% | default |
+--------------+-------------+----------+----------------+-------+---------+---------+ 
```

## General-VMCQ Format

### 1. Data Preparation

General-VMCQ adopts a structure similar to MMMU: question text can contain image placeholders `<image x>`, video placeholders `<video x>`, and audio placeholders `<audio x>`; `options` is a Python list string, options can be text or media placeholders. Media files are supplied via media columns (`image_k`, `images`, `video_k`, `audio_k`, etc.) described in the [Media Placeholder Mechanism][mp-feature] section.

**JSONL Example** (`example.jsonl`):
```json
{"question": "Which image shows a dog?", "options": ["<image 1>", "<image 2>", "<image 3>", "<image 4>"], "image_1": "custom_eval/multimodal/images/dog.jpg", "image_2": "custom_eval/multimodal/images/AMNH.jpg", "image_3": "custom_eval/multimodal/images/tesla.jpg", "image_4": "custom_eval/multimodal/images/tokyo.jpg", "answer": "A"}
{"question": "<image 1> What building is this?", "options": ["School", "Hospital", "Park", "Museum"], "image_1": "custom_eval/multimodal/images/AMNH.jpg", "answer": "D"}
{"question": "<video 1> What type of media is provided in this sample?", "options": ["Image", "Audio", "Video", "Text"], "video_1": "custom_eval/multimodal/videos/sample.mp4", "answer": "C"}
```

**TSV Example** (`example.tsv`):
```text
question	options	answer	image_1	image_2	image_3	image_4
Which image shows a dog?	["<image 1>", "<image 2>", "<image 3>", "<image 4>"]	A	custom_eval/multimodal/images/dog.jpg	custom_eval/multimodal/images/AMNH.jpg	custom_eval/multimodal/images/tesla.jpg	custom_eval/multimodal/images/tokyo.jpg
<image 1> What building is this?	["School", "Hospital", "Park", "Museum"]	D	custom_eval/multimodal/images/AMNH.jpg			
```

**Field Descriptions**:
- `question`: Question text, can contain `<image x>`, `<video x>`, or `<audio x>` placeholders
- `options`: List (JSON array), elements can be text (e.g., `"School"`) or media placeholders (e.g., `"<image 1>"`, `"<video 1>"`, `"<audio 1>"`), no need to add prefixes like `A.`, `B.`
- `answer`: Correct answer letter (e.g., `"A"`, `"B"`)
- Media columns (`image_k`, `images`, `video_k`, `videos`, `video_k_format`, `audio_k`, `audios`,  `audio_k_format`): See the [Media Placeholder Mechanism][mp-feature] section for full details.

### 2. Configure Evaluation Task

**Python API**:
```python
from evalscope.run import run_task
from evalscope.config import TaskConfig
from os import environ as env

task_cfg = TaskConfig(
    model='qwen-vl-plus',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=env.get('DASHSCOPE_API_KEY'),
    eval_type='openai_api',
    datasets=['general_vmcq'],
    dataset_args={
        'general_vmcq': {
            'local_path': 'custom_eval/multimodal/mcq',
            'subset_list': ['example'],
        }
    },
    limit=10,
)

result = run_task(task_cfg=task_cfg)
print(result)
```

**CLI**:
```bash
evalscope eval \
    --model qwen-vl-plus \
    --api-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
    --api-key "$DASHSCOPE_API_KEY" \
    --eval-type openai_api \
    --datasets general_vmcq \
    --dataset-args '{"general_vmcq": {"local_path": "custom_eval/multimodal/mcq", "subset_list": ["example"]}}' \
    --limit 10
```

### 3. Evaluation Results

Evaluation will output accuracy metrics:
```text
+--------------+--------------+----------+----------+-------+---------+---------+
| Model        | Dataset      | Metric   | Subset   |   Num |   Score | Cat.0   |
+==============+==============+==========+==========+=======+=========+=========+
| qwen-vl-plus | general_vmcq | Accuracy ↑ | example  |     3 |    100% | default |
+--------------+--------------+----------+----------+-------+---------+---------+ 
```

## Media Placeholder Mechanism
[mp-feature]: #media-placeholder-mechanism

General-VQA and General-VMCQ share media normalization utilities, but differ in when they validate media columns.

### How It Works

Placeholders like `<image 1>`, `<video 1>`, or `<audio 1>` in plain text are replaced with the corresponding media before prompting to an MLLM.

- **General-VQA** resolves only media referenced by plain-text user messages. Missing referenced media is dropped with a warning; if that would leave a user message empty, its original text is retained. Unreferenced media columns are ignored.
- **General-VMCQ** validates every non-empty media column before building its prompt. Unused malformed media columns therefore cause the record to fail.
- By default, indexed media columns are capped at 100: `image_k`/`video_k`/`audio_k` for k ∈ [1, 100], e.g., `<image 101>` and `image_101` will be ignored.
- For each media type, choose either indexed columns or its plural list column. Do not mix the two representations: General-VQA ignores the list when any referenced indexed column is non-empty, while General-VMCQ ignores it when any indexed column is non-empty. List columns are equivalent to `image_1`, `image_2`, ... and are not capped at 100.
- General-VQA converts resolved placeholder content into structured OpenAI-message content. General-VMCQ inserts it into its multiple-choice prompt.

### Trigger Conditions

- **General-VQA**: every message that satisfies both conditions: 1. its `"role"` is `"user"`, and 2. its `"content"` field is a plain string (type `str`).
- **General-VMCQ**: `question` and `options` field, always triggered.

To bypass it, you can provide structured content for General-VQA messages, or remove placeholders in General-VMCQ questions/options. A General-VQA example is provided below:

```json
// <image 1> tag will not be replaced, because it's structured into `{"type": "text"}` dict
{"answer": "Dog",
 "messages": [{"role": "user",
               "content": [{"type": "text",
                            "text": "<image 1> What animal is this?"}]}]}
```

### Media Column Names

| Column           | Description                                                                                              | k range   |
| ---------------- | -------------------------------------------------------------------------------------------------------- | --------- |
| `image_k`        | Image path/URL/base64 for placeholder `<image k>`                                                        | [1, 100]  |
| `video_k`        | Video path/URL/base64 for placeholder `<video k>`                                                        | [1, 100]  |
| `video_k_format` | Optional video format hint (`"mp4"`, `"mpeg"`, `"mov"`, `"avi"`), automatically guessed if not specified | [1, 100]  |
| `audio_k`        | Audio path/URL/base64 for placeholder `<audio k>`                                                        | [1, 100]  |
| `audio_k_format` | Optional audio format hint (`"wav"`, `"mp3"`), automatically guessed if not specified                    | [1, 100]  |
| `images`         | Image list equivalent to consecutive `image_1`, `image_2`, ... Do not mix with indexed image columns. | unbounded |
| `videos`         | Video list equivalent to consecutive `video_1`, `video_2`, ... Do not mix with indexed video columns. | unbounded |
| `audios`         | Audio list equivalent to consecutive `audio_1`, `audio_2`, ... Do not mix with indexed audio columns. | unbounded |

### Supported Media Values

Each media column accepts any of the following:

- **Local path**: `"custom_eval/multimodal/audio/sample.wav"`
- **HTTP/HTTPS URL**: `"https://.../sample.wav"`
- **Base64 Data URL**: `"data:audio/wav;base64,UklGRiQ..."`
- **Undecoded dict** (for parquet-loaded datasets): `{"path": "..."}` or `{"bytes": b"..."}`
- **Hugging Face Dataset features** (for parquet-loaded datasets): [Image][HFImage], [Video][HFVideo], or [Audio][HFAudio] feature objects

[HFImage]: https://huggingface.co/docs/datasets/about_dataset_features#image-feature
[HFVideo]: https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Video
[HFAudio]: https://huggingface.co/docs/datasets/en/about_dataset_features#audio-feature


---

## Based on VLMEvalKit (Deprecated)

````{warning}
The following format is the Legacy version. It is recommended to use the **General Multimodal Format** described above.

Legacy format requires additional VLMEvalKit dependencies:
```bash
pip install evalscope[vlmeval]
```
Reference: [Evaluating with VLMEvalKit Backend](../../user_guides/backend/vlmevalkit_backend.md)
````


### Multiple Choice Format (MCQ)

#### 1. Data Preparation
The evaluation metric is accuracy, and you need to define a tsv file in the following format (separated by `\t`):
```text
index	category	answer	question	A	B	C	D	image_path
1	Animals	A	What animal is this?	Dog	Cat	Tiger	Elephant	/root/LMUData/images/custom_mcq/dog.jpg
2	Buildings	D	What building is this?	School	Hospital	Park	Museum	/root/LMUData/images/custom_mcq/AMNH.jpg
3	Cities	B	Which city's skyline is this?	New York	Tokyo	Shanghai	Paris	/root/LMUData/images/custom_mcq/tokyo.jpg
4	Vehicles	C	What is the brand of this car?	BMW	Audi	Tesla	Mercedes	/root/LMUData/images/custom_mcq/tesla.jpg
5	Activities	A	What is the person in the picture doing?	Running	Swimming	Reading	Singing	/root/LMUData/images/custom_mcq/running.jpg
```
Where:
- `index` is the question number
- `question` is the question
- `answer` is the answer
- `A`, `B`, `C`, `D` are options, must have at least two options
- `answer` is the answer option
- `image_path` is the image path (absolute path recommended); can also be replaced with `image` field, which should be base64 encoded image
- `category` is the category (optional field)

Place this file in the `~/LMUData` path, and you can use the filename for evaluation. For example, if the filename is `custom_mcq.tsv`, use `custom_mcq` for evaluation.

#### 2. Configuration File
The configuration file can be in `python dict`, `yaml`, or `json` format. For example, the following `config.yaml` file:
```yaml
eval_backend: VLMEvalKit
eval_config:
  model: 
    - type: qwen-vl-chat   # Deployed model name
      name: CustomAPIModel # Fixed value
      api_base: http://localhost:8000/v1/chat/completions
      key: EMPTY
      temperature: 0.0
      img_size: -1
  data:
    - custom_mcq # Custom dataset name, placed in `~/LMUData` path
  mode: all
  limit: 10
  reuse: false
  work_dir: outputs
  nproc: 1
```

#### 3. Run Evaluation

Run the following code to start evaluation:
```python
from evalscope.run import run_task

run_task(task_cfg='config.yaml')
```

Evaluation results:
```text
----------  ----
split       none
Overall     1.0
Activities  1.0
Animals     1.0
Buildings   1.0
Cities      1.0
Vehicles    1.0
----------  ----
```

### Custom Question-Answer Format (VQA)

#### 1. Data Preparation

Prepare a tsv file in question-answer format as follows:
```text
index	answer	question	image_path
1	Dog	What animal is this?	/root/LMUData/images/custom_mcq/dog.jpg
2	Museum	What building is this?	/root/LMUData/images/custom_mcq/AMNH.jpg
3	Tokyo	Which city's skyline is this?	/root/LMUData/images/custom_mcq/tokyo.jpg
4	Tesla	What is the brand of this car?	/root/LMUData/images/custom_mcq/tesla.jpg
5	Running	What is the person in the picture doing?	/root/LMUData/images/custom_mcq/running.jpg
```
This file is the same format as the multiple-choice format, where:
- `index` is the question number
- `question` is the question
- `answer` is the answer
- `image_path` is the image path (absolute path recommended); can also be replaced with `image` field, which should be base64 encoded image

Place this file in the `~/LMUData` path, and you can use the filename for evaluation. For example, if the filename is `custom_vqa.tsv`, use `custom_vqa` for evaluation.

#### 2. Custom Evaluation Script

The following is an example of a custom dataset. This example implements a custom evaluation script for question-answer format, which automatically loads the dataset, uses default prompts for Q&A, and finally calculates accuracy as the evaluation metric.


```python
import os
import numpy as np
from vlmeval.dataset.image_base import ImageBaseDataset
from vlmeval.dataset.image_vqa import CustomVQADataset
from vlmeval.smp import load, dump, d2df

class CustomDataset:
    def load_data(self, dataset):
        # Custom dataset loading
        data_path = os.path.join(os.path.expanduser("~/LMUData"), f'{dataset}.tsv')
        return load(data_path)
        
    def build_prompt(self, line):
        msgs = ImageBaseDataset.build_prompt(self, line)
        # Add prompts or custom instructions here
        msgs[-1]['value'] += '\nAnswer the question with one word or phrase.'
        return msgs
    
    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        assert 'answer' in data and 'prediction' in data
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x) for x in data['answer']]
        
        print(data)
        
        # ========Calculate evaluation metrics as needed=========
        # Exact match
        result = np.mean(data['answer'] == data['prediction'])
        ret = {'Overall': result}
        ret = d2df(ret).round(2)
        # Save results
        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(ret, result_file)
        return ret
        # ====================================
        
# Need to keep the following code to override default dataset class
CustomVQADataset.load_data = CustomDataset.load_data
CustomVQADataset.build_prompt = CustomDataset.build_prompt
CustomVQADataset.evaluate = CustomDataset.evaluate
```

#### 3. Configuration File
The configuration file can be in `python dict`, `yaml`, or `json` format. For example, the following `config.yaml` file:
```{code-block} yaml 
:caption: config.yaml

eval_backend: VLMEvalKit
eval_config:
  model: 
    - type: qwen-vl-chat   
      name: CustomAPIModel 
      api_base: http://localhost:8000/v1/chat/completions
      key: EMPTY
      temperature: 0.0
      img_size: -1
  data:
    - custom_vqa # Custom dataset name, placed in `~/LMUData` path
  mode: all
  limit: 10
  reuse: false
  work_dir: outputs
  nproc: 1
```

#### 4. Run Evaluation

The complete evaluation script is as follows:
```{code-block} python
:emphasize-lines: 1

from custom_dataset import CustomDataset  # Import custom dataset
from evalscope.run import run_task

run_task(task_cfg='config.yaml')
```

Evaluation results:
```text
{'qwen-vl-chat_custom_vqa_acc': {'Overall': '1.0'}}
```
