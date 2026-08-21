# 多模态大模型

本框架支持两种自定义多模态评测方式：

- **通用问答题格式（General-VQA）**：适用于问答类多模态评测任务。支持两种输入风格：**OpenAI 消息数据** 和 **MMMU 风格媒体占位符数据**。
- **通用选择题格式（General-VMCQ）**：适用于选择题类多模态评测任务。使用[媒体占位符机制][mp-feature]在问题和选项中嵌入图片、视频和音频，类似 MMMU 格式。

## 通用问答题格式（General-VQA）

General-VQA 支持**两种输入风格**：

1. **OpenAI 消息数据** — 完整的结构化内容，在 OpenAI 消息模式中显式包含媒体部分（图片、音频、视频）。支持多轮对话、系统提示以及对每个内容部分的精细控制。
2. **MMMU 风格媒体占位符数据** — 一种更简单的方式，用户消息为包含 `<image N>`、`<video N>` 或 `<audio N>` 占位符的纯文本字符串，媒体文件通过单独的索引列提供（参见[媒体占位符机制][mp-feature]）。

两种格式均支持 **JSONL** 或 **TSV** 文件。

### OpenAI 消息数据

在此格式中，每条记录包含一个遵循 OpenAI 聊天补全模式的 `messages` 数组。媒体（图片、音频、视频）作为结构化内容部分直接嵌入到用户消息中。

**JSONL 示例** (`example_openai.jsonl`):
```json
{"messages": [{"role": "user", "content": [{"type": "text", "text": "What animal is this?"}, {"type": "image_url", "image_url": {"url": "custom_eval/multimodal/images/dog.jpg"}}]}], "answer": "Dog"}
{"messages": [{"role": "user", "content": [{"type": "text", "text": "What building is this?"}, {"type": "image_url", "image_url": {"url": "custom_eval/multimodal/images/AMNH.jpg"}}]}], "answer": "Museum"}
```

**TSV 示例** (`example_openai.tsv`):
```text
messages	answer
[{"role": "user", "content": [{"type": "text", "text": "What animal is this?"}, {"type": "image_url", "image_url": {"url": "custom_eval/multimodal/images/dog.jpg"}}]}]	Dog
[{"role": "user", "content": [{"type": "text", "text": "What building is this?"}, {"type": "image_url", "image_url": {"url": "custom_eval/multimodal/images/AMNH.jpg"}}]}]	Museum
```

**字段说明**：
- `messages`: OpenAI 格式的消息数组，支持：
  - 文本内容：`{"type": "text", "text": "问题文本"}`
  - 图片 URL：`{"type": "image_url", "image_url": {"url": "路径或base64"}}`
  - 音频输入：`{"type": "input_audio", "input_audio": {"data": "路径或base64", "format": "wav"}}`
  - 视频 URL：`{"type": "video_url", "video_url": {"url": "路径或base64"}}`
  - 系统消息：`{"role": "system", "content": "系统提示"}`
- `answer`: 参考答案（可选，用于计算 BLEU 和 Rouge 分数）

**支持的图片格式**：
- 本地路径：`"url": "custom_eval/multimodal/images/dog.jpg"`
- HTTP URL：`"url": "https://example.com/image.jpg"`（需模型服务侧支持）
- Base64 编码：`"url": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."`

**支持的音频格式**：
- 本地路径：`"data": "custom_eval/multimodal/audio/sample.wav"`
- Base64 编码：`"data": "data:audio/wav;base64,UklGRiQ..."`
- 音频格式（`format` 字段）：支持 `"wav"` 和 `"mp3"`

**支持的视频格式**：
- 本地路径：`"url": "custom_eval/multimodal/videos/sample.mp4"`
- HTTP URL：`"url": "https://example.com/video.mp4"`（需模型服务侧支持）
- Base64 编码：`"url": "data:video/mp4;base64,AAAAIGZ0eX..."`
- 视频格式会从路径、URL 或 data URI 中推断；支持 `"mp4"`、`"mpeg"`、`"mov"` 和 `"avi"`。

**多图片输入**

支持在一个问题中使用多张图片：

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

**系统提示**

可以添加系统消息来设置评测上下文：

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

**Base64 图片**

支持直接使用 base64 编码的图片：

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

**音频输入**

支持音频内容输入，使用 OpenAI `input_audio` 格式，`data` 字段支持本地路径或 base64 编码，`format` 支持 `wav` 和 `mp3`：

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "请描述这段音频的内容。"},
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
  "answer": "这是一段钢琴演奏的音乐。"
}
```

也可以使用 base64 编码的音频数据：

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "这段音频里说了什么？"},
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
  "answer": "你好，世界。"
}
```

**视频输入**

支持视频内容输入，使用 OpenAI-compatible `video_url` 格式，`url` 字段支持本地路径、HTTP URL 或 base64 编码：

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "请描述这段视频的内容。"},
        {
          "type": "video_url",
          "video_url": {
            "url": "custom_eval/multimodal/videos/sample.mp4"
          }
        }
      ]
    }
  ],
  "answer": "这是一段短视频。"
}
```

### MMMU 格式媒体占位符数据

在此格式中，`messages` 字段将用户消息保留为**纯文本字符串**，其中包含 `<image 1>`、`<video 1>` 或 `<audio 1>` 等占位符，并通过单独的索引列（`image_1`、`video_1`、`audio_1` 等）原地填充。有关占位符解析方式、支持的列名和媒体类型的完整详情，请参见[媒体占位符机制][mp-feature]章节。

**JSONL 示例** (`example_placeholder.jsonl`):
```json
{"messages": [{"role": "user", "content": "What animal is this?<image 1>"}], "image_1": "custom_eval/multimodal/images/dog.jpg", "answer": "Dog"}
{"messages": [{"role": "user", "content": "What building is this?<image 1>"}], "image_1": "custom_eval/multimodal/images/AMNH.jpg", "answer": "Museum"}
{"messages": [{"role": "user", "content": "Which city's skyline is this?<image 1>"}], "image_1": "custom_eval/multimodal/images/tokyo.jpg", "answer": "Tokyo"}
{"messages": [{"role": "user", "content": "What is the brand of this car?<image 1>"}], "image_1": "custom_eval/multimodal/images/tesla.jpg", "answer": "Tesla"}
{"messages": [{"role": "user", "content": "What is the person in the picture doing?<image 1>"}], "image_1": "custom_eval/multimodal/images/running.jpg", "answer": "Running"}
```

**混合媒体示例**：
```json
{"messages": [{"role": "user", 
               "content": "<image 1> Watch <video 1> and describe both."}],
 "answer": "A sunny beach and a wave video.",
 "image_1": "custom_eval/multimodal/images/beach.jpg",
 "video_1": "custom_eval/multimodal/videos/wave.mp4"}
```

**注意**：只有用户消息（`"role": "user"`）且 `content` 为纯文本字符串的消息才会被扫描占位符。已经具有结构化内容（内容部分列表）的消息或其他角色（system、assistant、tool）的消息保持不变。

### 2. 配置评测任务

使用 Python API 或 CLI 进行评测：

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
            'local_path': 'custom_eval/multimodal/vqa',  # 数据集目录
            'subset_list': ['example_openai'],  # 文件名（不含扩展名）
        }
    },
    limit=5,  # 可选：限制评测样本数
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

评测将输出 BLEU 和 Rouge 指标：
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

### 3. 配置裁判模型

可以通过 `judge_model` 参数指定裁判模型，用于生成参考答案进行评测，将获取准确率指标：

```python
from evalscope.run import run_task
from evalscope.constants import EvalType, JudgeStrategy
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
    judge_model_args={
        'model_id': 'qwen-plus', # 无需是多模态模型
        'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'api_key': env.get('DASHSCOPE_API_KEY'),
        'generation_config': {
            'temperature': 0.0,
            'max_tokens': 4096
        },
    },
    eval_batch_size=5,
    judge_strategy=JudgeStrategy.LLM,
)
result = run_task(task_cfg=task_cfg)
```

**CLI**（等效配置）:
```bash
evalscope eval \
  --model qwen-vl-plus \
  --api-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --api-key "$DASHSCOPE_API_KEY" \
  --eval-type openai_api \
  --datasets general_vqa \
  --dataset-args '{"general_vqa": {"local_path": "custom_eval/multimodal/vqa", "subset_list": ["example_openai"]}}' \
  --limit 5 \
  --judge-model-args '{"model_id": "qwen-plus", "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "api_key": "$DASHSCOPE_API_KEY", "generation_config": {"temperature": 0.0, "max_tokens": 4096}}' \
  --judge-worker-num 5 \
  --judge-strategy llm
```

评测将输出准确率指标（输出指标较ROUGE和BLEU更直观）：
```text
+--------------+-------------+----------+----------------+-------+---------+---------+
| Model        | Dataset     | Metric   | Subset         |   Num |   Score | Cat.0   |
+==============+=============+==========+================+=======+=========+=========+
| qwen-vl-plus | general_vqa | Accuracy ↑ | example_openai |     5 |    100% | default |
+--------------+-------------+----------+----------------+-------+---------+---------+ 
```

## 通用选择题格式（General-VMCQ）

### 1. 数据准备

General-VMCQ 采用与 MMMU 相似的结构：问题文本中可包含图片占位符 `<image x>`、视频占位符 `<video x>` 和音频占位符 `<audio x>`；`options` 为 Python 列表字符串，选项可为文本或媒体占位符。媒体文件通过[媒体占位符机制][mp-feature]章节中描述的媒体列（`image_k`、`images`、`video_k`、`audio_k` 等）提供。

**JSONL 示例**（`example.jsonl`）：
```json
{"question": "Which image shows a dog?", "options": ["<image 1>", "<image 2>", "<image 3>", "<image 4>"], "image_1": "custom_eval/multimodal/images/dog.jpg", "image_2": "custom_eval/multimodal/images/AMNH.jpg", "image_3": "custom_eval/multimodal/images/tesla.jpg", "image_4": "custom_eval/multimodal/images/tokyo.jpg", "answer": "A"}
{"question": "<image 1> What building is this?", "options": ["School", "Hospital", "Park", "Museum"], "image_1": "custom_eval/multimodal/images/AMNH.jpg", "answer": "D"}
{"question": "<video 1> What type of media is provided in this sample?", "options": ["Image", "Audio", "Video", "Text"], "video_1": "custom_eval/multimodal/videos/sample.mp4", "answer": "C"}
```

**TSV 示例**（`example.tsv`）：
```text
question	options	answer	image_1	image_2	image_3	image_4
Which image shows a dog?	["<image 1>", "<image 2>", "<image 3>", "<image 4>"]	A	custom_eval/multimodal/images/dog.jpg	custom_eval/multimodal/images/AMNH.jpg	custom_eval/multimodal/images/tesla.jpg	custom_eval/multimodal/images/tokyo.jpg
<image 1> What building is this?	["School", "Hospital", "Park", "Museum"]	D	custom_eval/multimodal/images/AMNH.jpg			
```

**字段说明**：
- `question`: 问题文本，可包含 `<image x>`、`<video x>` 或 `<audio x>` 占位符
- `options`: 列表（JSON 数组），元素可以是文本（如 `"School"`）或媒体占位符（如 `"<image 1>"`、`"<video 1>"`、`"<audio 1>"`），不需要添加 `A.`、`B.` 等前缀
- `answer`: 正确答案字母（如 `"A"`、`"B"`）
- 媒体列（`image_k`、`images`、`video_k`、`videos`、`video_k_format`、`audio_k`、`audios`、`audio_k_format`）：详见[媒体占位符机制][mp-feature]章节。

### 2. 配置评测任务

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

### 3. 评测结果

评测将输出准确率指标：
```text
+--------------+--------------+----------+----------+-------+---------+---------+
| Model        | Dataset      | Metric   | Subset   |   Num |   Score | Cat.0   |
+==============+==============+==========+==========+=======+=========+=========+
| qwen-vl-plus | general_vmcq | Accuracy ↑ | example  |     3 |    100% | default |
+--------------+--------------+----------+----------+-------+---------+---------+ 
```

## 媒体占位符机制
[mp-feature]: #媒体占位符机制

General-VQA 和 General-VMCQ 共享相同的底层机制来解析媒体占位符。本节记录其工作原理。

### 工作原理

占位符如 `<image 1>`、`<video 1>` 或 `<audio 1>` 在纯文本字符串中会被自动替换为记录中索引列的对应媒体。解析在评测时发送给多模态大模型之前完成。

- 占位符通过严格的**一对一映射**解析：`<image 1>` 取 `image_1` 的值，`<image 2>` 取 `image_2` 的值，依此类推。
- 如果占位符引用了无效的媒体值（例如 `None`），它会被忽略并发出警告。如果这会使用户消息为空，则会保留原始文本。未被引用的媒体列会被忽略。
- 默认情况下，索引媒体列上限为 100：`image_k`/`video_k`/`audio_k`，其中 k ∈ [1, 100]，例如 `<image 101>` 和 `image_101` 将被忽略。
- `images`/`videos`/`audios` 列表列仅在对应的 `image_k`/`video_k`/`audio_k` 列不存在时才会被使用。它们等同于按递增顺序编写 `image_1`、`image_2`……，但不受 100 的数量限制。
- 解析后，纯文本内容会被转换为相同的结构化 OpenAI 消息格式，因此无论您选择哪种格式，模型都会收到相同的输入。

### 触发条件

- **General-VQA**：满足以下两个条件的每条消息：1. 其 `"role"` 为 `"user"`，且 2. 其 `"content"` 字段为纯字符串（`str` 类型）。
- **General-VMCQ**：`question` 和 `options` 字段，始终触发。

要绕过此机制，可以为 General-VQA 消息提供结构化内容，或移除 General-VMCQ 问题/选项中的占位符。以下是一个 General-VQA 示例：

```json
// <image 1> 标签不会被替换，因为它被结构化为 `{"type": "text"}` 字典
{"answer": "Dog",
 "messages": [{"role": "user", 
               "content": [{"type": "text", 
                            "text": "<image 1> What animal is this?"}]}]}
```

### 媒体列名

| 列名             | 描述                                                                        | k 范围   |
| ---------------- | --------------------------------------------------------------------------- | -------- |
| `image_k`        | 占位符 `<image k>` 的图片路径/URL/base64                                    | [1, 100] |
| `video_k`        | 占位符 `<video k>` 的视频路径/URL/base64                                    | [1, 100] |
| `video_k_format` | 可选的视频格式提示（`"mp4"`、`"mpeg"`、`"mov"`、`"avi"`），未指定时自动推断 | [1, 100] |
| `audio_k`        | 占位符 `<audio k>` 的音频路径/URL/base64                                    | [1, 100] |
| `audio_k_format` | 可选的音频格式提示（`"wav"`、`"mp3"`），未指定时自动推断                    | [1, 100] |
| `images`         | 图片列表，相当于连续的 `image_1`、`image_2`……仅在 `image_k` 不存在时使用    | 无限制   |
| `videos`         | 视频列表，相当于连续的 `video_1`、`video_2`……仅在 `video_k` 不存在时使用    | 无限制   |
| `audios`         | 音频列表，相当于连续的 `audio_1`、`audio_2`……仅在 `audio_k` 不存在时使用    | 无限制   |

### 支持的媒体值

每个媒体列接受以下任意形式：

- **本地路径**：`"custom_eval/multimodal/audio/sample.wav"`
- **HTTP/HTTPS URL**：`"https://.../sample.wav"`
- **Base64 Data URL**：`"data:audio/wav;base64,UklGRiQ..."`
- **未解码字典**（用于 Parquet 加载的数据集）：`{"path": "..."}` 或 `{"bytes": b"..."}`
- **Hugging Face 数据集特征**（用于 Parquet 加载的数据集）：[Image][HFImage]、[Video][HFVideo] 或 [Audio][HFAudio] 特征对象

[HFImage]: https://huggingface.co/docs/datasets/about_dataset_features#image-feature
[HFVideo]: https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Video
[HFAudio]: https://huggingface.co/docs/datasets/en/about_dataset_features#audio-feature


---

## 基于 VLMEvalKit (已废弃)

````{warning}
以下格式为 Legacy 版本，推荐使用上述的**通用多模态格式**。

Legacy 格式需要额外依赖 VLMEvalKit：
```bash
pip install evalscope[vlmeval]
```
参考：[使用VLMEvalKit评测后端](../../user_guides/backend/vlmevalkit_backend.md)
````


### 选择题格式（MCQ）

#### 1. 数据准备
评测指标为准确率（accuracy），需要定义如下格式的tsv文件（使用`\t`分割）：
```text
index	category	answer	question	A	B	C	D	image_path
1	Animals	A	What animal is this?	Dog	Cat	Tiger	Elephant	/root/LMUData/images/custom_mcq/dog.jpg
2	Buildings	D	What building is this?	School	Hospital	Park	Museum	/root/LMUData/images/custom_mcq/AMNH.jpg
3	Cities	B	Which city's skyline is this?	New York	Tokyo	Shanghai	Paris	/root/LMUData/images/custom_mcq/tokyo.jpg
4	Vehicles	C	What is the brand of this car?	BMW	Audi	Tesla	Mercedes	/root/LMUData/images/custom_mcq/tesla.jpg
5	Activities	A	What is the person in the picture doing?	Running	Swimming	Reading	Singing	/root/LMUData/images/custom_mcq/running.jpg
```
其中：
- `index`为问题序号
- `question`为问题
- `answer`为答案
- `A`、`B`、`C`、`D`为选项，不得少于两个选项
- `answer`为答案选项
- `image_path`为图片路径（建议使用绝对路径）；也可替换为`image`字段，需为base64编码的图片
- `category`为类别（可选字段）

将该文件放在`~/LMUData`路径中，即可使用文件名来进行评测。例如该文件名为`custom_mcq.tsv`，则使用`custom_mcq`即可评测。

#### 2. 配置文件
配置文件，可以为`python dict`、`yaml`或`json`格式，例如如下`config.yaml`文件：
```yaml
eval_backend: VLMEvalKit
eval_config:
  model: 
    - type: qwen-vl-chat   # 部署的模型名称
      name: CustomAPIModel # 固定值
      api_base: http://localhost:8000/v1/chat/completions
      key: EMPTY
      temperature: 0.0
      img_size: -1
  data:
    - custom_mcq # 自定义数据集名称，放在`~/LMUData`路径中
  mode: all
  limit: 10
  reuse: false
  work_dir: outputs
  nproc: 1
```

#### 3. 运行评测

运行下面的代码，即可开始评测：
```python
from evalscope.run import run_task

run_task(task_cfg='config.yaml')
```

评测结果如下：
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

### 自定义问答题格式（VQA）

#### 1. 数据准备

准备一个问答题格式的tsv文件，格式如下：
```text
index	answer	question	image_path
1	Dog	What animal is this?	/root/LMUData/images/custom_mcq/dog.jpg
2	Museum	What building is this?	/root/LMUData/images/custom_mcq/AMNH.jpg
3	Tokyo	Which city's skyline is this?	/root/LMUData/images/custom_mcq/tokyo.jpg
4	Tesla	What is the brand of this car?	/root/LMUData/images/custom_mcq/tesla.jpg
5	Running	What is the person in the picture doing?	/root/LMUData/images/custom_mcq/running.jpg
```
该文件与选择题格式相同，其中：
- `index`为问题序号
- `question`为问题
- `answer`为答案
- `image_path`为图片路径（建议使用绝对路径）；也可替换为`image`字段，需为base64编码的图片

将该文件放在`~/LMUData`路径中，即可使用文件名来进行评测。例如该文件名为`custom_vqa.tsv`，则使用`custom_vqa`即可评测。

#### 2. 自定义评测脚本

以下是一个自定义数据集的示例，该示例实现了一个自定义的问答题格式的评测脚本，该脚本会自动加载数据集，并使用默认的提示进行问答，最后计算准确率作为评测指标。


```python
import os
import numpy as np
from vlmeval.dataset.image_base import ImageBaseDataset
from vlmeval.dataset.image_vqa import CustomVQADataset
from vlmeval.smp import load, dump, d2df

class CustomDataset:
    def load_data(self, dataset):
        # 自定义数据集的加载
        data_path = os.path.join(os.path.expanduser("~/LMUData"), f'{dataset}.tsv')
        return load(data_path)
        
    def build_prompt(self, line):
        msgs = ImageBaseDataset.build_prompt(self, line)
        # 这里添加提示或自定义指令
        msgs[-1]['value'] += '\n用一个单词或短语回答问题。'
        return msgs
    
    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        assert 'answer' in data and 'prediction' in data
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x) for x in data['answer']]
        
        print(data)
        
        # ========根据需要计算评测指标=========
        # 精确匹配
        result = np.mean(data['answer'] == data['prediction'])
        ret = {'Overall': result}
        ret = d2df(ret).round(2)
        # 保存结果
        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(ret, result_file)
        return ret
        # ====================================
        
# 需保留以下代码，重写默认的数据集类
CustomVQADataset.load_data = CustomDataset.load_data
CustomVQADataset.build_prompt = CustomDataset.build_prompt
CustomVQADataset.evaluate = CustomDataset.evaluate
```

#### 3. 配置文件
配置文件，可以为`python dict`、`yaml`或`json`格式，例如如下`config.yaml`文件：
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
    - custom_vqa # 自定义数据集名称，放在`~/LMUData`路径中
  mode: all
  limit: 10
  reuse: false
  work_dir: outputs
  nproc: 1
```

#### 4. 运行评测

完整评测脚本如下：
```{code-block} python
:emphasize-lines: 1

from custom_dataset import CustomDataset  # 导入自定义数据集
from evalscope.run import run_task

run_task(task_cfg='config.yaml')
```

评测结果如下：
```text
{'qwen-vl-chat_custom_vqa_acc': {'Overall': '1.0'}}
```
