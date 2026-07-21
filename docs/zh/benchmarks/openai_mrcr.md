# OpenAI MRCR


## 概述

MRCR（Memory-Recall with Contextual Retrieval，上下文检索记忆召回）是 OpenAI 用于评估模型在长上下文场景中检索与回忆能力的基准测试。该基准测试检验模型能否从长提示中准确提取并使用嵌入的特定信息（称为“针”/needles）。

## 任务描述

- **任务类型**：长上下文记忆与检索
- **输入**：包含 2、4 或 8 个嵌入“针”的长对话
- **输出**：精确召回“针”的内容
- **领域**：长上下文理解、信息检索

## 核心特性

- 测试不同数量“针”（2、4 或 8 个）下的检索能力
- 在多个上下文长度区间（4K 至 1M tokens）进行评估
- 衡量嵌入信息的精确召回率
- 使用序列比率（sequence ratio）评分实现精准匹配
- 支持可配置的上下文长度过滤

## 评估说明

- 默认使用 **train** 数据划分进行评估
- 主要指标：**MRCR Score**（序列比率）
- 结果按 token 数量分桶：
  - 4K-8K、8K-16K、16K-32K、32K-64K
  - 64K-128K、128K-256K、256K-512K、512K-1M
- 可通过 `extra_params` 配置：`max_context_size`、`min_context_size`、`needle_count`、`tokenizer`
- 使用 `tiktoken` 进行精确的 token 计数

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `openai_mrcr` |
| **数据集ID** | [openai-mirror/mrcr](https://modelscope.cn/datasets/openai-mirror/mrcr/summary) |
| **论文** | N/A |
| **标签** | `LongContext`, `Retrieval` |
| **指标** | `mrcr_score` |
| **默认示例数** | 0-shot |
| **评估划分** | `train` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 2,400 |
| 提示词长度（平均） | 944363.11 字符 |
| 提示词长度（最小/最大） | 15459 / 5244934 字符 |

## 样例示例

**子集**: `default`

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

## 提示模板

*未定义提示模板。*

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `max_context_size` | `int | null` | `None` | 最大上下文 token 数；超过此值的样本将被跳过。默认为 None（无限制）。 |
| `min_context_size` | `int | null` | `None` | 最小上下文 token 数；低于此值的样本将被跳过。默认为 None（无限制）。 |
| `needle_count` | `list[int] | null` | `None` | “针”数量过滤器（允许值：2、4、8）。必须为列表，例如 `[2]`、`[4]` 或 `[2, 4, 8]`。设为 None 则保留所有样本。 |
| `tik_enc` | `str` | `o200k_base` | 用于 token 计数的 tiktoken 编码名称。 |
| `prefix_filter` | `str | null` | `None` | 用于过滤答案的正则表达式模式。默认为 None（无过滤）。 |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets openai_mrcr \
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
    datasets=['openai_mrcr'],
    dataset_args={
        'openai_mrcr': {
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
