# FLORES-IN

## 概述

FLORES-IN 评估英语与印度及周边地区29种语言之间的翻译能力。

## 任务描述

- **任务类型**：机器翻译
- **输入**：一个英文句子（`en-xx` 子集）或配对语言的句子（`xx-en` 子集）
- **输出**：该句子翻译成另一种语言的结果
- **领域**：英语与来自印度及周边地区的29种语言之间的通用领域翻译

## 主要特点

- 包含 58,696 个测试句子，分布在 58 个子集中，每个子集对应一种语言和翻译方向。
- 默认使用论文中的提示词以及来自同一子集 `dev` 拆分的一个示例。

## 评估说明

- 使用语料库 SacreBLEU chrF（字符顺序为 6，单词顺序为 0，beta 为 2）对第一个非空行进行评分。
- 每个方向的 `macro_score` 对其包含的 29 种语言赋予相等的权重。
- 根据 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) 许可发布。

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `flores_in` |
| **数据集ID** | [google/IndicGenBench_flores_in](https://modelscope.cn/datasets/google/IndicGenBench_flores_in/summary) |
| **论文** | [Paper](https://aclanthology.org/2024.acl-long.595/) |
| **标签** | `MachineTranslation`, `MultiLingual` |
| **指标** | `chrf` |
| **默认示例数** | 1-shot |
| **评估划分** | `test` |
| **训练划分** | `dev` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 58,696 |
| 提示词长度（平均值） | 461.71 字符 |
| 提示词长度（最小/最大） | 234 / 946 字符 |

**各子集统计：**

| 子集 | 样本数 | 提示词平均值 | 提示词最小值 | 提示词最大值 |
|--------|---------|-------------|------------|------------|
| `en-as` | 1,012 | 581.4 | 486 | 819 |
| `en-awa` | 1,012 | 554.4 | 459 | 792 |
| `en-bgc` | 1,012 | 425.4 | 330 | 663 |
| `en-bho` | 1,012 | 566.4 | 471 | 804 |
| `en-bn` | 1,012 | 389.4 | 294 | 627 |
| `en-bo` | 1,012 | 491.4 | 396 | 729 |
| `en-brx` | 1,012 | 436.4 | 341 | 674 |
| `en-gbm` | 1,012 | 441.4 | 346 | 679 |
| `en-gom` | 1,012 | 441.4 | 346 | 679 |
| `en-gu` | 1,012 | 493.4 | 398 | 731 |
| `en-hi` | 1,012 | 401.4 | 306 | 639 |
| `en-hne` | 1,012 | 554.4 | 459 | 792 |
| `en-hoj` | 1,012 | 425.4 | 330 | 663 |
| `en-kn` | 1,012 | 507.4 | 412 | 745 |
| `en-mai` | 1,012 | 418.4 | 323 | 656 |
| `en-ml` | 1,012 | 454.4 | 359 | 692 |
| `en-mni` | 1,012 | 540.4 | 445 | 778 |
| `en-mr` | 1,012 | 398.4 | 303 | 636 |
| `en-mup` | 1,012 | 422.4 | 327 | 660 |
| `en-mwr` | 1,012 | 421.4 | 326 | 659 |
| `en-ne` | 1,012 | 422.4 | 327 | 660 |
| `en-or` | 1,012 | 386.4 | 291 | 624 |
| `en-pa` | 1,012 | 448.4 | 353 | 686 |
| `en-ps` | 1,012 | 496.4 | 401 | 734 |
| `en-sa` | 1,012 | 423.4 | 328 | 661 |
| `en-sat` | 1,012 | 400.4 | 305 | 638 |
| `en-ta` | 1,012 | 558.4 | 463 | 796 |
| `en-te` | 1,012 | 432.4 | 337 | 670 |
| `en-ur` | 1,012 | 459.4 | 364 | 697 |
| `as-en` | 1,012 | 488.44 | 399 | 693 |
| `awa-en` | 1,012 | 635.65 | 543 | 887 |
| `bgc-en` | 1,012 | 417.0 | 331 | 625 |
| `bho-en` | 1,012 | 325.09 | 234 | 565 |
| `bn-en` | 1,012 | 461.51 | 372 | 667 |
| `bo-en` | 1,012 | 661.39 | 557 | 946 |
| `brx-en` | 1,012 | 455.85 | 348 | 708 |
| `gbm-en` | 1,012 | 436.53 | 347 | 642 |
| `gom-en` | 1,012 | 436.71 | 348 | 646 |
| `gu-en` | 1,012 | 483.93 | 397 | 708 |
| `hi-en` | 1,012 | 424.5 | 336 | 676 |
| `hne-en` | 1,012 | 465.43 | 382 | 666 |
| `hoj-en` | 1,012 | 417.48 | 327 | 627 |
| `kn-en` | 1,012 | 450.4 | 357 | 702 |
| `mai-en` | 1,012 | 555.87 | 464 | 778 |
| `ml-en` | 1,012 | 387.57 | 288 | 616 |
| `mni-en` | 1,012 | 424.16 | 327 | 642 |
| `mr-en` | 1,012 | 435.47 | 338 | 659 |
| `mup-en` | 1,012 | 424.53 | 336 | 636 |
| `mwr-en` | 1,012 | 421.22 | 326 | 636 |
| `ne-en` | 1,012 | 431.4 | 340 | 668 |
| `or-en` | 1,012 | 535.24 | 440 | 756 |
| `pa-en` | 1,012 | 490.84 | 396 | 739 |
| `ps-en` | 1,012 | 461.0 | 375 | 665 |
| `sa-en` | 1,012 | 561.74 | 468 | 793 |
| `sat-en` | 1,012 | 465.13 | 370 | 693 |
| `ta-en` | 1,012 | 475.31 | 371 | 727 |
| `te-en` | 1,012 | 348.9 | 257 | 577 |
| `ur-en` | 1,012 | 408.38 | 314 | 628 |

## 样例示例

**子集**: `en-as`

```json
{
  "input": [
    {
      "id": "7c97bce9",
      "content": "Translate the following:\n\nTo Assamese: In the 1480s, when its construction was begun by Lord Hastings, the country was relatively peaceful and defense was only required against small bands of roving marauders.\nOutput: 1480 চনত লর্ড হেষ্টিংছে  ... [TRUNCATED 105 chars] ... াওডাৰৰ সৰু সৰু গোটৰ বিৰুদ্ধেহে সেনেবাহিনীৰ দৰকাৰ হৈছিল।\n\nTranslate the following:\n\nTo Assamese: Subcultures bring together like-minded individuals who feel neglected by societal standards and allow them to develop a sense of identity.\nOutput:"
    }
  ],
  "target": "উপসংস্কৃতিসমূহে ব্যক্তি বিশেষৰ একেলগে থকা মনোভাৱ আনি দিয়ে যিসকলে সমাজৰ পৰা অৱহেলাৰ সন্মুখীন হৈছিল আৰু তেওঁলোকক নিজা পৰিচয়ৰ মনোভাৱ বিকশিত কৰাত সহায় কৰে।",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "source": "Subcultures bring together like-minded individuals who feel neglected by societal standards and allow them to develop a sense of identity.",
    "source_language": "en",
    "target_language": "as",
    "target_language_name": "Assamese"
  }
}
```

*注：为便于显示，部分内容已截断。*

## 提示模板

**提示模板：**
```text
Translate the following:

To {language}: {source}
Output:
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets flores_in \
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
    datasets=['flores_in'],
    dataset_args={
        'flores_in': {
            # subset_list: ['en-as', 'en-awa', 'en-bgc']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
