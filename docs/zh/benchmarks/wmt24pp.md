# WMT2024++


## 概述

WMT2024++ 是一个基于 WMT 2024 新闻翻译任务的综合性机器翻译基准测试。它支持以英语为源语言的 54 个语言对，可用于评估模型在多种目标语言上的翻译质量。

## 任务描述

- **任务类型**：机器翻译
- **输入**：带有翻译提示的英文源文本
- **输出**：目标语言的翻译文本
- **语言对**：54 个（英语到 54 种目标语言）

## 主要特性

- 广泛的多语言覆盖（54 种目标语言）
- 新闻领域文本，贴近实际应用场景
- 多种评估指标（BLEU、BERTScore、COMET）
- 标准化的提示模板，确保评估一致性
- 支持批量评分以提升效率

## 评估说明

- 默认配置使用 **0-shot** 评估
- 评估指标：**BLEU**、**BERTScore**（XLM-RoBERTa）、**COMET**（wmt22-comet-da）
- 在 **test** 划分上进行评估
- 应用语言特定的归一化处理
- COMET 指标需要安装 `unbabel-comet` 包
- 子集代表单个语言对（例如 `en-zh_cn`、`en-de_de`）

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `wmt24pp` |
| **数据集ID** | [extraordinarylab/wmt24pp](https://modelscope.cn/datasets/extraordinarylab/wmt24pp/summary) |
| **论文** | N/A |
| **标签** | `MachineTranslation`, `MultiLingual` |
| **指标** | `bleu`, `bert_score`, `comet` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 52,800 |
| 提示词长度（平均） | 265.45 字符 |
| 提示词长度（最小/最大） | 71 / 1047 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示平均长度 | 提示最小长度 | 提示最大长度 |
|--------|---------|-------------|------------|------------|
| `en-ar_eg` | 960 | 263.26 | 75 | 1039 |
| `en-ar_sa` | 960 | 263.26 | 75 | 1039 |
| `en-bg_bg` | 960 | 269.26 | 81 | 1045 |
| `en-bn_in` | 960 | 265.26 | 77 | 1041 |
| `en-ca_es` | 960 | 265.26 | 77 | 1041 |
| `en-cs_cz` | 960 | 261.26 | 73 | 1037 |
| `en-da_dk` | 960 | 263.26 | 75 | 1039 |
| `en-de_de` | 960 | 263.26 | 75 | 1039 |
| `en-el_gr` | 960 | 261.26 | 73 | 1037 |
| `en-es_mx` | 960 | 265.26 | 77 | 1041 |
| `en-et_ee` | 960 | 267.26 | 79 | 1043 |
| `en-fa_ir` | 960 | 261.26 | 73 | 1037 |
| `en-fi_fi` | 960 | 265.26 | 77 | 1041 |
| `en-fil_ph` | 960 | 267.26 | 79 | 1043 |
| `en-fr_ca` | 960 | 263.26 | 75 | 1039 |
| `en-fr_fr` | 960 | 263.26 | 75 | 1039 |
| `en-gu_in` | 960 | 267.26 | 79 | 1043 |
| `en-he_il` | 960 | 263.26 | 75 | 1039 |
| `en-hi_in` | 960 | 261.26 | 73 | 1037 |
| `en-hr_hr` | 960 | 267.26 | 79 | 1043 |
| `en-hu_hu` | 960 | 269.26 | 81 | 1045 |
| `en-id_id` | 960 | 271.26 | 83 | 1047 |
| `en-is_is` | 960 | 269.26 | 81 | 1045 |
| `en-it_it` | 960 | 265.26 | 77 | 1041 |
| `en-ja_jp` | 960 | 267.26 | 79 | 1043 |
| `en-kn_in` | 960 | 265.26 | 77 | 1041 |
| `en-ko_kr` | 960 | 263.26 | 75 | 1039 |
| `en-lt_lt` | 960 | 271.26 | 83 | 1047 |
| `en-lv_lv` | 960 | 265.26 | 77 | 1041 |
| `en-ml_in` | 960 | 269.26 | 81 | 1045 |
| `en-mr_in` | 960 | 265.26 | 77 | 1041 |
| `en-nl_nl` | 960 | 261.26 | 73 | 1037 |
| `en-no_no` | 960 | 269.26 | 81 | 1045 |
| `en-pa_in` | 960 | 265.26 | 77 | 1041 |
| `en-pl_pl` | 960 | 263.26 | 75 | 1039 |
| `en-pt_br` | 960 | 271.26 | 83 | 1047 |
| `en-pt_pt` | 960 | 271.26 | 83 | 1047 |
| `en-ro_ro` | 960 | 267.26 | 79 | 1043 |
| `en-ru_ru` | 960 | 265.26 | 77 | 1041 |
| `en-sk_sk` | 960 | 263.26 | 75 | 1039 |
| `en-sl_si` | 960 | 269.26 | 81 | 1045 |
| `en-sr_rs` | 960 | 265.26 | 77 | 1041 |
| `en-sv_se` | 960 | 265.26 | 77 | 1041 |
| `en-sw_ke` | 960 | 265.26 | 77 | 1041 |
| `en-sw_tz` | 960 | 265.26 | 77 | 1041 |
| `en-ta_in` | 960 | 261.26 | 73 | 1037 |
| `en-te_in` | 960 | 263.26 | 75 | 1039 |
| `en-th_th` | 960 | 259.26 | 71 | 1035 |
| `en-tr_tr` | 960 | 265.26 | 77 | 1041 |
| `en-uk_ua` | 960 | 269.26 | 81 | 1045 |
| `en-ur_pk` | 960 | 259.26 | 71 | 1035 |
| `en-vi_vn` | 960 | 271.26 | 83 | 1047 |
| `en-zh_cn` | 960 | 267.26 | 79 | 1043 |
| `en-zh_tw` | 960 | 267.26 | 79 | 1043 |
| `en-zu_za` | 960 | 259.26 | 71 | 1035 |

## 样例示例

**子集**: `en-ar_eg`

```json
{
  "input": [
    {
      "id": "557f3aa1",
      "content": [
        {
          "text": "Translate the following english sentence into arabic:\n\nenglish: Siso's depictions of land, water center new gallery exhibition\narabic:"
        }
      ]
    }
  ],
  "target": "رسومات سيسو عن الأرض والمية في معرضه الجديد",
  "id": 0,
  "group_id": 0,
  "subset_key": "en-ar_eg",
  "metadata": {
    "source_text": "Siso's depictions of land, water center new gallery exhibition",
    "target_text": "رسومات سيسو عن الأرض والمية في معرضه الجديد",
    "source_language": "en",
    "target_language": "ar_eg"
  }
}
```

## 提示模板

**提示模板：**
```text
Translate the following {source_language} sentence into {target_language}:

{source_language}: {source_text}
{target_language}:
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets wmt24pp \
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
    datasets=['wmt24pp'],
    dataset_args={
        'wmt24pp': {
            # subset_list: ['en-ar_eg', 'en-ar_sa', 'en-bg_bg']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```