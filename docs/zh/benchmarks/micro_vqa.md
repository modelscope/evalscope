# MicroVQA

## 概述

MicroVQA 是一个由专家精心策划的基准测试，用于评估人工智能模型在基于显微镜的科学研究中的多模态推理能力。该基准测试评估模型对各类科学领域中显微镜图像的理解与推理能力。

## 任务描述

- **任务类型**：科学显微镜视觉问答（Visual Question Answering）
- **输入**：显微镜图像 + 带选项的科学问题
- **输出**：正确答案选项
- **领域**：显微镜技术、科学研究、医学影像

## 主要特点

- 由专家精心设计的显微镜相关问题
- 包含多种类型的显微镜图像
- 测试科学视觉推理能力
- 聚焦医学与生物学成像
- 采用多项选择题格式，并结合思维链（Chain-of-Thought, CoT）提示

## 评估说明

- 默认配置使用 **0-shot** 评估方式
- 在测试集（test split）上进行评估
- 使用简单准确率（accuracy）作为评估指标
- 采用思维链（CoT）提示方法

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `micro_vqa` |
| **数据集 ID** | [evalscope/MicroVQA](https://modelscope.cn/datasets/evalscope/MicroVQA/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `MCQ`, `Medical`, `MultiModal` |
| **指标** | `acc` |
| **默认示例数（Shots）** | 0-shot |
| **评估划分** | `test` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,042 |
| 提示词长度（平均） | 884.16 字符 |
| 提示词长度（最小/最大） | 414 / 1704 字符 |

**图像统计信息：**

| 指标 | 值 |
|--------|-------|
| 总图像数 | 1,977 |
| 每样本图像数 | 最小: 1，最大: 8，平均: 1.9 |
| 分辨率范围 | 78x70 - 2782x3533 |
| 图像格式 | png |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "9bca1e27",
      "content": [
        {
          "text": "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D,E. Think step by step before answering.\n\nA cryo-electron tom ... [TRUNCATED] ... Facilitation of mitochondrial biogenesis to support increased cell growth\nD) Influence on mitochondrial calcium buffering capacity to maintain cellular homeostasis\nE) Modification of mitochondrial lipid composition affecting membrane fluidity"
        },
        {
          "image": "[BASE64_IMAGE: png, ~188.3KB]"
        }
      ]
    }
  ],
  "choices": [
    "Regulation of mitochondrial metabolism to meet high ATP demand",
    "Enhancement of oxidative phosphorylation efficiency under varying energy demands",
    "Facilitation of mitochondrial biogenesis to support increased cell growth",
    "Influence on mitochondrial calcium buffering capacity to maintain cellular homeostasis",
    "Modification of mitochondrial lipid composition affecting membrane fluidity"
  ],
  "target": "A",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "key_question": 644,
    "key_image": 125,
    "correct_answer": "Regulation of mitochondrial metabolism to meet high ATP demand",
    "question_0": "The mitochondria in the tomographic slice are shown in dark blue, with calcium granules highlighted in yellow. How might the presence of calcium granules within the mitochondria affect their function?",
    "answer_0": "The presence of calcium granules within mitochondria can regulate mitochondrial metabolism and energy production, potentially indicating a high demand for ATP in the neuron.",
    "comments_0": null,
    "incorrect_answer_0": "The calcium granules within the mitochondria lead to their structural destabilization, causing the mitochondria to fragment.",
    "question_1": "A cryo-electron tomography (cryo-ET) image displays primary *Drosophila melanogaster* neurons cultured on a micropatterned grid that directs cytoskeletal growth. In the tomographic slice, mitochondria are shown in dark blue, and calcium granules are highlighted in yellow. How might the presence of calcium granules within the mitochondria affect their function?",
    "choices_1": [
      "Inhibition of the electron transport chain, reducing ATP production",
      "Regulation of mitochondrial metabolism to meet high ATP demand",
      "Promotion of mitochondrial fission, leading to fragmented mitochondria",
      "Induction of apoptosis through the release of cytochrome c"
    ],
    "correct_index_1": 1,
    "question_2": "A cryo-electron tomography (cryo-ET) image showcases primary eukaryotic neurons cultured on a specialized substrate that guides cytoskeletal organization. In the tomographic slice, mitochondria appear in deep blue, and calcium granules are depicted in bright yellow. What potential impact could the accumulation of calcium granules within the mitochondria have on their function?",
    "choices_2": [
      "Facilitation of mitochondrial biogenesis to support increased cell growth",
      "Enhancement of oxidative phosphorylation efficiency under varying energy demands",
      "Regulation of mitochondrial metabolism to meet high ATP demand",
      "Modification of mitochondrial lipid composition affecting membrane fluidity",
      "Influence on mitochondrial calcium buffering capacity to maintain cellular homeostasis"
    ],
    "correct_index_2": 2,
    "question_3": "A cryo-electron tomography (cryo-ET) image showcases primary eukaryotic neurons cultured on a specialized substrate that guides cytoskeletal organization. In the tomographic slice, mitochondria appear in deep blue, and calcium granules are depicted in bright yellow. What potential impact could the accumulation of calcium granules within the mitochondria have on their function?",
    "choices_3": [
      "Facilitation of mitochondrial biogenesis to support increased cell growth",
      "Enhancement of oxidative phosphorylation efficiency under varying energy demands",
      "Regulation of mitochondrial metabolism to meet high ATP demand",
      "Modification of mitochondrial lipid composition affecting membrane fluidity",
      "Influence on mitochondrial calcium buffering capacity to maintain cellular homeostasis"
    ],
    "correct_index_3": 2,
    "task": 2,
    "task_str": "hypothesis_gen",
    "context_image_generation": "This image was generated using cryo-ET, where a 3 × 4 montage tomogram was reconstructed from primary Drosophila melanogaster neurons cultured on a micropatterned grid designed to guide cytoskeleton growth. The grid provides a straight-line p ... [TRUNCATED] ... he organization of cellular structures such as microtubules. The tomographic slice was then segmented, with specific cellular components like microtubules, mitochondria, and ribosomes identified and highlighted using color coding for clarity.",
    "context_motivation": "The motivation behind generating these images was to study the organization of the cytoskeleton and associated cellular structures in primary Drosophila melanogaster neurons. By culturing the neurons on a straight-line micropatterned grid, th ... [TRUNCATED] ... ly microtubules. This approach helps in understanding how physical cues from the environment can influence the spatial arrangement and interactions of cellular organelles, which is crucial for insights into cellular architecture and function.",
    "images_source": "https://www.nature.com/articles/s41592-023-02000-z",
    "image_caption": "The image shows a tomographic slice with a cross-section of a Drosophila melanogaster neuron with various cellular structures visible. The image is a fully stitched montage, providing a comprehensive view of the neuron's internal organization ... [TRUNCATED] ... distribution and alignment within the neuron, Mitochondria are depicted in dark blue, with yellow indicating the presence of calcium granules within them, Ribosomes are shown in light pink, with associated vesicles represented in darker cyan.",
    "key_person": 0
  }
}
```

*注：部分内容因展示需要已被截断。*

## 提示模板

**提示模板：**
```text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

{question}

{choices}
```

## 使用方法

### 使用命令行（CLI）

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets micro_vqa \
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
    datasets=['micro_vqa'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```