# VLM评测集

以下是支持的VLM评测集列表，点击数据集标准名称可跳转详细信息。

| 数据集名称 | 标准名称 | 任务类别 |
|------------|----------|----------|
| `ai2d` | [AI2D](#ai2d) | `Knowledge`, `MultiModal`, `QA` |
| `math_vista` | [MathVista](#mathvista) | `MCQ`, `Math`, `MultiModal`, `Reasoning` |
| `mmmu` | [MMMU](#mmmu) | `Knowledge`, `MultiModal`, `QA` |
| `mmmu_pro` | [MMMU-PRO](#mmmu-pro) | `Knowledge`, `MultiModal`, `QA` |
| `real_world_qa` | [RealWorldQA](#realworldqa) | `Knowledge`, `MultiModal`, `QA` |

---

## 数据集详情

### AI2D

[返回目录](#vlm评测集)
- **数据集名称**: `ai2d`
- **数据集ID**: [lmms-lab/ai2d](https://modelscope.cn/datasets/lmms-lab/ai2d/summary)
- **数据集描述**:  
  > A Diagram Is Worth A Dozen Images
- **任务类别**: `Knowledge`, `MultiModal`, `QA`
- **评估指标**: `acc`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `default`

- **提示模板**: 
```text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}. Think step by step before answering.

{question}

{choices}
```

---

### MathVista

[返回目录](#vlm评测集)
- **数据集名称**: `math_vista`
- **数据集ID**: [evalscope/MathVista](https://modelscope.cn/datasets/evalscope/MathVista/summary)
- **数据集描述**:  
  > MathVista is a consolidated Mathematical reasoning benchmark within Visual contexts. It consists of three newly created datasets, IQTest, FunctionQA, and PaperQA, which address the missing visual domains and are tailored to evaluate logical reasoning on puzzle test figures, algebraic reasoning over functional plots, and scientific reasoning with academic paper figures, respectively. It also incorporates 9 MathQA datasets and 19 VQA datasets from the literature, which significantly enrich the diversity and complexity of visual perception and mathematical reasoning challenges within our benchmark. In total, MathVista includes 6,141 examples collected from 31 different datasets.
- **任务类别**: `MCQ`, `Math`, `MultiModal`, `Reasoning`
- **评估指标**: `acc`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `default`

- **提示模板**: 
```text

Solve the following problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{question}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \boxed command.

```

---

### MMMU

[返回目录](#vlm评测集)
- **数据集名称**: `mmmu`
- **数据集ID**: [AI-ModelScope/MMMU](https://modelscope.cn/datasets/AI-ModelScope/MMMU/summary)
- **数据集描述**:  
  > MMMU (A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI) benchmark designed to evaluate multimodal models on massive multi-discipline tasks demanding college-level subject knowledge and deliberate reasoning. MMMU includes 11.5K meticulously collected multimodal questions from college exams, quizzes, and textbooks, covering six core disciplines: Art & Design, Business, Science, Health & Medicine, Humanities & Social Science, and Tech & Engineering. These questions span 30 subjects and 183 subfields, comprising 30 highly heterogeneous image types, such as charts, diagrams, maps, tables, music sheets, and chemical structures.
- **任务类别**: `Knowledge`, `MultiModal`, `QA`
- **评估指标**: `acc`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `Accounting`, `Agriculture`, `Architecture_and_Engineering`, `Art_Theory`, `Art`, `Basic_Medical_Science`, `Biology`, `Chemistry`, `Clinical_Medicine`, `Computer_Science`, `Design`, `Diagnostics_and_Laboratory_Medicine`, `Economics`, `Electronics`, `Energy_and_Power`, `Finance`, `Geography`, `History`, `Literature`, `Manage`, `Marketing`, `Materials`, `Math`, `Mechanical_Engineering`, `Music`, `Pharmacy`, `Physics`, `Psychology`, `Public_Health`, `Sociology`

- **提示模板**: 
```text

Solve the following problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{question}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \boxed command.

```

---

### MMMU-PRO

[返回目录](#vlm评测集)
- **数据集名称**: `mmmu_pro`
- **数据集ID**: [AI-ModelScope/MMMU_Pro](https://modelscope.cn/datasets/AI-ModelScope/MMMU_Pro/summary)
- **数据集描述**:  
  > MMMU-Pro is an enhanced multimodal benchmark designed to rigorously assess the true understanding capabilities of advanced AI models across multiple modalities. It builds upon the original MMMU benchmark by introducing several key improvements that make it more challenging and realistic, ensuring that models are evaluated on their genuine ability to integrate and comprehend both visual and textual information.
- **任务类别**: `Knowledge`, `MultiModal`, `QA`
- **评估指标**: `acc`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `Accounting`, `Agriculture`, `Architecture_and_Engineering`, `Art_Theory`, `Art`, `Basic_Medical_Science`, `Biology`, `Chemistry`, `Clinical_Medicine`, `Computer_Science`, `Design`, `Diagnostics_and_Laboratory_Medicine`, `Economics`, `Electronics`, `Energy_and_Power`, `Finance`, `Geography`, `History`, `Literature`, `Manage`, `Marketing`, `Materials`, `Math`, `Mechanical_Engineering`, `Music`, `Pharmacy`, `Physics`, `Psychology`, `Public_Health`, `Sociology`

- **额外参数**: 
```json
{
    "dataset_format": "# choose from ['standard (4 options)', 'standard (10 options)', 'vision'], default 'standard (4 options)'"
}
```
- **提示模板**: 
```text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}. Think step by step before answering.

{question}

{choices}
```

---

### RealWorldQA

[返回目录](#vlm评测集)
- **数据集名称**: `real_world_qa`
- **数据集ID**: [lmms-lab/RealWorldQA](https://modelscope.cn/datasets/lmms-lab/RealWorldQA/summary)
- **数据集描述**:  
  > RealWorldQA is a benchmark designed to evaluate the real-world spatial understanding capabilities of multimodal AI models, contributed by XAI. It assesses how well these models comprehend physical environments. The benchmark consists of 700+ images, each accompanied by a question and a verifiable answer. These images are drawn from real-world scenarios, including those captured from vehicles. The goal is to advance AI models' understanding of our physical world.
- **任务类别**: `Knowledge`, `MultiModal`, `QA`
- **评估指标**: `acc`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `default`

- **提示模板**: 
```text
Read the picture and solve the following problem step by step.The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{question}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \boxed command.
```
