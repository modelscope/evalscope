# VLM评测集

以下是支持的VLM评测集列表，点击数据集标准名称可跳转详细信息。

| 数据集名称 | 标准名称 | 任务类别 |
|------------|----------|----------|
| `a_okvqa` | [A-OKVQA](#a-okvqa) | `Knowledge`, `MCQ`, `MultiModal` |
| `ai2d` | [AI2D](#ai2d) | `Knowledge`, `MultiModal`, `QA` |
| `blink` | [BLINK](#blink) | `Knowledge`, `MCQ`, `MultiModal` |
| `cc_bench` | [CCBench](#ccbench) | `Knowledge`, `MCQ`, `MultiModal` |
| `chartqa` | [ChartQA](#chartqa) | `Knowledge`, `MultiModal`, `QA` |
| `cmmmu` | [CMMMU](#cmmmu) | `Chinese`, `Knowledge`, `MultiModal`, `QA` |
| `cmmu` | [CMMU](#cmmu) | `Knowledge`, `MCQ`, `MultiModal`, `QA` |
| `docvqa` | [DocVQA](#docvqa) | `Knowledge`, `MultiModal`, `QA` |
| `fleurs` | [FLEURS](#fleurs) | `Audio`, `MultiLingual`, `SpeechRecognition` |
| `general_vmcq` | [General-VMCQ](#general-vmcq) | `Custom`, `MCQ`, `MultiModal` |
| `general_vqa` | [General-VQA](#general-vqa) | `Custom`, `MultiModal`, `QA` |
| `gsm8k_v` | [GSM8K-V](#gsm8k-v) | `Math`, `MultiModal`, `Reasoning` |
| `hallusion_bench` | [HallusionBench](#hallusionbench) | `Hallucination`, `MultiModal`, `Yes/No` |
| `infovqa` | [InfoVQA](#infovqa) | `Knowledge`, `MultiModal`, `QA` |
| `librispeech` | [LibriSpeech](#librispeech) | `Audio`, `SpeechRecognition` |
| `math_verse` | [MathVerse](#mathverse) | `MCQ`, `Math`, `MultiModal`, `Reasoning` |
| `math_vision` | [MathVision](#mathvision) | `MCQ`, `Math`, `MultiModal`, `Reasoning` |
| `math_vista` | [MathVista](#mathvista) | `MCQ`, `Math`, `MultiModal`, `Reasoning` |
| `micro_vqa` | [MicroVQA](#microvqa) | `Knowledge`, `MCQ`, `Medical`, `MultiModal` |
| `mm_bench` | [MMBench](#mmbench) | `Knowledge`, `MultiModal`, `QA` |
| `mm_star` | [MMStar](#mmstar) | `Knowledge`, `MCQ`, `MultiModal` |
| `mmmu` | [MMMU](#mmmu) | `Knowledge`, `MultiModal`, `QA` |
| `mmmu_pro` | [MMMU-PRO](#mmmu-pro) | `Knowledge`, `MCQ`, `MultiModal` |
| `ocr_bench` | [OCRBench](#ocrbench) | `Knowledge`, `MultiModal`, `QA` |
| `ocr_bench_v2` | [OCRBench-v2](#ocrbench-v2) | `Knowledge`, `MultiModal`, `QA` |
| `olympiad_bench` | [OlympiadBench](#olympiadbench) | `Math`, `Reasoning` |
| `omni_bench` | [OmniBench](#omnibench) | `Knowledge`, `MCQ`, `MultiModal` |
| `omni_doc_bench` | [OmniDocBench](#omnidocbench) | `Knowledge`, `MultiModal`, `QA` |
| `pope` | [POPE](#pope) | `Hallucination`, `MultiModal`, `Yes/No` |
| `real_world_qa` | [RealWorldQA](#realworldqa) | `Knowledge`, `MultiModal`, `QA` |
| `science_qa` | [ScienceQA](#scienceqa) | `Knowledge`, `MCQ`, `MultiModal` |
| `seed_bench_2_plus` | [SEED-Bench-2-Plus](#seed-bench-2-plus) | `Knowledge`, `MCQ`, `MultiModal`, `Reasoning` |
| `simple_vqa` | [SimpleVQA](#simplevqa) | `MultiModal`, `QA`, `Reasoning` |
| `visulogic` | [VisuLogic](#visulogic) | `MCQ`, `Math`, `MultiModal`, `Reasoning` |
| `vstar_bench` | [V*Bench](#vbench) | `Grounding`, `MCQ`, `MultiModal` |
| `zerobench` | [ZeroBench](#zerobench) | `Knowledge`, `MultiModal`, `QA` |

---

## 数据集详情

### A-OKVQA

[返回目录](#vlm评测集)
- **数据集名称**: `a_okvqa`
- **数据集ID**: [HuggingFaceM4/A-OKVQA](https://modelscope.cn/datasets/HuggingFaceM4/A-OKVQA/summary)
- **数据集介绍**:
  > A-OKVQA 是一个用于探究视觉问答中常识推理和外部知识的基准。与仅依赖图像内容的基础 VQA 任务不同，A-OKVQA 要求模型运用广泛的常识和事实性世界知识来回答问题，包含选择题和开放性问题，是对 AI 系统推理能力的一项极具挑战性的测试。
- **任务类别**: `Knowledge`, `MCQ`, `MultiModal`
- **评估指标**: `acc`
- **聚合方法**: `mean`
- **是否需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **评测数据集划分**: `validation`
- **数据集子集**: `default`

- **提示模板**:
<details><summary>View</summary>

````text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

{question}

{choices}
````

</details>

---

### AI2D

[返回目录](#vlm评测集)
- **数据集名称**: `ai2d`
- **数据集ID**: [lmms-lab/ai2d](https://modelscope.cn/datasets/lmms-lab/ai2d/summary)
- **数据集介绍**:
  > AI2D 是一个用于研究 AI 理解图表能力的基准数据集，包含来自科学教科书的 5000 多个多样化图表（如水循环、食物网）。每个图表附带多项选择题，用于测试 AI 对视觉元素、文本标签及其关系的理解。该基准具有挑战性，因为正确回答问题需要同时理解布局、符号和文本。
- **任务类别**: `Knowledge`, `MultiModal`, `QA`
- **评估指标**: `acc`
- **聚合方法**: `mean`
- **是否需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **评测数据集划分**: `test`
- **数据集子集**: `default`

- **提示模板**:
<details><summary>View</summary>

````text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

{question}

{choices}
````

</details>

---

### BLINK

[返回目录](#vlm评测集)
- **数据集名称**: `blink`
- **数据集ID**: [evalscope/BLINK](https://modelscope.cn/datasets/evalscope/BLINK/summary)
- **数据集介绍**:
  > BLINK 是一个用于评估多模态大语言模型（MLLM）核心视觉感知能力的基准，它将14个经典计算机视觉任务转化为3,807道选择题，每道题配有单张或多张图像及视觉提示。
- **任务类别**: `Knowledge`, `MCQ`, `MultiModal`
- **评估指标**: `acc`
- **聚合方法**: `mean`
- **是否需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **评测数据集划分**: `val`
- **数据集子集**: `Art_Style`, `Counting`, `Forensic_Detection`, `Functional_Correspondence`, `IQ_Test`, `Jigsaw`, `Multi-view_Reasoning`, `Object_Localization`, `Relative_Depth`, `Relative_Reflectance`, `Semantic_Correspondence`, `Spatial_Relation`, `Visual_Correspondence`, `Visual_Similarity`

- **提示模板**:
<details><summary>View</summary>

````text
Answer the following multiple choice question. The last line of your response should be of the following format:
'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}.

{question}
````

</details>

---

### CCBench

[返回目录](#vlm评测集)
- **数据集名称**: `cc_bench`
- **数据集ID**: [lmms-lab/MMBench](https://modelscope.cn/datasets/lmms-lab/MMBench/summary)
- **数据集介绍**:
  > CCBench 是 MMBench 的扩展，新增了关于中国传统文化的题目，涵盖书法绘画、文物、饮食服饰、历史人物、风景建筑、素描推理和传统展演。
- **任务类别**: `Knowledge`, `MCQ`, `MultiModal`
- **评估指标**: `acc`
- **聚合方法**: `mean`
- **是否需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **评测数据集划分**: `test`
- **数据集子集**: `cc`

- **提示模板**:
<details><summary>View</summary>

````text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

{question}

{choices}
````

</details>

---

### ChartQA

[返回目录](#vlm评测集)
- **数据集名称**: `chartqa`
- **数据集ID**: [lmms-lab/ChartQA](https://modelscope.cn/datasets/lmms-lab/ChartQA/summary)
- **数据集介绍**:
  > ChartQA 是一个用于评估图表问答能力的基准，涵盖柱状图、折线图、饼图等，侧重于视觉和逻辑推理。
- **任务类别**: `Knowledge`, `MultiModal`, `QA`
- **评估指标**: `relaxed_acc`
- **聚合方法**: `mean`
- **是否需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **评测数据集划分**: `test`
- **数据集子集**: `augmented_test`, `human_test`

- **提示模板**:
<details><summary>View</summary>

````text

{question}

The last line of your response should be of the form "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the a single word answer or number to the problem.

````

</details>

---

### CMMMU

[返回目录](#vlm评测集)
- **数据集名称**: `cmmmu`
- **数据集ID**: [lmms-lab/CMMMU](https://modelscope.cn/datasets/lmms-lab/CMMMU/summary)
- **数据集介绍**:
  > CMMMU 包含从大学考试、测验和教科书中人工收集的多模态问题，涵盖艺术与设计、商业、科学、健康与医学、人文与社会科学、技术与工程六大核心学科，与其姊妹数据集 MMMU 类似。这些问题覆盖30个学科，包含图表、示意图、地图、表格、乐谱和化学结构等39种高度异构的图像类型。
- **任务类别**: `Chinese`, `Knowledge`, `MultiModal`, `QA`
- **评估指标**: `acc`
- **聚合方法**: `mean`
- **是否需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **评测数据集划分**: `val`
- **数据集子集**: `临床医学`, `会计`, `公共卫生`, `农业`, `制药`, `化学`, `历史`, `地理`, `基础医学`, `建筑学`, `心理学`, `数学`, `文献学`, `机械工程`, `材料`, `物理`, `生物`, `电子学`, `社会学`, `管理`, `经济`, `能源和电力`, `艺术`, `艺术理论`, `营销`, `计算机科学`, `设计`, `诊断学与实验室医学`, `金融`, `音乐`


---

### CMMU

[返回目录](#vlm评测集)
- **数据集名称**: `cmmu`
- **数据集ID**: [evalscope/CMMU](https://modelscope.cn/datasets/evalscope/CMMU/summary)
- **数据集介绍**:
  > CMMU是一个新颖的多模态基准，旨在评估数学、生物、物理、化学、地理、政治和历史七个基础学科领域的专业知识。
- **任务类别**: `Knowledge`, `MCQ`, `MultiModal`, `QA`
- **评估指标**: `{'acc': {'numeric': True}}`
- **聚合方法**: `mean`
- **是否需要LLM Judge**: 是
- **默认提示方式**: 0-shot
- **评测数据集划分**: `val`
- **数据集子集**: `biology`, `chemistry`, `geography`, `history`, `math`, `physics`, `politics`

- **提示模板**:
<details><summary>View</summary>

````text
回答下面的单项选择题，请选出其中的正确答案。你的回答的最后一行应该是这样的格式："答案：[LETTER]"（不带引号），其中 [LETTER] 是 {letters} 中的一个。请在回答前进行一步步思考。

问题：{question}
选项：
{choices}

````

</details>

---

### DocVQA

[返回目录](#vlm评测集)
- **数据集名称**: `docvqa`
- **数据集ID**: [lmms-lab/DocVQA](https://modelscope.cn/datasets/lmms-lab/DocVQA/summary)
- **数据集介绍**:
  > DocVQA（文档视觉问答）是一个基准，用于评估AI系统基于文档图像（如扫描页、表格或发票）内容回答问题的能力。与通用视觉问答不同，它不仅需要理解OCR提取的文本，还需理解文档复杂的布局、结构和视觉元素。
- **任务类别**: `Knowledge`, `MultiModal`, `QA`
- **评估指标**: `anls`
- **聚合方法**: `mean`
- **是否需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **评测数据集划分**: `validation`
- **数据集子集**: `DocVQA`

- **提示模板**:
<details><summary>View</summary>

````text
Answer the question according to the image using a single word or phrase.
{question}
The last line of your response should be of the form "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the answer to the question.
````

</details>

---

### FLEURS

[返回目录](#vlm评测集)
- **数据集名称**: `fleurs`
- **数据集ID**: [lmms-lab/fleurs](https://modelscope.cn/datasets/lmms-lab/fleurs/summary)
- **数据集介绍**:
  > FLEURS 是一个包含 102 种语言的多语言基准，用于评估语音识别、口语理解和语音翻译。
- **任务类别**: `Audio`, `MultiLingual`, `SpeechRecognition`
- **评估指标**: `wer`
- **聚合方法**: `mean`
- **是否需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **评测数据集划分**: `test`
- **数据集子集**: `cmn_hans_cn`, `en_us`, `yue_hant_hk`

- **提示模板**:
<details><summary>View</summary>

````text
Please recognize the speech and only output the recognized content:
````

</details>

---

### General-VMCQ

[返回目录](#vlm评测集)
- **数据集名称**: `general_vmcq`
- **数据集ID**: general_vmcq
- **数据集介绍**:
  > 一个用于自定义多模态评估的通用视觉多选问答数据集。格式类似MMMU，非OpenAI消息格式。图像为普通字符串（本地/远程路径或base64数据URL）。有关如何使用此基准的详细说明，请参阅[用户指南](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/custom_dataset/vlm.html)。
- **任务类别**: `Custom`, `MCQ`, `MultiModal`
- **评估指标**: `acc`
- **聚合方法**: `mean`
- **是否需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **评测数据集划分**: `val`
- **数据集子集**: `default`

- **提示模板**:
<details><summary>View</summary>

````text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

{question}

{choices}
````

</details>

---

### General-VQA

[返回目录](#vlm评测集)
- **数据集名称**: `general_vqa`
- **数据集ID**: general_vqa
- **数据集介绍**:
  > 一个用于自定义多模态评估的通用视觉问答数据集。支持兼容 OpenAI 的消息格式（包含图像，支持本地路径或 base64 编码）。详细说明请参见[用户指南](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/custom_dataset/vlm.html)。
- **任务类别**: `Custom`, `MultiModal`, `QA`
- **评估指标**: `BLEU`, `Rouge`
- **聚合方法**: `mean`
- **是否需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **评测数据集划分**: `test`
- **数据集子集**: `default`


---

### GSM8K-V

[返回目录](#vlm评测集)
- **数据集名称**: `gsm8k_v`
- **数据集ID**: [evalscope/GSM8K-V](https://modelscope.cn/datasets/evalscope/GSM8K-V/summary)
- **数据集介绍**:
  > GSM8K-V 是一个纯视觉的多图像数学推理基准，系统地将每个 GSM8K 数学文字题映射为其对应的视觉版本，以实现跨模态的清晰、逐项对比。
- **任务类别**: `Math`, `MultiModal`, `Reasoning`
- **评估指标**: `{'acc': {'numeric': True}}`
- **聚合方法**: `mean`
- **是否需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **评测数据集划分**: `train`
- **数据集子集**: `default`

- **提示模板**:
<details><summary>View</summary>

````text
You are an expert at solving mathematical word problems. Please solve the following problem step by step, showing your reasoning.

When providing your final answer:
- If the answer can be expressed as a whole number (integer), provide it as an integer
Problem: {question}

Please think step by step. After your reasoning, put your final answer within \boxed{{}} with the number only.

````

</details>

---

### HallusionBench

[返回目录](#vlm评测集)
- **数据集名称**: `hallusion_bench`
- **数据集ID**: [lmms-lab/HallusionBench](https://modelscope.cn/datasets/lmms-lab/HallusionBench/summary)
- **数据集介绍**:
  > HallusionBench 是一个先进的诊断基准，旨在评估大型视觉语言模型（LVLM）在图像-上下文推理方面的能力，并分析其语言幻觉和视觉错觉的倾向。
- **任务类别**: `Hallucination`, `MultiModal`, `Yes/No`
- **评估指标**: `aAcc`, `fAcc`, `qAcc`
- **聚合方法**: `f1`
- **是否需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **评测数据集划分**: `image`
- **数据集子集**: `default`

- **提示模板**:
<details><summary>View</summary>

````text
{question}
Please answer YES or NO without an explanation.
````

</details>

---

### InfoVQA

[返回目录](#vlm评测集)
- **数据集名称**: `infovqa`
- **数据集ID**: [lmms-lab/DocVQA](https://modelscope.cn/datasets/lmms-lab/DocVQA/summary)
- **数据集介绍**:
  > InfoVQA（信息视觉问答）是一个基准，用于评估AI模型基于信息密集型图像（如图表、图形、示意图、地图和信息图）回答问题的能力。
- **任务类别**: `Knowledge`, `MultiModal`, `QA`
- **评估指标**: `anls`
- **聚合方法**: `mean`
- **是否需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **评测数据集划分**: `validation`
- **数据集子集**: `InfographicVQA`

- **提示模板**:
<details><summary>View</summary>

````text
Answer the question according to the image using a single word or phrase.
{question}
The last line of your response should be of the form "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the answer to the question.
````

</details>

---

### LibriSpeech

[返回目录](#vlm评测集)
- **数据集名称**: `librispeech`
- **数据集ID**: [lmms-lab/Librispeech-concat](https://modelscope.cn/datasets/lmms-lab/Librispeech-concat/summary)
- **数据集介绍**:
  > LibriSpeech 是一个包含 1000 小时有声读物英语朗读的语料库，广泛用于自动语音识别系统的基准测试。
- **任务类别**: `Audio`, `SpeechRecognition`
- **评估指标**: `wer`
- **聚合方法**: `mean`
- **是否需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **评测数据集划分**: `test_clean`
- **数据集子集**: `default`

- **提示模板**:
<details><summary>View</summary>

````text
Please recognize the speech and only output the recognized content:
````

</details>

---

### MathVerse

[返回目录](#vlm评测集)
- **数据集名称**: `math_verse`
- **数据集ID**: [evalscope/MathVerse](https://modelscope.cn/datasets/evalscope/MathVerse/summary)
- **数据集介绍**:
  > MathVerse 是一个全面的视觉数学基准，旨在公平、深入地评估多模态大语言模型（MLLMs）。它包含 2,612 道来自公开资源的高质量、多学科带图数学题。每道题目由人工标注员转化为六种不同版本，分别提供不同程度的多模态信息，共生成约 1.5 万项测试样本。该方法可全面评估 MLLMs 是否以及在多大程度上真正理解视觉图表以进行数学推理。
- **任务类别**: `MCQ`, `Math`, `MultiModal`, `Reasoning`
- **评估指标**: `{'acc': {'numeric': True}}`
- **聚合方法**: `mean`
- **是否需要LLM Judge**: 是
- **默认提示方式**: 0-shot
- **评测数据集划分**: `testmini`
- **数据集子集**: `Text Dominant`, `Text Lite`, `Vision Dominant`, `Vision Intensive`, `Vision Only`

- **提示模板**:
<details><summary>View</summary>

````text
{question}
Please reason step by step, and put your final answer within \boxed{{}}.
````

</details>

---

### MathVision

[返回目录](#vlm评测集)
- **数据集名称**: `math_vision`
- **数据集ID**: [evalscope/MathVision](https://modelscope.cn/datasets/evalscope/MathVision/summary)
- **数据集介绍**:
  > MATH-Vision（MATH-V）数据集是一个精心整理的包含3,040道高质量数学题的数据集，题目均来自真实数学竞赛，并配有视觉情境。
- **任务类别**: `MCQ`, `Math`, `MultiModal`, `Reasoning`
- **评估指标**: `{'acc': {'numeric': True}}`
- **聚合方法**: `mean`
- **是否需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **评测数据集划分**: `test`
- **数据集子集**: `level 1`, `level 2`, `level 3`, `level 4`, `level 5`

- **提示模板**:
<details><summary>View</summary>

````text
{question}
Please reason step by step, and put your final answer within \boxed{{}} without units.
````

</details>

---

### MathVista

[返回目录](#vlm评测集)
- **数据集名称**: `math_vista`
- **数据集ID**: [evalscope/MathVista](https://modelscope.cn/datasets/evalscope/MathVista/summary)
- **数据集介绍**:
  > MathVista 是一个整合的视觉情境下数学推理基准，包含三个新构建的数据集：IQTest、FunctionQA 和 PaperQA，分别针对谜题图形的逻辑推理、函数图像的代数推理以及学术论文图表的科学推理，填补了现有视觉领域的空白。此外，它还整合了文献中的 9 个 MathQA 数据集和 19 个 VQA 数据集，显著提升了基准在视觉感知与数学推理挑战上的多样性和复杂性。总计，MathVista 汇集了来自 31 个不同数据集的 6,141 个样本。
- **任务类别**: `MCQ`, `Math`, `MultiModal`, `Reasoning`
- **评估指标**: `{'acc': {'numeric': True}}`
- **聚合方法**: `mean`
- **是否需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **评测数据集划分**: `testmini`
- **数据集子集**: `default`

- **提示模板**:
<details><summary>View</summary>

````text
{question}
Please reason step by step, and put your final answer within \boxed{{}} without units.
````

</details>

---

### MicroVQA

[返回目录](#vlm评测集)
- **数据集名称**: `micro_vqa`
- **数据集ID**: [evalscope/MicroVQA](https://modelscope.cn/datasets/evalscope/MicroVQA/summary)
- **数据集介绍**:
  > MicroVQA 是一个由专家策划的、面向显微镜科学研究的多模态推理基准。
- **任务类别**: `Knowledge`, `MCQ`, `Medical`, `MultiModal`
- **评估指标**: `acc`
- **聚合方法**: `mean`
- **是否需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **评测数据集划分**: `test`
- **数据集子集**: `default`

- **提示模板**:
<details><summary>View</summary>

````text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

{question}

{choices}
````

</details>

---

### MMBench

[返回目录](#vlm评测集)
- **数据集名称**: `mm_bench`
- **数据集ID**: [lmms-lab/MMBench](https://modelscope.cn/datasets/lmms-lab/MMBench/summary)
- **数据集介绍**:
  > MMBench 是一个综合评测流程，包含精心整理的多模态数据集和利用 ChatGPT 的新颖循环评测策略。它涵盖 MMBench 定义的 20 个能力维度，并包含翻译成中文的问题版本。
- **任务类别**: `Knowledge`, `MultiModal`, `QA`
- **评估指标**: `acc`
- **聚合方法**: `mean`
- **是否需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **评测数据集划分**: `dev`
- **数据集子集**: `cn`, `en`

- **提示模板**:
<details><summary>View</summary>

````text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

{question}

{choices}
````

</details>

---

### MMStar

[返回目录](#vlm评测集)
- **数据集名称**: `mm_star`
- **数据集ID**: [evalscope/MMStar](https://modelscope.cn/datasets/evalscope/MMStar/summary)
- **数据集介绍**:
  > MMStar：一个精英级不可或缺的视觉多模态基准，旨在确保每个精选样本都具备视觉依赖性、最小数据泄露，并需要先进的多模态能力。
- **任务类别**: `Knowledge`, `MCQ`, `MultiModal`
- **评估指标**: `acc`
- **聚合方法**: `mean`
- **是否需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **评测数据集划分**: `val`
- **数据集子集**: `coarse perception`, `fine-grained perception`, `instance reasoning`, `logical reasoning`, `math`, `science & technology`

- **提示模板**:
<details><summary>View</summary>

````text
Answer the following multiple choice question.
The last line of your response should be of the following format:
'ANSWER: [LETTER]' (without quotes)
where [LETTER] is one of A,B,C,D. Think step by step before answering.

{question}
````

</details>

---

### MMMU

[返回目录](#vlm评测集)
- **数据集名称**: `mmmu`
- **数据集ID**: [AI-ModelScope/MMMU](https://modelscope.cn/datasets/AI-ModelScope/MMMU/summary)
- **数据集介绍**:
  > MMMU（面向专家级通用人工智能的大规模多学科多模态理解与推理基准）旨在评估多模态模型在需要大学水平知识和深度推理的跨学科任务上的表现。该基准包含从大学考试、测验和教材中精心收集的1.15万个多模态问题，涵盖艺术与设计、商业、科学、健康与医学、人文与社会科学、技术与工程六大核心学科，涉及30个学科领域和183个子领域，包含图表、示意图、地图、表格、乐谱、化学结构等30种高度异构的图像类型。
- **任务类别**: `Knowledge`, `MultiModal`, `QA`
- **评估指标**: `acc`
- **聚合方法**: `mean`
- **是否需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **评测数据集划分**: `validation`
- **数据集子集**: `Accounting`, `Agriculture`, `Architecture_and_Engineering`, `Art_Theory`, `Art`, `Basic_Medical_Science`, `Biology`, `Chemistry`, `Clinical_Medicine`, `Computer_Science`, `Design`, `Diagnostics_and_Laboratory_Medicine`, `Economics`, `Electronics`, `Energy_and_Power`, `Finance`, `Geography`, `History`, `Literature`, `Manage`, `Marketing`, `Materials`, `Math`, `Mechanical_Engineering`, `Music`, `Pharmacy`, `Physics`, `Psychology`, `Public_Health`, `Sociology`

- **提示模板**:
<details><summary>View</summary>

````text

Solve the following problem step by step. The last line of your response should be of the form "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the answer to the problem.

{question}

Remember to put your answer on its own line at the end in the form "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the answer to the problem, and you do not need to use a \boxed command.


````

</details>

---

### MMMU-PRO

[返回目录](#vlm评测集)
- **数据集名称**: `mmmu_pro`
- **数据集ID**: [AI-ModelScope/MMMU_Pro](https://modelscope.cn/datasets/AI-ModelScope/MMMU_Pro/summary)
- **数据集介绍**:
  > MMMU-Pro 是一个增强的多模态基准，旨在严格评估先进AI模型在多种模态下的真实理解能力。它在原始 MMMU 基准基础上引入多项关键改进，提升了挑战性和现实性，确保对模型真正融合与理解图文信息的能力进行全面评估。
- **任务类别**: `Knowledge`, `MCQ`, `MultiModal`
- **评估指标**: `acc`
- **聚合方法**: `mean`
- **是否需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **评测数据集划分**: `test`
- **数据集子集**: `Accounting`, `Agriculture`, `Architecture_and_Engineering`, `Art_Theory`, `Art`, `Basic_Medical_Science`, `Biology`, `Chemistry`, `Clinical_Medicine`, `Computer_Science`, `Design`, `Diagnostics_and_Laboratory_Medicine`, `Economics`, `Electronics`, `Energy_and_Power`, `Finance`, `Geography`, `History`, `Literature`, `Manage`, `Marketing`, `Materials`, `Math`, `Mechanical_Engineering`, `Music`, `Pharmacy`, `Physics`, `Psychology`, `Public_Health`, `Sociology`

- **额外参数**: 
```json
{
    "dataset_format": {
        "type": "str",
        "description": "Dataset format variant. Choices: ['standard (4 options)', 'standard (10 options)', 'vision'].",
        "value": "standard (4 options)",
        "choices": [
            "standard (4 options)",
            "standard (10 options)",
            "vision"
        ]
    }
}
```
- **提示模板**:
<details><summary>View</summary>

````text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

{question}

{choices}
````

</details>

---

### OCRBench

[返回目录](#vlm评测集)
- **数据集名称**: `ocr_bench`
- **数据集ID**: [evalscope/OCRBench](https://modelscope.cn/datasets/evalscope/OCRBench/summary)
- **数据集介绍**:
  > OCRBench 是一个全面评估大模型OCR能力的基准，包含文本识别、以场景文本为中心的视觉问答、面向文档的视觉问答、关键信息提取和手写数学表达式识别五个部分。该基准包含1000个问答对，所有答案均经过人工校验与修正，以确保评估更加准确。
- **任务类别**: `Knowledge`, `MultiModal`, `QA`
- **评估指标**: `acc`
- **聚合方法**: `mean`
- **是否需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **评测数据集划分**: `test`
- **数据集子集**: `Artistic Text Recognition`, `Digit String Recognition`, `Doc-oriented VQA`, `Handwriting Recognition`, `Handwritten Mathematical Expression Recognition`, `Irregular Text Recognition`, `Key Information Extraction`, `Non-Semantic Text Recognition`, `Regular Text Recognition`, `Scene Text-centric VQA`

- **提示模板**:
<details><summary>View</summary>

````text
{question}
````

</details>

---

### OCRBench-v2

[返回目录](#vlm评测集)
- **数据集名称**: `ocr_bench_v2`
- **数据集ID**: [evalscope/OCRBench_v2](https://modelscope.cn/datasets/evalscope/OCRBench_v2/summary)
- **数据集介绍**:
  > OCRBench v2 是一个大规模双语文本中心型基准，包含目前最全面的任务集（任务数量是此前多场景基准 OCRBench 的 4 倍）、覆盖最广泛的场景（共 31 种多样场景，如街景、收据、公式、图表等），并提供完善的评估指标，总计包含 10,000 个人工验证的问答对，且难题样本占比较高。
- **任务类别**: `Knowledge`, `MultiModal`, `QA`
- **评估指标**: `acc`
- **聚合方法**: `mean`
- **是否需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **评测数据集划分**: `test`
- **数据集子集**: `APP agent en`, `ASCII art classification en`, `VQA with position en`, `chart parsing en`, `cognition VQA cn`, `cognition VQA en`, `diagram QA en`, `document classification en`, `document parsing cn`, `document parsing en`, `fine-grained text recognition en`, `formula recognition cn`, `formula recognition en`, `full-page OCR cn`, `full-page OCR en`, `handwritten answer extraction cn`, `key information extraction cn`, `key information extraction en`, `key information mapping en`, `math QA en`, `reasoning VQA cn`, `reasoning VQA en`, `science QA en`, `table parsing cn`, `table parsing en`, `text counting en`, `text grounding en`, `text recognition en`, `text spotting en`, `text translation cn`

- **提示模板**:
<details><summary>View</summary>

````text
{question}
````

</details>

---

### OlympiadBench

[返回目录](#vlm评测集)
- **数据集名称**: `olympiad_bench`
- **数据集ID**: [AI-ModelScope/OlympiadBench](https://modelscope.cn/datasets/AI-ModelScope/OlympiadBench/summary)
- **数据集介绍**:
  > OlympiadBench 是一个奥林匹克级别的双语多模态科学评测基准，包含来自数学和物理奥林匹克竞赛及中国高考的 8,476 道题目。子集说明：`OE` 表示 `开放题`，`TP` 表示 `定理证明`，`MM` 表示 `多模态`，`TO` 表示 `纯文本`，`CEE` 表示 `中国高考`，`COMP` 表示 `综合`。**注意：目前 `TP` 子集无法通过自动评测**。
- **任务类别**: `Math`, `Reasoning`
- **评估指标**: `acc`
- **聚合方法**: `mean`
- **是否需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **评测数据集划分**: `train`
- **数据集子集**: `OE_MM_maths_en_COMP`, `OE_MM_maths_zh_CEE`, `OE_MM_maths_zh_COMP`, `OE_MM_physics_en_COMP`, `OE_MM_physics_zh_CEE`, `OE_TO_maths_en_COMP`, `OE_TO_maths_zh_CEE`, `OE_TO_maths_zh_COMP`, `OE_TO_physics_en_COMP`, `OE_TO_physics_zh_CEE`, `TP_MM_maths_en_COMP`, `TP_MM_maths_zh_CEE`, `TP_MM_maths_zh_COMP`, `TP_MM_physics_en_COMP`, `TP_TO_maths_en_COMP`, `TP_TO_maths_zh_CEE`, `TP_TO_maths_zh_COMP`, `TP_TO_physics_en_COMP`

- **提示模板**:
<details><summary>View</summary>

````text
{question}
Please reason step by step, and put your final answer within \boxed{{}}.
````

</details>

---

### OmniBench

[返回目录](#vlm评测集)
- **数据集名称**: `omni_bench`
- **数据集ID**: [m-a-p/OmniBench](https://modelscope.cn/datasets/m-a-p/OmniBench/summary)
- **数据集介绍**:
  > OmniBench 是一个开创性的通用多模态基准，旨在严格评估多模态大模型同时识别、理解和推理视觉、听觉和文本输入的能力。
- **任务类别**: `Knowledge`, `MCQ`, `MultiModal`
- **评估指标**: `acc`
- **聚合方法**: `mean`
- **是否需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **评测数据集划分**: `train`
- **数据集子集**: `default`

- **额外参数**: 
```json
{
    "use_image": {
        "type": "bool",
        "description": "Whether to provide the raw image. False uses textual alternative.",
        "value": true
    },
    "use_audio": {
        "type": "bool",
        "description": "Whether to provide the raw audio. False uses textual alternative.",
        "value": true
    }
}
```
- **提示模板**:
<details><summary>View</summary>

````text
Answer the following multiple choice question based on the image and audio content. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

{question}

{choices}
````

</details>

---

### OmniDocBench

[返回目录](#vlm评测集)
- **数据集名称**: `omni_doc_bench`
- **数据集ID**: [evalscope/OmniDocBench_tsv](https://modelscope.cn/datasets/evalscope/OmniDocBench_tsv/summary)
- **数据集介绍**:
  > OmniDocBench 是一个面向真实场景下多样化文档解析的评估数据集，具有以下特点：  
  > - **多样化的文档类型**：评测集包含 1355 个 PDF 页面，涵盖 9 种文档类型、4 种布局类型和 3 种语言类型，广泛覆盖学术论文、财务报告、报纸、教科书、手写笔记等。  
  > - **丰富的标注信息**：包含 15 类块级元素（如文本段落、标题、表格等，共超过 2 万个）和 4 类跨行级元素（如文本行、行内公式、上下标等，共超过 8 万个）的位置信息，以及各元素区域的识别结果（文本标注、LaTeX 公式标注、同时支持 LaTeX 和 HTML 的表格标注）。OmniDocBench 还提供了文档组件的阅读顺序标注。此外，包含页面和块级别的多种属性标签，分别为 5 种页面属性、3 种文本属性和 6 种表格属性。  
  > **EvalScope 中的评测实现了官方 [OmniDocBench-v1.5 仓库](https://github.com/opendatalab/OmniDocBench) 提供的 `end2end` 和 `quick_match` 方法。**
- **任务类别**: `Knowledge`, `MultiModal`, `QA`
- **评估指标**: `{'display_formula': {'metric': ['Edit_dist']}}`, `{'reading_order': {'metric': ['Edit_dist']}}`, `{'table': {'metric': ['TEDS', 'Edit_dist']}}`, `{'text_block': {'metric': ['Edit_dist', 'BLEU', 'METEOR']}}`
- **聚合方法**: `mean`
- **是否需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **评测数据集划分**: `train`
- **数据集子集**: `default`

- **额外参数**: 
```json
{
    "match_method": {
        "type": "str",
        "description": "Scoring match method used for evaluation.",
        "value": "quick_match",
        "choices": [
            "quick_match",
            "simple_match",
            "no_split"
        ]
    }
}
```
- **提示模板**:
<details><summary>View</summary>

````text
 You are an AI assistant specialized in converting PDF images to Markdown format. Please follow these instructions for the conversion:

    1. Text Processing:
    - Accurately recognize all text content in the PDF image without guessing or inferring.
    - Convert the recognized text into Markdown format.
    - Maintain the original document structure, including headings, paragraphs, lists, etc.

    2. Mathematical Formula Processing:
    - Convert all mathematical formulas to LaTeX format.
    - Enclose inline formulas with \( \). For example: This is an inline formula \( E = mc^2 \)
    - Enclose block formulas with \\[ \\]. For example: \[ \frac{-b \pm \sqrt{b^2 - 4ac}}{2a} \]

    3. Table Processing:
    - Convert tables to HTML format.
    - Wrap the entire table with <table> and </table>.

    4. Figure Handling:
    - Ignore figures content in the PDF image. Do not attempt to describe or convert images.

    5. Output Format:
    - Ensure the output Markdown document has a clear structure with appropriate line breaks between elements.
    - For complex layouts, try to maintain the original document's structure and format as closely as possible.

    Please strictly follow these guidelines to ensure accuracy and consistency in the conversion. Your task is to accurately convert the content of the PDF image into Markdown format without adding any extra explanations or comments.

````

</details>

---

### POPE

[返回目录](#vlm评测集)
- **数据集名称**: `pope`
- **数据集ID**: [lmms-lab/POPE](https://modelscope.cn/datasets/lmms-lab/POPE/summary)
- **数据集介绍**:
  > POPE（基于查询的对象探测评估）是一个用于评估大视觉语言模型（LVLM）中对象幻觉的基准。它通过让模型回答图像中是否存在特定对象的是/否问题来测试模型，旨在衡量模型响应与视觉内容的一致性，重点识别模型错误声称存在实际不存在对象的情况。该基准采用随机、流行和对抗等多种采样策略，构建了鲁棒的评估问题集。
- **任务类别**: `Hallucination`, `MultiModal`, `Yes/No`
- **评估指标**: `accuracy`, `f1_score`, `precision`, `recall`, `yes_ratio`
- **聚合方法**: `f1`
- **是否需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `adversarial`, `popular`, `random`

- **提示模板**:
<details><summary>View</summary>

````text
{question}
Please answer YES or NO without an explanation.
````

</details>

---

### RealWorldQA

[返回目录](#vlm评测集)
- **数据集名称**: `real_world_qa`
- **数据集ID**: [lmms-lab/RealWorldQA](https://modelscope.cn/datasets/lmms-lab/RealWorldQA/summary)
- **数据集介绍**:
  > RealWorldQA 是由 XAI 提供的一个基准，旨在评估多模态 AI 模型对现实世界空间的理解能力，衡量其对物理环境的理解水平。该基准包含 700 多张来自真实场景（包括车载拍摄）的图像，每张图像均附有一个问题和可验证的答案，旨在推动 AI 模型对物理世界的理解。
- **任务类别**: `Knowledge`, `MultiModal`, `QA`
- **评估指标**: `acc`
- **聚合方法**: `mean`
- **是否需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **评测数据集划分**: `test`
- **数据集子集**: `default`

- **提示模板**:
<details><summary>View</summary>

````text
Read the picture and solve the following problem step by step.The last line of your response should be of the form "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the answer to the problem.

{question}

Remember to put your answer on its own line at the end in the form "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the answer to the problem, and you do not need to use a \boxed command.
````

</details>

---

### ScienceQA

[返回目录](#vlm评测集)
- **数据集名称**: `science_qa`
- **数据集ID**: [AI-ModelScope/ScienceQA](https://modelscope.cn/datasets/AI-ModelScope/ScienceQA/summary)
- **数据集介绍**:
  > ScienceQA 是一个包含多项选择题的多模态科学问答基准，题目源自小学和中学课程，涵盖自然科学、社会科学和语言科学等多个领域。该基准的主要特点是大多数问题均配有图像和文本上下文，并附有详细的讲解与解释，支持正确答案，有助于推动能够生成思维链的模型研究。
- **任务类别**: `Knowledge`, `MCQ`, `MultiModal`
- **评估指标**: `acc`
- **聚合方法**: `mean`
- **是否需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **评测数据集划分**: `test`
- **数据集子集**: `default`

- **提示模板**:
<details><summary>View</summary>

````text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

{question}

{choices}
````

</details>

---

### SEED-Bench-2-Plus

[返回目录](#vlm评测集)
- **数据集名称**: `seed_bench_2_plus`
- **数据集ID**: [evalscope/SEED-Bench-2-Plus](https://modelscope.cn/datasets/evalscope/SEED-Bench-2-Plus/summary)
- **数据集介绍**:
  > SEED-Bench-2-Plus 是一个大规模多模态大语言模型（MLLM）评测基准，包含 2.3K 道带精确人工标注的多项选择题，涵盖图表、地图和网页三大类别，覆盖现实世界中丰富的文本场景。
- **任务类别**: `Knowledge`, `MCQ`, `MultiModal`, `Reasoning`
- **评估指标**: `acc`
- **聚合方法**: `mean`
- **是否需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **评测数据集划分**: `test`
- **数据集子集**: `chart`, `map`, `web`

- **提示模板**:
<details><summary>View</summary>

````text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

{question}

{choices}
````

</details>

---

### SimpleVQA

[返回目录](#vlm评测集)
- **数据集名称**: `simple_vqa`
- **数据集ID**: [m-a-p/SimpleVQA](https://modelscope.cn/datasets/m-a-p/SimpleVQA/summary)
- **数据集介绍**:
  > SimpleVQA 是首个全面评估多模态大模型（MLLMs）回答自然语言简答题事实准确性的多模态基准。其具有六大特点：覆盖多种任务与场景，确保问题高质量且具挑战性，参考答案静态且不受时间影响，评估过程简单直接。
- **任务类别**: `MultiModal`, `QA`, `Reasoning`
- **评估指标**: `acc`
- **聚合方法**: `mean`
- **是否需要LLM Judge**: 是
- **默认提示方式**: 0-shot
- **评测数据集划分**: `test`
- **数据集子集**: `default`

- **提示模板**:
<details><summary>View</summary>

````text
Answer the question:

{question}
````

</details>

---

### VisuLogic

[返回目录](#vlm评测集)
- **数据集名称**: `visulogic`
- **数据集ID**: [evalscope/VisuLogic](https://modelscope.cn/datasets/evalscope/VisuLogic/summary)
- **数据集介绍**:
  > VisuLogic 是一个旨在评估多模态大语言模型（MLLM）视觉推理能力的基准，独立于文本推理过程。它包含精心构建的跨类别视觉推理任务，根据所需推理技能分为六类（例如定量推理，涉及理解并推断图像中元素数量的变化）。与现有基准不同，VisuLogic 的视觉推理任务本身难以用语言描述，因而更具挑战性，能够更严格地评估 MLLM 的视觉推理能力。
- **任务类别**: `MCQ`, `Math`, `MultiModal`, `Reasoning`
- **评估指标**: `acc`
- **聚合方法**: `mean`
- **是否需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **评测数据集划分**: `test`
- **数据集子集**: `Attribute Reasoning`, `Other`, `Positional Reasoning`, `Quantitative Reasoning`, `Spatial Reasoning`, `Stylistic Reasoning`

- **提示模板**:
<details><summary>View</summary>

````text

Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A, B, C, D. Think step by step before answering.

{question}

````

</details>

---

### V*Bench

[返回目录](#vlm评测集)
- **数据集名称**: `vstar_bench`
- **数据集ID**: [lmms-lab/vstar-bench](https://modelscope.cn/datasets/lmms-lab/vstar-bench/summary)
- **数据集介绍**:
  > V*Bench 是一个用于评估多模态推理系统中视觉搜索能力的基准，专注于在高分辨率图像中主动定位和识别特定视觉信息的能力，这对于需要细粒度视觉理解的任务至关重要。该基准有助于评估模型在自然语言指令引导下，在复杂视觉场景中执行目标视觉查询并进行推理的表现。
- **任务类别**: `Grounding`, `MCQ`, `MultiModal`
- **评估指标**: `acc`
- **聚合方法**: `mean`
- **是否需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **评测数据集划分**: `test`
- **数据集子集**: `default`

- **提示模板**:
<details><summary>View</summary>

````text

Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A, B, C, D. Think step by step before answering.

{question}

````

</details>

---

### ZeroBench

[返回目录](#vlm评测集)
- **数据集名称**: `zerobench`
- **数据集ID**: [evalscope/zerobench](https://modelscope.cn/datasets/evalscope/zerobench/summary)
- **数据集介绍**:
  > ZeroBench 是一个具有挑战性的大型多模态模型（LMM）视觉推理基准，包含 100 道高质量、人工整理的核心问题，涵盖多个领域、推理类型和图像种类。ZeroBench 的问题设计和校准均超出当前主流模型的能力范围，因此所有评测模型均未能取得非零的 pass@1 分数（使用贪婪解码）或 5/5 的可靠性得分。
- **任务类别**: `Knowledge`, `MultiModal`, `QA`
- **评估指标**: `acc`
- **聚合方法**: `mean`
- **是否需要LLM Judge**: 是
- **默认提示方式**: 0-shot
- **评测数据集划分**: `zerobench`
- **数据集子集**: `default`

- **提示模板**:
<details><summary>View</summary>

````text
{question}



Let's think step by step and give the final answer in curly braces,
like this: {{final answer}}"

````

</details>
