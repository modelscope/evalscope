# VLM Benchmarks

Below is the list of supported VLM benchmarks. Click on a benchmark name to jump to details.

| Benchmark Name | Pretty Name | Task Categories |
|------------|----------|----------|
| `ai2d` | [AI2D](#ai2d) | `Knowledge`, `MultiModal`, `QA` |
| `blink` | [BLINK](#blink) | `Knowledge`, `MCQ`, `MultiModal` |
| `cc_bench` | [CCBench](#ccbench) | `Knowledge`, `MCQ`, `MultiModal` |
| `chartqa` | [ChartQA](#chartqa) | `Knowledge`, `MultiModal`, `QA` |
| `docvqa` | [DocVQA](#docvqa) | `Knowledge`, `MultiModal`, `QA` |
| `infovqa` | [InfoVQA](#infovqa) | `Knowledge`, `MultiModal`, `QA` |
| `math_vista` | [MathVista](#mathvista) | `MCQ`, `Math`, `MultiModal`, `Reasoning` |
| `mm_bench` | [MMBench](#mmbench) | `Knowledge`, `MultiModal`, `QA` |
| `mm_star` | [MMStar](#mmstar) | `Knowledge`, `MCQ`, `MultiModal` |
| `mmmu` | [MMMU](#mmmu) | `Knowledge`, `MultiModal`, `QA` |
| `mmmu_pro` | [MMMU-PRO](#mmmu-pro) | `Knowledge`, `MCQ`, `MultiModal` |
| `ocr_bench` | [OCRBench](#ocrbench) | `Knowledge`, `MultiModal`, `QA` |
| `ocr_bench_v2` | [OCRBench-v2](#ocrbench-v2) | `Knowledge`, `MultiModal`, `QA` |
| `olympiad_bench` | [OlympiadBench](#olympiadbench) | `Math`, `Reasoning` |
| `omni_bench` | [OmniBench](#omnibench) | `Knowledge`, `MCQ`, `MultiModal` |
| `real_world_qa` | [RealWorldQA](#realworldqa) | `Knowledge`, `MultiModal`, `QA` |

---

## Benchmark Details

### AI2D

[Back to Top](#vlm-benchmarks)
- **Dataset Name**: `ai2d`
- **Dataset ID**: [lmms-lab/ai2d](https://modelscope.cn/datasets/lmms-lab/ai2d/summary)
- **Description**:
  > AI2D is a benchmark dataset for researching the understanding of diagrams by AI. It contains over 5,000 diverse diagrams from science textbooks (e.g., the water cycle, food webs). Each diagram is accompanied by multiple-choice questions that test an AI's ability to interpret visual elements, text labels, and their relationships. The benchmark is challenging because it requires jointly understanding the layout, symbols, and text to answer questions correctly.
- **Task Categories**: `Knowledge`, `MultiModal`, `QA`
- **Evaluation Metrics**: `acc`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Subsets**: `default`

- **Prompt Template**: 
```text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}. Think step by step before answering.

{question}

{choices}
```

---

### BLINK

[Back to Top](#vlm-benchmarks)
- **Dataset Name**: `blink`
- **Dataset ID**: [evalscope/BLINK](https://modelscope.cn/datasets/evalscope/BLINK/summary)
- **Description**:
  > BLINK is a benchmark designed to evaluate the core visual perception abilities of multimodal large language models (MLLMs). It transforms 14 classic computer vision tasks into 3,807 multiple-choice questions, accompanied by single or multiple images and visual prompts.
- **Task Categories**: `Knowledge`, `MCQ`, `MultiModal`
- **Evaluation Metrics**: `acc`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Subsets**: `Art_Style`, `Counting`, `Forensic_Detection`, `Functional_Correspondence`, `IQ_Test`, `Jigsaw`, `Multi-view_Reasoning`, `Object_Localization`, `Relative_Depth`, `Relative_Reflectance`, `Semantic_Correspondence`, `Spatial_Relation`, `Visual_Correspondence`, `Visual_Similarity`

- **Prompt Template**: 
```text
Answer the following multiple choice question. The last line of your response should be of the following format:
'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}.

{question}
```

---

### CCBench

[Back to Top](#vlm-benchmarks)
- **Dataset Name**: `cc_bench`
- **Dataset ID**: [lmms-lab/MMBench](https://modelscope.cn/datasets/lmms-lab/MMBench/summary)
- **Description**:
  > CCBench is an extension of MMBench with newly design questions about Chinese traditional culture, including Calligraphy Painting, Cultural Relic, Food & Clothes, Historical Figures, Scenery & Building, Sketch Reasoning and Traditional Show.
- **Task Categories**: `Knowledge`, `MCQ`, `MultiModal`
- **Evaluation Metrics**: `acc`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Subsets**: `cc`

- **Prompt Template**: 
```text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}. Think step by step before answering.

{question}

{choices}
```

---

### ChartQA

[Back to Top](#vlm-benchmarks)
- **Dataset Name**: `chartqa`
- **Dataset ID**: [lmms-lab/ChartQA](https://modelscope.cn/datasets/lmms-lab/ChartQA/summary)
- **Description**:
  > ChartQA is a benchmark designed to evaluate question-answering capabilities about charts (e.g., bar charts, line graphs, pie charts), focusing on both visual and logical reasoning.
- **Task Categories**: `Knowledge`, `MultiModal`, `QA`
- **Evaluation Metrics**: `relaxed_acc`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Subsets**: `augmented_test`, `human_test`

- **Prompt Template**: 
```text

{question}

The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the a single word answer to the problem.

```

---

### DocVQA

[Back to Top](#vlm-benchmarks)
- **Dataset Name**: `docvqa`
- **Dataset ID**: [lmms-lab/DocVQA](https://modelscope.cn/datasets/lmms-lab/DocVQA/summary)
- **Description**:
  > DocVQA (Document Visual Question Answering) is a benchmark designed to evaluate AI systems on their ability to answer questions based on the content of document images, such as scanned pages, forms, or invoices. Unlike general visual question answering, it requires understanding not just the text extracted by OCR, but also the complex layout, structure, and visual elements of a document.
- **Task Categories**: `Knowledge`, `MultiModal`, `QA`
- **Evaluation Metrics**: `anls`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Subsets**: `DocVQA`

- **Prompt Template**: 
```text
Answer the question according to the image using a single word or phrase.
{question}
The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the question.
```

---

### InfoVQA

[Back to Top](#vlm-benchmarks)
- **Dataset Name**: `infovqa`
- **Dataset ID**: [lmms-lab/DocVQA](https://modelscope.cn/datasets/lmms-lab/DocVQA/summary)
- **Description**:
  > InfoVQA (Information Visual Question Answering) is a benchmark designed to evaluate how well AI models can answer questions based on information-dense images, such as charts, graphs, diagrams, maps, and infographics.
- **Task Categories**: `Knowledge`, `MultiModal`, `QA`
- **Evaluation Metrics**: `anls`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Subsets**: `InfographicVQA`

- **Prompt Template**: 
```text
Answer the question according to the image using a single word or phrase.
{question}
The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the question.
```

---

### MathVista

[Back to Top](#vlm-benchmarks)
- **Dataset Name**: `math_vista`
- **Dataset ID**: [evalscope/MathVista](https://modelscope.cn/datasets/evalscope/MathVista/summary)
- **Description**:
  > MathVista is a consolidated Mathematical reasoning benchmark within Visual contexts. It consists of three newly created datasets, IQTest, FunctionQA, and PaperQA, which address the missing visual domains and are tailored to evaluate logical reasoning on puzzle test figures, algebraic reasoning over functional plots, and scientific reasoning with academic paper figures, respectively. It also incorporates 9 MathQA datasets and 19 VQA datasets from the literature, which significantly enrich the diversity and complexity of visual perception and mathematical reasoning challenges within our benchmark. In total, MathVista includes 6,141 examples collected from 31 different datasets.
- **Task Categories**: `MCQ`, `Math`, `MultiModal`, `Reasoning`
- **Evaluation Metrics**: `acc`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Subsets**: `default`

- **Prompt Template**: 
```text

Solve the following problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{question}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \boxed command.

```

---

### MMBench

[Back to Top](#vlm-benchmarks)
- **Dataset Name**: `mm_bench`
- **Dataset ID**: [lmms-lab/MMBench](https://modelscope.cn/datasets/lmms-lab/MMBench/summary)
- **Description**:
  > MMBench is a comprehensive evaluation pipeline comprised of meticulously curated multimodal dataset and a novel circulareval strategy using ChatGPT. It is comprised of 20 ability dimensions defined by MMBench. It also contains chinese version with translated question.
- **Task Categories**: `Knowledge`, `MultiModal`, `QA`
- **Evaluation Metrics**: `acc`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Subsets**: `cn`, `en`

- **Prompt Template**: 
```text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}. Think step by step before answering.

{question}

{choices}
```

---

### MMStar

[Back to Top](#vlm-benchmarks)
- **Dataset Name**: `mm_star`
- **Dataset ID**: [evalscope/MMStar](https://modelscope.cn/datasets/evalscope/MMStar/summary)
- **Description**:
  > MMStar: an elite vision-indispensible multi-modal benchmark, aiming to ensure each curated sample exhibits visual dependency, minimal data leakage, and requires advanced multi-modal capabilities.
- **Task Categories**: `Knowledge`, `MCQ`, `MultiModal`
- **Evaluation Metrics**: `acc`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Subsets**: `coarse perception`, `fine-grained perception`, `instance reasoning`, `logical reasoning`, `math`, `science & technology`

- **Prompt Template**: 
```text
Answer the following multiple choice question.
The last line of your response should be of the following format:
'ANSWER: $LETTER' (without quotes)
where LETTER is one of A,B,C,D. Think step by step before answering.

{question}
```

---

### MMMU

[Back to Top](#vlm-benchmarks)
- **Dataset Name**: `mmmu`
- **Dataset ID**: [AI-ModelScope/MMMU](https://modelscope.cn/datasets/AI-ModelScope/MMMU/summary)
- **Description**:
  > MMMU (A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI) benchmark designed to evaluate multimodal models on massive multi-discipline tasks demanding college-level subject knowledge and deliberate reasoning. MMMU includes 11.5K meticulously collected multimodal questions from college exams, quizzes, and textbooks, covering six core disciplines: Art & Design, Business, Science, Health & Medicine, Humanities & Social Science, and Tech & Engineering. These questions span 30 subjects and 183 subfields, comprising 30 highly heterogeneous image types, such as charts, diagrams, maps, tables, music sheets, and chemical structures.
- **Task Categories**: `Knowledge`, `MultiModal`, `QA`
- **Evaluation Metrics**: `acc`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Subsets**: `Accounting`, `Agriculture`, `Architecture_and_Engineering`, `Art_Theory`, `Art`, `Basic_Medical_Science`, `Biology`, `Chemistry`, `Clinical_Medicine`, `Computer_Science`, `Design`, `Diagnostics_and_Laboratory_Medicine`, `Economics`, `Electronics`, `Energy_and_Power`, `Finance`, `Geography`, `History`, `Literature`, `Manage`, `Marketing`, `Materials`, `Math`, `Mechanical_Engineering`, `Music`, `Pharmacy`, `Physics`, `Psychology`, `Public_Health`, `Sociology`

- **Prompt Template**: 
```text

Solve the following problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{question}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \boxed command.

```

---

### MMMU-PRO

[Back to Top](#vlm-benchmarks)
- **Dataset Name**: `mmmu_pro`
- **Dataset ID**: [AI-ModelScope/MMMU_Pro](https://modelscope.cn/datasets/AI-ModelScope/MMMU_Pro/summary)
- **Description**:
  > MMMU-Pro is an enhanced multimodal benchmark designed to rigorously assess the true understanding capabilities of advanced AI models across multiple modalities. It builds upon the original MMMU benchmark by introducing several key improvements that make it more challenging and realistic, ensuring that models are evaluated on their genuine ability to integrate and comprehend both visual and textual information.
- **Task Categories**: `Knowledge`, `MCQ`, `MultiModal`
- **Evaluation Metrics**: `acc`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Subsets**: `Accounting`, `Agriculture`, `Architecture_and_Engineering`, `Art_Theory`, `Art`, `Basic_Medical_Science`, `Biology`, `Chemistry`, `Clinical_Medicine`, `Computer_Science`, `Design`, `Diagnostics_and_Laboratory_Medicine`, `Economics`, `Electronics`, `Energy_and_Power`, `Finance`, `Geography`, `History`, `Literature`, `Manage`, `Marketing`, `Materials`, `Math`, `Mechanical_Engineering`, `Music`, `Pharmacy`, `Physics`, `Psychology`, `Public_Health`, `Sociology`

- **Extra Parameters**: 
```json
{
    "dataset_format": "# choose from ['standard (4 options)', 'standard (10 options)', 'vision'], default 'standard (4 options)'"
}
```
- **Prompt Template**: 
```text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}. Think step by step before answering.

{question}

{choices}
```

---

### OCRBench

[Back to Top](#vlm-benchmarks)
- **Dataset Name**: `ocr_bench`
- **Dataset ID**: [evalscope/OCRBench](https://modelscope.cn/datasets/evalscope/OCRBench/summary)
- **Description**:
  > OCRBench is a comprehensive evaluation benchmark designed to assess the OCR capabilities of Large Multimodal Models. It comprises five components: Text Recognition, SceneText-Centric VQA, Document-Oriented VQA, Key Information Extraction, and Handwritten Mathematical Expression Recognition. The benchmark includes 1000 question-answer pairs, and all the answers undergo manual verification and correction to ensure a more precise evaluation.
- **Task Categories**: `Knowledge`, `MultiModal`, `QA`
- **Evaluation Metrics**: `acc`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Subsets**: `Artistic Text Recognition`, `Digit String Recognition`, `Doc-oriented VQA`, `Handwriting Recognition`, `Handwritten Mathematical Expression Recognition`, `Irregular Text Recognition`, `Key Information Extraction`, `Non-Semantic Text Recognition`, `Regular Text Recognition`, `Scene Text-centric VQA`

- **Prompt Template**: 
```text
{question}
```

---

### OCRBench-v2

[Back to Top](#vlm-benchmarks)
- **Dataset Name**: `ocr_bench_v2`
- **Dataset ID**: [evalscope/OCRBench_v2](https://modelscope.cn/datasets/evalscope/OCRBench_v2/summary)
- **Description**:
  > OCRBench v2 is a large-scale bilingual text-centric benchmark with currently the most comprehensive set of tasks (4x more tasks than the previous multi-scene benchmark OCRBench), the widest coverage of scenarios (31 diverse scenarios including street scene, receipt, formula, diagram, and so on), and thorough evaluation metrics, with a total of 10, 000 human-verified question-answering pairs and a high proportion of difficult samples.
- **Task Categories**: `Knowledge`, `MultiModal`, `QA`
- **Evaluation Metrics**: `acc`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Subsets**: `APP agent en`, `ASCII art classification en`, `VQA with position en`, `chart parsing en`, `cognition VQA cn`, `cognition VQA en`, `diagram QA en`, `document classification en`, `document parsing cn`, `document parsing en`, `fine-grained text recognition en`, `formula recognition cn`, `formula recognition en`, `full-page OCR cn`, `full-page OCR en`, `handwritten answer extraction cn`, `key information extraction cn`, `key information extraction en`, `key information mapping en`, `math QA en`, `reasoning VQA cn`, `reasoning VQA en`, `science QA en`, `table parsing cn`, `table parsing en`, `text counting en`, `text grounding en`, `text recognition en`, `text spotting en`, `text translation cn`

- **Prompt Template**: 
```text
{question}
```

---

### OlympiadBench

[Back to Top](#vlm-benchmarks)
- **Dataset Name**: `olympiad_bench`
- **Dataset ID**: [AI-ModelScope/OlympiadBench](https://modelscope.cn/datasets/AI-ModelScope/OlympiadBench/summary)
- **Description**:
  > OlympiadBench is an Olympiad-level bilingual multimodal scientific benchmark, featuring 8,476 problems from Olympiad-level mathematics and physics competitions, including the Chinese college entrance exam. In the subsets: `OE` stands for `Open-Ended`, `TP` stands for `Theorem Proving`, `MM` stands for `Multimodal`, `TO` stands for `Text-Only`, `CEE` stands for `Chinese Entrance Exam`, `COMP` stands for `Comprehensive`. **Note: The `TP` subsets can't be evaluated with auto-judge for now**.
- **Task Categories**: `Math`, `Reasoning`
- **Evaluation Metrics**: `acc`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Subsets**: `OE_MM_maths_en_COMP`, `OE_MM_maths_zh_CEE`, `OE_MM_maths_zh_COMP`, `OE_MM_physics_en_COMP`, `OE_MM_physics_zh_CEE`, `OE_TO_maths_en_COMP`, `OE_TO_maths_zh_CEE`, `OE_TO_maths_zh_COMP`, `OE_TO_physics_en_COMP`, `OE_TO_physics_zh_CEE`, `TP_MM_maths_en_COMP`, `TP_MM_maths_zh_CEE`, `TP_MM_maths_zh_COMP`, `TP_MM_physics_en_COMP`, `TP_TO_maths_en_COMP`, `TP_TO_maths_zh_CEE`, `TP_TO_maths_zh_COMP`, `TP_TO_physics_en_COMP`

- **Prompt Template**: 
```text
{question}
Please reason step by step, and put your final answer within \boxed{{}}.
```

---

### OmniBench

[Back to Top](#vlm-benchmarks)
- **Dataset Name**: `omni_bench`
- **Dataset ID**: [m-a-p/OmniBench](https://modelscope.cn/datasets/m-a-p/OmniBench/summary)
- **Description**:
  > OmniBench, a pioneering universal multimodal benchmark designed to rigorously evaluate MLLMs' capability to recognize, interpret, and reason across visual, acoustic, and textual inputs simultaneously.
- **Task Categories**: `Knowledge`, `MCQ`, `MultiModal`
- **Evaluation Metrics**: `acc`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Subsets**: `default`

- **Extra Parameters**: 
```json
{
    "use_image": true,
    "use_audio": true
}
```
- **Prompt Template**: 
```text
Answer the following multiple choice question based on the image and audio content. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}. Think step by step before answering.

{question}

{choices}
```

---

### RealWorldQA

[Back to Top](#vlm-benchmarks)
- **Dataset Name**: `real_world_qa`
- **Dataset ID**: [lmms-lab/RealWorldQA](https://modelscope.cn/datasets/lmms-lab/RealWorldQA/summary)
- **Description**:
  > RealWorldQA is a benchmark designed to evaluate the real-world spatial understanding capabilities of multimodal AI models, contributed by XAI. It assesses how well these models comprehend physical environments. The benchmark consists of 700+ images, each accompanied by a question and a verifiable answer. These images are drawn from real-world scenarios, including those captured from vehicles. The goal is to advance AI models' understanding of our physical world.
- **Task Categories**: `Knowledge`, `MultiModal`, `QA`
- **Evaluation Metrics**: `acc`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Subsets**: `default`

- **Prompt Template**: 
```text
Read the picture and solve the following problem step by step.The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{question}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \boxed command.
```
