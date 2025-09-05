# VLM Benchmarks

Below is the list of supported VLM benchmarks. Click on a benchmark name to jump to details.

| Benchmark Name | Pretty Name | Task Categories |
|------------|----------|----------|
| `math_vista` | [MathVista](#mathvista) | `MCQ`, `Math`, `MultiModal`, `Reasoning` |
| `mmmu` | [MMMU](#mmmu) | `Knowledge`, `MultiModal`, `QA` |
| `mmmu_pro` | [MMMU-PRO](#mmmu-pro) | `Knowledge`, `MultiModal`, `QA` |

---

## Benchmark Details

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
- **Task Categories**: `Knowledge`, `MultiModal`, `QA`
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
