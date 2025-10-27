# LLM评测集

以下是支持的LLM评测集列表，点击数据集标准名称可跳转详细信息。

| 数据集名称 | 标准名称 | 任务类别 |
|------------|----------|----------|
| `aa_lcr` | [AA-LCR](#aa-lcr) | `Knowledge`, `LongContext`, `Reasoning` |
| `aime24` | [AIME-2024](#aime-2024) | `Math`, `Reasoning` |
| `aime25` | [AIME-2025](#aime-2025) | `Math`, `Reasoning` |
| `alpaca_eval` | [AlpacaEval2.0](#alpacaeval20) | `Arena`, `InstructionFollowing` |
| `amc` | [AMC](#amc) | `Math`, `Reasoning` |
| `arc` | [ARC](#arc) | `MCQ`, `Reasoning` |
| `arena_hard` | [ArenaHard](#arenahard) | `Arena`, `InstructionFollowing` |
| `bbh` | [BBH](#bbh) | `Reasoning` |
| `bfcl_v3` | [BFCL-v3](#bfcl-v3) | `FunctionCalling` |
| `broad-twitter-corpus` | [BroadTwitterCorpus](#broadtwittercorpus) | `Knowledge`, `NER` |
| `ceval` | [C-Eval](#c-eval) | `Chinese`, `Knowledge`, `MCQ` |
| `chinese_simpleqa` | [Chinese-SimpleQA](#chinese-simpleqa) | `Chinese`, `Knowledge`, `QA` |
| `cmmlu` | [C-MMLU](#c-mmlu) | `Chinese`, `Knowledge`, `MCQ` |
| `commonsense_qa` | [CommonsenseQA](#commonsenseqa) | `Commonsense`, `MCQ`, `Reasoning` |
| `competition_math` | [MATH](#math) | `Math`, `Reasoning` |
| `conll2003` | [CoNLL2003](#conll2003) | `Knowledge`, `NER` |
| `copious` | [Copious](#copious) | `Knowledge`, `NER` |
| `cross-ner` | [CrossNER](#crossner) | `Knowledge`, `NER` |
| `data_collection` | [Data-Collection](#data-collection) | `Custom` |
| `docmath` | [DocMath](#docmath) | `LongContext`, `Math`, `Reasoning` |
| `drop` | [DROP](#drop) | `Reasoning` |
| `frames` | [FRAMES](#frames) | `LongContext`, `Reasoning` |
| `general_arena` | [GeneralArena](#generalarena) | `Arena`, `Custom` |
| `general_mcq` | [General-MCQ](#general-mcq) | `Custom`, `MCQ` |
| `general_qa` | [General-QA](#general-qa) | `Custom`, `QA` |
| `genia-ner` | [GeniaNER](#genianer) | `Knowledge`, `NER` |
| `gpqa_diamond` | [GPQA-Diamond](#gpqa-diamond) | `Knowledge`, `MCQ` |
| `gsm8k` | [GSM8K](#gsm8k) | `Math`, `Reasoning` |
| `halueval` | [HaluEval](#halueval) | `Hallucination`, `Knowledge`, `Yes/No` |
| `harvey-ner` | [HarveyNER](#harveyner) | `Knowledge`, `NER` |
| `health_bench` | [HealthBench](#healthbench) | `Knowledge`, `QA` |
| `hellaswag` | [HellaSwag](#hellaswag) | `Commonsense`, `Knowledge`, `MCQ` |
| `hle` | [Humanity's-Last-Exam](#humanitys-last-exam) | `Knowledge`, `QA` |
| `humaneval` | [HumanEval](#humaneval) | `Coding` |
| `ifeval` | [IFEval](#ifeval) | `InstructionFollowing` |
| `iquiz` | [IQuiz](#iquiz) | `Chinese`, `Knowledge`, `MCQ` |
| `live_code_bench` | [Live-Code-Bench](#live-code-bench) | `Coding` |
| `logi_qa` | [LogiQA](#logiqa) | `MCQ`, `Reasoning` |
| `maritime_bench` | [MaritimeBench](#maritimebench) | `Chinese`, `Knowledge`, `MCQ` |
| `math_500` | [MATH-500](#math-500) | `Math`, `Reasoning` |
| `math_qa` | [MathQA](#mathqa) | `MCQ`, `Math`, `Reasoning` |
| `minerva_math` | [Minerva-Math](#minerva-math) | `Math`, `Reasoning` |
| `mit-movie-trivia` | [MIT-Movie-Trivia](#mit-movie-trivia) | `Knowledge`, `NER` |
| `mit-restaurant` | [MIT-Restaurant](#mit-restaurant) | `Knowledge`, `NER` |
| `mmlu` | [MMLU](#mmlu) | `Knowledge`, `MCQ` |
| `mmlu_pro` | [MMLU-Pro](#mmlu-pro) | `Knowledge`, `MCQ` |
| `mmlu_redux` | [MMLU-Redux](#mmlu-redux) | `Knowledge`, `MCQ` |
| `mri_mcqa` | [MRI-MCQA](#mri-mcqa) | `Knowledge`, `MCQ` |
| `multi_if` | [Multi-IF](#multi-if) | `InstructionFollowing`, `MultiLingual`, `MultiTurn` |
| `musr` | [MuSR](#musr) | `MCQ`, `Reasoning` |
| `needle_haystack` | [Needle-in-a-Haystack](#needle-in-a-haystack) | `LongContext`, `Retrieval` |
| `ontonotes5` | [OntoNotes5](#ontonotes5) | `Knowledge`, `NER` |
| `piqa` | [PIQA](#piqa) | `Commonsense`, `MCQ`, `Reasoning` |
| `poly_math` | [PolyMath](#polymath) | `Math`, `MultiLingual`, `Reasoning` |
| `process_bench` | [ProcessBench](#processbench) | `Math`, `Reasoning` |
| `pubmedqa` | [PubMedQA](#pubmedqa) | `Knowledge`, `Yes/No` |
| `qasc` | [QASC](#qasc) | `Knowledge`, `MCQ` |
| `race` | [RACE](#race) | `MCQ`, `Reasoning` |
| `simple_qa` | [SimpleQA](#simpleqa) | `Knowledge`, `QA` |
| `siqa` | [SIQA](#siqa) | `Commonsense`, `MCQ`, `Reasoning` |
| `super_gpqa` | [SuperGPQA](#supergpqa) | `Knowledge`, `MCQ` |
| `tau_bench` | [τ-bench](#τ-bench) | `FunctionCalling`, `Reasoning` |
| `tool_bench` | [ToolBench-Static](#toolbench-static) | `FunctionCalling`, `Reasoning` |
| `trivia_qa` | [TriviaQA](#triviaqa) | `QA`, `ReadingComprehension` |
| `truthful_qa` | [TruthfulQA](#truthfulqa) | `Knowledge` |
| `winogrande` | [Winogrande](#winogrande) | `MCQ`, `Reasoning` |
| `wnut2017` | [WNUT2017](#wnut2017) | `Knowledge`, `NER` |

---

## 数据集详情

### AA-LCR

[返回目录](#llm评测集)
- **数据集名称**: `aa_lcr`
- **数据集ID**: [evalscope/AA-LCR](https://modelscope.cn/datasets/evalscope/AA-LCR/summary)
- **数据集描述**:
  > AA-LCR（人工分析长上下文检索）是一个用于评估语言模型在多文档场景下长上下文检索与推理能力的基准。
- **任务类别**: `Knowledge`, `LongContext`, `Reasoning`
- **评估指标**: `acc`
- **需要LLM Judge**: 是
- **默认提示方式**: 0-shot
- **数据集子集**: `default`

- **额外参数**: 
```json
{
    "text_dir": null
}
```
- **提示模板**: 
```text

BEGIN INPUT DOCUMENTS

{documents_text}

END INPUT DOCUMENTS

Answer the following question using the input documents provided above.

START QUESTION

{question}

END QUESTION

```

---

### AIME-2024

[返回目录](#llm评测集)
- **数据集名称**: `aime24`
- **数据集ID**: [HuggingFaceH4/aime_2024](https://modelscope.cn/datasets/HuggingFaceH4/aime_2024/summary)
- **数据集描述**:
  > AIME 2024 基准基于美国数学邀请赛（AIME）的题目，该赛事是一项享有盛誉的高中数学竞赛。此基准通过生成逐步解答并提供正确最终答案，来测试模型解决复杂数学问题的能力。
- **任务类别**: `Math`, `Reasoning`
- **评估指标**: `{'acc': {'numeric': True}}`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `default`

- **提示模板**: 
```text
{question}
Please reason step by step, and put your final answer within \boxed{{}}.
```

---

### AIME-2025

[返回目录](#llm评测集)
- **数据集名称**: `aime25`
- **数据集ID**: [opencompass/AIME2025](https://modelscope.cn/datasets/opencompass/AIME2025/summary)
- **数据集描述**:
  > AIME 2025 基准基于美国数学邀请赛（AIME）的题目，该赛事是一项享有盛誉的高中数学竞赛。此基准通过生成逐步解题过程并给出正确最终答案，来测试模型解决复杂数学问题的能力。
- **任务类别**: `Math`, `Reasoning`
- **评估指标**: `{'acc': {'numeric': True}}`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `AIME2025-II`, `AIME2025-I`

- **提示模板**: 
```text
{question}
Please reason step by step, and put your final answer within \boxed{{}}.
```

---

### AlpacaEval2.0

[返回目录](#llm评测集)
- **数据集名称**: `alpaca_eval`
- **数据集ID**: [AI-ModelScope/alpaca_eval](https://modelscope.cn/datasets/AI-ModelScope/alpaca_eval/summary)
- **数据集描述**:
  > Alpaca Eval 2.0 是一个改进的指令遵循语言模型评估框架，具备升级的自动标注器、更新的基线模型和持续偏好计算，可提供更准确且成本更低的模型评估。目前不支持“长度控制胜率”；官方裁判模型为 `gpt-4-1106-preview`，基线模型为 `gpt-4-turbo`。
- **任务类别**: `Arena`, `InstructionFollowing`
- **评估指标**: `winrate`
- **需要LLM Judge**: 是
- **默认提示方式**: 0-shot
- **数据集子集**: `alpaca_eval_gpt4_baseline`

- **提示模板**: 
```text
{question}
```

---

### AMC

[返回目录](#llm评测集)
- **数据集名称**: `amc`
- **数据集ID**: [evalscope/amc_22-24](https://modelscope.cn/datasets/evalscope/amc_22-24/summary)
- **数据集描述**:
  > AMC（美国数学竞赛）是一系列面向高中生的数学竞赛。
- **任务类别**: `Math`, `Reasoning`
- **评估指标**: `{'acc': {'numeric': True}}`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `amc22`, `amc23`, `amc24`

- **提示模板**: 
```text
{question}
Please reason step by step, and put your final answer within \boxed{{}}.
```

---

### ARC

[返回目录](#llm评测集)
- **数据集名称**: `arc`
- **数据集ID**: [allenai/ai2_arc](https://modelscope.cn/datasets/allenai/ai2_arc/summary)
- **数据集描述**:
  > ARC（AI2推理挑战）基准通过科学考试中的选择题来评估AI模型的推理能力，包含难度不同的两个子集：ARC-Easy和ARC-Challenge。
- **任务类别**: `MCQ`, `Reasoning`
- **评估指标**: `acc`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `ARC-Challenge`, `ARC-Easy`

- **提示模板**: 
```text
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}.

{question}

{choices}
```

---

### ArenaHard

[返回目录](#llm评测集)
- **数据集名称**: `arena_hard`
- **数据集ID**: [AI-ModelScope/arena-hard-auto-v0.1](https://modelscope.cn/datasets/AI-ModelScope/arena-hard-auto-v0.1/summary)
- **数据集描述**:
  > ArenaHard 是一个用于评估大语言模型在竞争环境中表现的基准，通过一系列任务将模型相互对战，以衡量其相对优劣。该基准包含需要推理、理解和生成能力的高难度任务。目前不支持“风格控制胜率”；官方裁判模型为 `gpt-4-1106-preview`，基线模型为 `gpt-4-0314`。
- **任务类别**: `Arena`, `InstructionFollowing`
- **评估指标**: `winrate`
- **需要LLM Judge**: 是
- **默认提示方式**: 0-shot
- **数据集子集**: `default`

- **提示模板**: 
```text
{question}
```

---

### BBH

[返回目录](#llm评测集)
- **数据集名称**: `bbh`
- **数据集ID**: [evalscope/bbh](https://modelscope.cn/datasets/evalscope/bbh/summary)
- **数据集描述**:
  > BBH（Big Bench Hard）基准是一组具有挑战性的任务，旨在评估AI模型的推理能力。它包含开放式和选择题任务，涵盖多种推理技能。
- **任务类别**: `Reasoning`
- **评估指标**: `acc`
- **需要LLM Judge**: 否
- **默认提示方式**: 3-shot
- **数据集子集**: `boolean_expressions`, `causal_judgement`, `date_understanding`, `disambiguation_qa`, `dyck_languages`, `formal_fallacies`, `geometric_shapes`, `hyperbaton`, `logical_deduction_five_objects`, `logical_deduction_seven_objects`, `logical_deduction_three_objects`, `movie_recommendation`, `multistep_arithmetic_two`, `navigate`, `object_counting`, `penguins_in_a_table`, `reasoning_about_colored_objects`, `ruin_names`, `salient_translation_error_detection`, `snarks`, `sports_understanding`, `temporal_sequences`, `tracking_shuffled_objects_five_objects`, `tracking_shuffled_objects_seven_objects`, `tracking_shuffled_objects_three_objects`, `web_of_lies`, `word_sorting`

- **提示模板**: 
```text
Q: {question}
A: Let's think step by step. Put your final answer in the format of "So the answer is $ANSWER" (without quotes and markdown) where $ANSWER is the answer to the problem.

```

---

### BFCL-v3

[返回目录](#llm评测集)
- **数据集名称**: `bfcl_v3`
- **数据集ID**: [AI-ModelScope/bfcl_v3](https://modelscope.cn/datasets/AI-ModelScope/bfcl_v3/summary)
- **数据集描述**:
  > Berkeley Function Calling Leaderboard (BFCL) 是首个专注于评估大语言模型（LLM）调用函数能力的**全面且可执行的函数调用评测**。与以往评测不同，BFCL 考虑了多种函数调用形式、多样化场景以及可执行性。评测前需安装 `pip install bfcl-eval==2025.6.16`。[使用示例](https://evalscope.readthedocs.io/zh-cn/latest/third_party/bfcl_v3.html)
- **任务类别**: `FunctionCalling`
- **评估指标**: `acc`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `irrelevance`, `java`, `javascript`, `live_irrelevance`, `live_multiple`, `live_parallel_multiple`, `live_parallel`, `live_relevance`, `live_simple`, `multi_turn_base`, `multi_turn_long_context`, `multi_turn_miss_func`, `multi_turn_miss_param`, `multiple`, `parallel_multiple`, `parallel`, `simple`

- **额外参数**: 
```json
{
    "underscore_to_dot": true,
    "is_fc_model": true
}
```

---

### BroadTwitterCorpus

[返回目录](#llm评测集)
- **数据集名称**: `broad-twitter-corpus`
- **数据集ID**: [extraordinarylab/broad-twitter-corpus](https://modelscope.cn/datasets/extraordinarylab/broad-twitter-corpus/summary)
- **数据集描述**:
  > BroadTwitterCorpus 是一个通过分层抽样在不同时间、地点和社会用途下收集的推文数据集。其目标是涵盖广泛的活动，从而提供一个更能代表这种最难处理的社交媒体形式中所用语言的数据集。
- **任务类别**: `Knowledge`, `NER`
- **评估指标**: `accuracy`, `f1_score`, `precision`, `recall`
- **需要LLM Judge**: 否
- **默认提示方式**: 5-shot
- **数据集子集**: `default`

- **提示模板**: 
```text
You are a named entity recognition system that identifies the following entity types:
{entities}

Process the provided text and mark all named entities with XML-style tags.

For example:
<person>John Smith</person> works at <organization>Google</organization> in <location>Mountain View</location>.

Available entity tags: {entity_list}

INSTRUCTIONS:
1. Wrap your entire response in <response>...</response> tags.
2. Inside these tags, include the original text with entity tags inserted.
3. Do not change the original text in any way (preserve spacing, punctuation, case, etc.).
4. Tag ALL entities you can identify using the exact tag names provided.
5. Do not include explanations, just the tagged text.
6. If entity spans overlap, choose the most specific entity type.
7. Ensure every opening tag has a matching closing tag.

Text to process:
{text}

```

---

### C-Eval

[返回目录](#llm评测集)
- **数据集名称**: `ceval`
- **数据集ID**: [evalscope/ceval](https://modelscope.cn/datasets/evalscope/ceval/summary)
- **数据集描述**:
  > C-Eval 是一个评估AI模型在包括STEM、社会科学和人文学科等多个学科中文考试中表现的基准，包含测试知识和推理能力的多项选择题。
- **任务类别**: `Chinese`, `Knowledge`, `MCQ`
- **评估指标**: `acc`
- **需要LLM Judge**: 否
- **默认提示方式**: 5-shot
- **数据集子集**: `accountant`, `advanced_mathematics`, `art_studies`, `basic_medicine`, `business_administration`, `chinese_language_and_literature`, `civil_servant`, `clinical_medicine`, `college_chemistry`, `college_economics`, `college_physics`, `college_programming`, `computer_architecture`, `computer_network`, `discrete_mathematics`, `education_science`, `electrical_engineer`, `environmental_impact_assessment_engineer`, `fire_engineer`, `high_school_biology`, `high_school_chemistry`, `high_school_chinese`, `high_school_geography`, `high_school_history`, `high_school_mathematics`, `high_school_physics`, `high_school_politics`, `ideological_and_moral_cultivation`, `law`, `legal_professional`, `logic`, `mao_zedong_thought`, `marxism`, `metrology_engineer`, `middle_school_biology`, `middle_school_chemistry`, `middle_school_geography`, `middle_school_history`, `middle_school_mathematics`, `middle_school_physics`, `middle_school_politics`, `modern_chinese_history`, `operating_system`, `physician`, `plant_protection`, `probability_and_statistics`, `professional_tour_guide`, `sports_science`, `tax_accountant`, `teacher_qualification`, `urban_and_rural_planner`, `veterinary_medicine`

- **提示模板**: 
```text
以下是中国关于{subject}的单项选择题，请选出其中的正确答案。你的回答的最后一行应该是这样的格式："答案：LETTER"（不带引号），其中 LETTER 是 A、B、C、D 中的一个。

问题：{question}
选项：
{choices}

```

---

### Chinese-SimpleQA

[返回目录](#llm评测集)
- **数据集名称**: `chinese_simpleqa`
- **数据集ID**: [AI-ModelScope/Chinese-SimpleQA](https://modelscope.cn/datasets/AI-ModelScope/Chinese-SimpleQA/summary)
- **数据集描述**:
  > Chinese SimpleQA 是一个中文问答数据集，旨在评估语言模型在简单事实问题上的表现。该数据集涵盖多种主题，用于测试模型理解和生成中文正确答案的能力。
- **任务类别**: `Chinese`, `Knowledge`, `QA`
- **评估指标**: `is_correct`, `is_incorrect`, `is_not_attempted`
- **需要LLM Judge**: 是
- **默认提示方式**: 0-shot
- **数据集子集**: `中华文化`, `人文与社会科学`, `工程、技术与应用科学`, `生活、艺术与文化`, `社会`, `自然与自然科学`

- **提示模板**: 
```text
请回答问题：

{question}
```

---

### C-MMLU

[返回目录](#llm评测集)
- **数据集名称**: `cmmlu`
- **数据集ID**: [evalscope/cmmlu](https://modelscope.cn/datasets/evalscope/cmmlu/summary)
- **数据集描述**:
  > C-MMLU 是一个用于评估AI模型在中文语言任务上性能的基准，包括阅读理解、文本分类等。
- **任务类别**: `Chinese`, `Knowledge`, `MCQ`
- **评估指标**: `acc`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `agronomy`, `anatomy`, `ancient_chinese`, `arts`, `astronomy`, `business_ethics`, `chinese_civil_service_exam`, `chinese_driving_rule`, `chinese_food_culture`, `chinese_foreign_policy`, `chinese_history`, `chinese_literature`, `chinese_teacher_qualification`, `clinical_knowledge`, `college_actuarial_science`, `college_education`, `college_engineering_hydrology`, `college_law`, `college_mathematics`, `college_medical_statistics`, `college_medicine`, `computer_science`, `computer_security`, `conceptual_physics`, `construction_project_management`, `economics`, `education`, `electrical_engineering`, `elementary_chinese`, `elementary_commonsense`, `elementary_information_and_technology`, `elementary_mathematics`, `ethnology`, `food_science`, `genetics`, `global_facts`, `high_school_biology`, `high_school_chemistry`, `high_school_geography`, `high_school_mathematics`, `high_school_physics`, `high_school_politics`, `human_sexuality`, `international_law`, `journalism`, `jurisprudence`, `legal_and_moral_basis`, `logical`, `machine_learning`, `management`, `marketing`, `marxist_theory`, `modern_chinese`, `nutrition`, `philosophy`, `professional_accounting`, `professional_law`, `professional_medicine`, `professional_psychology`, `public_relations`, `security_study`, `sociology`, `sports_science`, `traditional_chinese_medicine`, `virology`, `world_history`, `world_religions`

- **提示模板**: 
```text
回答下面的单项选择题，请选出其中的正确答案。你的回答的最后一行应该是这样的格式："答案：LETTER"（不带引号），其中 LETTER 是 {letters} 中的一个。请在回答前进行一步步思考。

问题：{question}
选项：
{choices}

```

---

### CommonsenseQA

[返回目录](#llm评测集)
- **数据集名称**: `commonsense_qa`
- **数据集ID**: [extraordinarylab/commonsense-qa](https://modelscope.cn/datasets/extraordinarylab/commonsense-qa/summary)
- **数据集描述**:
  > CommonsenseQA 需要不同类型的常识知识来预测正确答案。
- **任务类别**: `Commonsense`, `MCQ`, `Reasoning`
- **评估指标**: `acc`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `default`

- **提示模板**: 
```text
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}.

{question}

{choices}
```

---

### MATH

[返回目录](#llm评测集)
- **数据集名称**: `competition_math`
- **数据集ID**: [evalscope/competition_math](https://modelscope.cn/datasets/evalscope/competition_math/summary)
- **数据集描述**:
  > MATH（数学）基准通过算术、代数、几何等多种题型，评估AI模型的数学推理能力。
- **任务类别**: `Math`, `Reasoning`
- **评估指标**: `{'acc': {'numeric': True}}`
- **需要LLM Judge**: 否
- **默认提示方式**: 4-shot
- **数据集子集**: `Level 1`, `Level 2`, `Level 3`, `Level 4`, `Level 5`

- **提示模板**: 
```text
Problem:
{question}

Please reason step by step, and put your final answer within \boxed{{}}.

```

---

### CoNLL2003

[返回目录](#llm评测集)
- **数据集名称**: `conll2003`
- **数据集ID**: [evalscope/conll2003](https://modelscope.cn/datasets/evalscope/conll2003/summary)
- **数据集描述**:
  > ConLL-2003 数据集用于命名实体识别（NER）任务，是 ConLL-2003 共享任务会议的一部分，包含标注了人名、组织、地点及各类名称的文本。
- **任务类别**: `Knowledge`, `NER`
- **评估指标**: `accuracy`, `f1_score`, `precision`, `recall`
- **需要LLM Judge**: 否
- **默认提示方式**: 5-shot
- **数据集子集**: `default`

- **提示模板**: 
```text
You are a named entity recognition system that identifies the following entity types:
{entities}

Process the provided text and mark all named entities with XML-style tags.

For example:
<person>John Smith</person> works at <organization>Google</organization> in <location>Mountain View</location>.

Available entity tags: {entity_list}

INSTRUCTIONS:
1. Wrap your entire response in <response>...</response> tags.
2. Inside these tags, include the original text with entity tags inserted.
3. Do not change the original text in any way (preserve spacing, punctuation, case, etc.).
4. Tag ALL entities you can identify using the exact tag names provided.
5. Do not include explanations, just the tagged text.
6. If entity spans overlap, choose the most specific entity type.
7. Ensure every opening tag has a matching closing tag.

Text to process:
{text}

```

---

### Copious

[返回目录](#llm评测集)
- **数据集名称**: `copious`
- **数据集ID**: [extraordinarylab/copious](https://modelscope.cn/datasets/extraordinarylab/copious/summary)
- **数据集描述**:
  > Copious语料库是一个涵盖广泛生物多样性实体的黄金标准语料库，包含从生物多样性遗产图书馆下载的668份文档，超过2.6万句句子和2.8万余个实体。
- **任务类别**: `Knowledge`, `NER`
- **评估指标**: `accuracy`, `f1_score`, `precision`, `recall`
- **需要LLM Judge**: 否
- **默认提示方式**: 5-shot
- **数据集子集**: `default`

- **提示模板**: 
```text
You are a named entity recognition system that identifies the following entity types:
{entities}

Process the provided text and mark all named entities with XML-style tags.

For example:
<person>John Smith</person> works at <organization>Google</organization> in <location>Mountain View</location>.

Available entity tags: {entity_list}

INSTRUCTIONS:
1. Wrap your entire response in <response>...</response> tags.
2. Inside these tags, include the original text with entity tags inserted.
3. Do not change the original text in any way (preserve spacing, punctuation, case, etc.).
4. Tag ALL entities you can identify using the exact tag names provided.
5. Do not include explanations, just the tagged text.
6. If entity spans overlap, choose the most specific entity type.
7. Ensure every opening tag has a matching closing tag.

Text to process:
{text}

```

---

### CrossNER

[返回目录](#llm评测集)
- **数据集名称**: `cross-ner`
- **数据集ID**: [extraordinarylab/cross-ner](https://modelscope.cn/datasets/extraordinarylab/cross-ner/summary)
- **数据集描述**:
  > CrossNER 是一个完全标注的命名实体识别（NER）数据集，涵盖五个不同领域（人工智能、文学、音乐、政治、科学）。
- **任务类别**: `Knowledge`, `NER`
- **评估指标**: `accuracy`, `f1_score`, `precision`, `recall`
- **需要LLM Judge**: 否
- **默认提示方式**: 5-shot
- **数据集子集**: `ai`, `literature`, `music`, `politics`, `science`

- **提示模板**: 
```text
You are a named entity recognition system that identifies the following entity types:
{entities}

Process the provided text and mark all named entities with XML-style tags.

For example:
<person>John Smith</person> works at <organization>Google</organization> in <location>Mountain View</location>.

Available entity tags: {entity_list}

INSTRUCTIONS:
1. Wrap your entire response in <response>...</response> tags.
2. Inside these tags, include the original text with entity tags inserted.
3. Do not change the original text in any way (preserve spacing, punctuation, case, etc.).
4. Tag ALL entities you can identify using the exact tag names provided.
5. Do not include explanations, just the tagged text.
6. If entity spans overlap, choose the most specific entity type.
7. Ensure every opening tag has a matching closing tag.

Text to process:
{text}

```

---

### Data-Collection

[返回目录](#llm评测集)
- **数据集名称**: `data_collection`
- **数据集ID**: 
- **数据集描述**:
  > 自定义数据收集，混合多个评估数据集进行统一评估，旨在使用更少的数据实现对模型能力的更全面评估。[使用参考](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/collection/index.html)
- **任务类别**: `Custom`
- **评估指标**: `acc`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `default`


---

### DocMath

[返回目录](#llm评测集)
- **数据集名称**: `docmath`
- **数据集ID**: [yale-nlp/DocMath-Eval](https://modelscope.cn/datasets/yale-nlp/DocMath-Eval/summary)
- **数据集描述**:
  > DocMath-Eval 是一个专注于特定领域内数值推理的综合基准，要求模型理解长篇且专业的文档，并通过数值推理回答问题。
- **任务类别**: `LongContext`, `Math`, `Reasoning`
- **评估指标**: `acc`
- **需要LLM Judge**: 是
- **默认提示方式**: 0-shot
- **数据集子集**: `complong_testmini`, `compshort_testmini`, `simplong_testmini`, `simpshort_testmini`

- **提示模板**: 
```text
Please read the following text and answer the question below.

<text>
{context}
</text>

{question}

Format your response as follows: "Therefore, the answer is (insert answer here)".
```

---

### DROP

[返回目录](#llm评测集)
- **数据集名称**: `drop`
- **数据集ID**: [AI-ModelScope/DROP](https://modelscope.cn/datasets/AI-ModelScope/DROP/summary)
- **数据集描述**:
  > DROP（段落离散推理）基准用于评估AI模型的阅读理解与推理能力，包含多种任务，要求模型阅读文本并根据内容回答问题。
- **任务类别**: `Reasoning`
- **评估指标**: `acc`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `default`

- **提示模板**: 
```text
You will be asked to read a passage and answer a question. {drop_examples}
# Your Task

---
{query}

Think step by step, then write a line of the form "Answer: $ANSWER" at the end of your response.
```

---

### FRAMES

[返回目录](#llm评测集)
- **数据集名称**: `frames`
- **数据集ID**: [iic/frames](https://modelscope.cn/datasets/iic/frames/summary)
- **数据集描述**:
  > FRAMES 是一个综合评估数据集，旨在测试检索增强生成（RAG）系统在事实性、检索准确性和推理能力方面的表现。
- **任务类别**: `LongContext`, `Reasoning`
- **评估指标**: `acc`
- **需要LLM Judge**: 是
- **默认提示方式**: 0-shot
- **数据集子集**: `default`

- **提示模板**: 
```text
Please read the following text and answer the question below.

<text>
{context}
</text>

{question}

Format your response as follows: "Therefore, the answer is (insert answer here)".
```

---

### GeneralArena

[返回目录](#llm评测集)
- **数据集名称**: `general_arena`
- **数据集ID**: general_arena
- **数据集描述**:
  > GeneralArena 是一个自定义基准，旨在通过将大语言模型置于竞争性任务中相互对抗，评估其性能并分析各自的优缺点。您应以字典列表格式提供模型输出，每个字典包含模型名称及其报告路径。有关使用此基准的详细说明，请参阅 [Arena 用户指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/arena.html)。
- **任务类别**: `Arena`, `Custom`
- **评估指标**: `winrate`
- **需要LLM Judge**: 是
- **默认提示方式**: 0-shot
- **数据集子集**: `default`

- **额外参数**: 
```json
{
    "models": [
        {
            "name": "qwen-plus",
            "report_path": "outputs/20250627_172550/reports/qwen-plus"
        },
        {
            "name": "qwen2.5-7b",
            "report_path": "outputs/20250627_172817/reports/qwen2.5-7b-instruct"
        }
    ],
    "baseline": "qwen2.5-7b"
}
```
- **系统提示词**: 
```text
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.

Begin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.

When evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.

Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.

Then consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.

After providing your explanation, you must output only one of the following choices as your final verdict with a label:

1. Assistant A is significantly better: [[A>>B]]
2. Assistant A is slightly better: [[A>B]]
3. Tie, relatively the same: [[A=B]]
4. Assistant B is slightly better: [[B>A]]
5. Assistant B is significantly better: [[B>>A]]

Example output: "My final verdict is tie: [[A=B]]".
```
- **提示模板**: 
```text
<|User Prompt|>
{question}

<|The Start of Assistant A's Answer|>
{answer_1}
<|The End of Assistant A's Answer|>

<|The Start of Assistant B's Answer|>
{answer_2}
<|The End of Assistant B's Answer|>
```

---

### General-MCQ

[返回目录](#llm评测集)
- **数据集名称**: `general_mcq`
- **数据集ID**: general_mcq
- **数据集描述**:
  > 一个用于自定义评估的通用多项选择题问答数据集。有关如何使用此基准的详细说明，请参阅[用户指南](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/custom_dataset/llm.html#mcq)。
- **任务类别**: `Custom`, `MCQ`
- **评估指标**: `acc`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `default`

- **提示模板**: 
```text
回答下面的单项选择题，请选出其中的正确答案。你的回答的最后一行应该是这样的格式："答案：LETTER"（不带引号），其中 LETTER 是 {letters} 中的一个。

问题：{question}
选项：
{choices}

```

---

### General-QA

[返回目录](#llm评测集)
- **数据集名称**: `general_qa`
- **数据集ID**: general_qa
- **数据集描述**:
  > 一个用于自定义评估的通用问答数据集。有关如何使用此基准的详细说明，请参阅[用户指南](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/custom_dataset/llm.html#qa)。
- **任务类别**: `Custom`, `QA`
- **评估指标**: `BLEU`, `Rouge`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `default`

- **提示模板**: 
```text
请回答问题
{question}
```

---

### GeniaNER

[返回目录](#llm评测集)
- **数据集名称**: `genia-ner`
- **数据集ID**: [extraordinarylab/genia-ner](https://modelscope.cn/datasets/extraordinarylab/genia-ner/summary)
- **数据集描述**:
  > GeniaNER 包含 2,000 篇 MEDLINE 摘要，超过 40 万词和近 10 万条生物术语标注，现已发布。
- **任务类别**: `Knowledge`, `NER`
- **评估指标**: `accuracy`, `f1_score`, `precision`, `recall`
- **需要LLM Judge**: 否
- **默认提示方式**: 5-shot
- **数据集子集**: `default`

- **提示模板**: 
```text
You are a named entity recognition system that identifies the following entity types:
{entities}

Process the provided text and mark all named entities with XML-style tags.

For example:
<person>John Smith</person> works at <organization>Google</organization> in <location>Mountain View</location>.

Available entity tags: {entity_list}

INSTRUCTIONS:
1. Wrap your entire response in <response>...</response> tags.
2. Inside these tags, include the original text with entity tags inserted.
3. Do not change the original text in any way (preserve spacing, punctuation, case, etc.).
4. Tag ALL entities you can identify using the exact tag names provided.
5. Do not include explanations, just the tagged text.
6. If entity spans overlap, choose the most specific entity type.
7. Ensure every opening tag has a matching closing tag.

Text to process:
{text}

```

---

### GPQA-Diamond

[返回目录](#llm评测集)
- **数据集名称**: `gpqa_diamond`
- **数据集ID**: [AI-ModelScope/gpqa_diamond](https://modelscope.cn/datasets/AI-ModelScope/gpqa_diamond/summary)
- **数据集描述**:
  > GPQA 是一个用于评估大语言模型（LLM）在复杂数学问题上推理能力的数据集，包含需要逐步推理才能得出正确答案的问题。
- **任务类别**: `Knowledge`, `MCQ`
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

### GSM8K

[返回目录](#llm评测集)
- **数据集名称**: `gsm8k`
- **数据集ID**: [AI-ModelScope/gsm8k](https://modelscope.cn/datasets/AI-ModelScope/gsm8k/summary)
- **数据集描述**:
  > GSM8K（小学数学8K）是一个小学数学问题数据集，旨在评估AI模型的数学推理能力。
- **任务类别**: `Math`, `Reasoning`
- **评估指标**: `acc`
- **需要LLM Judge**: 否
- **默认提示方式**: 4-shot
- **数据集子集**: `main`

- **提示模板**: 
```text
Solve the following math problem step by step. The last line of your response should display the answer enclosed within \boxed{{\text{{$ANSWER}}}}.

Example:

Let's solve the problem step by step.

Problem: Eliza's rate per hour for the first 40 hours she works each week is $10. She also receives an overtime pay of 1.2 times her regular hourly rate. If Eliza worked for 45 hours this week, how much are her earnings for this week?

Step 1: Calculate Eliza's earnings for the first 40 hours. Eliza's hourly rate is $10, so her earnings for the first 40 hours are $10/hour x 40 hours = $400.
Step 2: Calculate Eliza's overtime pay rate. Eliza's overtime pay rate is 1.2 times her regular hourly rate, so her overtime pay rate is $10/hour x 1.2 = $12/hour.
Step 3: Calculate Eliza's earnings for the overtime hours. Eliza worked for 45 hours, so her overtime hours are 45 hours - 40 hours = 5 hours. Her earnings for the overtime hours are $12/hour x 5 hours = $60.
Step 4: Calculate Eliza's total earnings for the week. Eliza's total earnings for the week are her earnings for the first 40 hours plus her earnings for the overtime hours, which is $400 + $60 = $460.

Answer:
\boxed{{\text{{460}}}}

question:
{question}

Remember to put your answer on its own line at the end in the form "\boxed{{\text{{$ANSWER}}}}" (without quotes), where $ANSWER is replaced by the actual answer to the problem.

```

---

### HaluEval

[返回目录](#llm评测集)
- **数据集名称**: `halueval`
- **数据集ID**: [evalscope/HaluEval](https://modelscope.cn/datasets/evalscope/HaluEval/summary)
- **数据集描述**:
  > HaluEval 是一个大规模生成并由人工标注的幻觉样本集合，用于评估大语言模型在识别幻觉方面的性能。
- **任务类别**: `Hallucination`, `Knowledge`, `Yes/No`
- **评估指标**: `accuracy`, `f1_score`, `precision`, `recall`, `yes_ratio`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `dialogue_samples`, `qa_samples`, `summarization_samples`

- **提示模板**: 
```text
{question}
```

---

### HarveyNER

[返回目录](#llm评测集)
- **数据集名称**: `harvey-ner`
- **数据集ID**: [extraordinarylab/harvey-ner](https://modelscope.cn/datasets/extraordinarylab/harvey-ner/summary)
- **数据集描述**:
  > HarveyNER 是一个在推文中标注了细粒度位置的数据集。该数据集具有独特挑战性，包含大量非正式描述中的复杂且较长的位置提及。
- **任务类别**: `Knowledge`, `NER`
- **评估指标**: `accuracy`, `f1_score`, `precision`, `recall`
- **需要LLM Judge**: 否
- **默认提示方式**: 5-shot
- **数据集子集**: `default`

- **提示模板**: 
```text
You are a named entity recognition system that identifies the following entity types:
{entities}

Process the provided text and mark all named entities with XML-style tags.

For example:
<person>John Smith</person> works at <organization>Google</organization> in <location>Mountain View</location>.

Available entity tags: {entity_list}

INSTRUCTIONS:
1. Wrap your entire response in <response>...</response> tags.
2. Inside these tags, include the original text with entity tags inserted.
3. Do not change the original text in any way (preserve spacing, punctuation, case, etc.).
4. Tag ALL entities you can identify using the exact tag names provided.
5. Do not include explanations, just the tagged text.
6. If entity spans overlap, choose the most specific entity type.
7. Ensure every opening tag has a matching closing tag.

Text to process:
{text}

```

---

### HealthBench

[返回目录](#llm评测集)
- **数据集名称**: `health_bench`
- **数据集ID**: [openai-mirror/healthbench](https://modelscope.cn/datasets/openai-mirror/healthbench/summary)
- **数据集描述**:
  > HealthBench：一个旨在更好衡量AI系统医疗能力的新基准。该基准与来自60个国家的262名医生合作构建，包含5,000个真实医疗对话，每个对话均配有医生定制的评分标准来评估模型回复。
- **任务类别**: `Knowledge`, `QA`
- **评估指标**: `accuracy`, `communication_quality`, `completeness`, `context_awareness`, `instruction_following`
- **需要LLM Judge**: 是
- **默认提示方式**: 0-shot
- **数据集子集**: `communication`, `complex_responses`, `context_seeking`, `emergency_referrals`, `global_health`, `health_data_tasks`, `hedging`

- **额外参数**: 
```json
{
    "version": "# File version, choose from ['Consensus', 'Hard', 'All'], default to Consensus"
}
```
- **提示模板**: 
```text
Answer the question:

{question}
```

---

### HellaSwag

[返回目录](#llm评测集)
- **数据集名称**: `hellaswag`
- **数据集ID**: [evalscope/hellaswag](https://modelscope.cn/datasets/evalscope/hellaswag/summary)
- **数据集描述**:
  > HellaSwag 是一个用于自然语言理解中常识推理的基准测试，包含多项选择题，要求模型从给定上下文中选出最合理的后续内容。
- **任务类别**: `Commonsense`, `Knowledge`, `MCQ`
- **评估指标**: `acc`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `default`

- **提示模板**: 
```text
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}.

{question}

{choices}
```

---

### Humanity's-Last-Exam

[返回目录](#llm评测集)
- **数据集名称**: `hle`
- **数据集ID**: [cais/hle](https://modelscope.cn/datasets/cais/hle/summary)
- **数据集描述**:
  > 人类最后的考试（HLE）是一个涵盖2500道题的语言模型基准，由AI安全中心和Scale AI联合创建。题目分为以下几大类：数学（41%）、物理（9%）、生物/医学（11%）、人文/社会科学（9%）、计算机科学/人工智能（10%）、工程（4%）、化学（7%）及其他（9%）。约14%的题目需理解文本和图像，即多模态能力。24%为选择题，其余为短答案精确匹配题。  
  > **如需评估不具备多模态能力的模型，请将 `extra_params["include_multi_modal"]` 设为 `False`。**
- **任务类别**: `Knowledge`, `QA`
- **评估指标**: `acc`
- **需要LLM Judge**: 是
- **默认提示方式**: 0-shot
- **数据集子集**: `Biology/Medicine`, `Chemistry`, `Computer Science/AI`, `Engineering`, `Humanities/Social Science`, `Math`, `Other`, `Physics`

- **额外参数**: 
```json
{
    "include_multi_modal": true
}
```
- **提示模板**: 
```text
{question}
```

---

### HumanEval

[返回目录](#llm评测集)
- **数据集名称**: `humaneval`
- **数据集ID**: [opencompass/humaneval](https://modelscope.cn/datasets/opencompass/humaneval/summary)
- **数据集描述**:
  > HumanEval 是一个基准测试，用于评估代码生成模型根据给定规范编写 Python 函数的能力。它包含一系列具有明确定义输入输出行为的编程任务。**默认情况下，代码在本地环境中执行。我们建议使用沙箱执行以安全地运行和评估生成的代码，请参考[文档](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/sandbox.html)了解详情。**
- **任务类别**: `Coding`
- **评估指标**: `Pass@1`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `openai_humaneval`

- **评测超时时间（秒）**: 4
- **提示模板**: 
```text
Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.
{question}
```

---

### IFEval

[返回目录](#llm评测集)
- **数据集名称**: `ifeval`
- **数据集ID**: [opencompass/ifeval](https://modelscope.cn/datasets/opencompass/ifeval/summary)
- **数据集描述**:
  > IFEval 是一个用于评估指令跟随型语言模型的基准，侧重于测试模型理解和响应各类提示的能力。它包含多样化的任务和指标，以全面评估模型性能。
- **任务类别**: `InstructionFollowing`
- **评估指标**: `inst_level_loose`, `inst_level_strict`, `prompt_level_loose`, `prompt_level_strict`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `default`


---

### IQuiz

[返回目录](#llm评测集)
- **数据集名称**: `iquiz`
- **数据集ID**: [AI-ModelScope/IQuiz](https://modelscope.cn/datasets/AI-ModelScope/IQuiz/summary)
- **数据集描述**:
  > IQuiz 是一个用于评估 AI 模型智商与情商的基准测试，包含多项选择题，要求模型选出正确答案并提供解释。
- **任务类别**: `Chinese`, `Knowledge`, `MCQ`
- **评估指标**: `acc`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `EQ`, `IQ`

- **提示模板**: 
```text
回答下面的单项选择题，请选出其中的正确答案。你的回答的最后一行应该是这样的格式："答案：LETTER"（不带引号），其中 LETTER 是 {letters} 中的一个。请在回答前进行一步步思考。

问题：{question}
选项：
{choices}

```

---

### Live-Code-Bench

[返回目录](#llm评测集)
- **数据集名称**: `live_code_bench`
- **数据集ID**: [AI-ModelScope/code_generation_lite](https://modelscope.cn/datasets/AI-ModelScope/code_generation_lite/summary)
- **数据集描述**:
  > Live Code Bench 是一个用于评估代码生成模型在真实编程任务中表现的基准测试，包含多种编程题目及测试用例，用以衡量模型生成正确且高效代码的能力。**默认情况下代码在本地环境中执行。我们建议使用沙箱执行以安全地运行和评估生成的代码，请参考[文档](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/sandbox.html)了解详情。**
- **任务类别**: `Coding`
- **评估指标**: `Pass@1`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `release_latest`

- **评测超时时间（秒）**: 6
- **额外参数**: 
```json
{
    "start_date": null,
    "end_date": null,
    "debug": false
}
```
- **提示模板**: 
```text
### Question:
{question_content}

{format_prompt} ### Answer: (use the provided format with backticks)


```

---

### LogiQA

[返回目录](#llm评测集)
- **数据集名称**: `logi_qa`
- **数据集ID**: [extraordinarylab/logiqa](https://modelscope.cn/datasets/extraordinarylab/logiqa/summary)
- **数据集描述**:
  > LogiQA 是一个源自专家编写的问题的数据集，用于测试人类的逻辑推理能力。
- **任务类别**: `MCQ`, `Reasoning`
- **评估指标**: `acc`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `default`

- **提示模板**: 
```text
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}.

{question}

{choices}
```

---

### MaritimeBench

[返回目录](#llm评测集)
- **数据集名称**: `maritime_bench`
- **数据集ID**: [HiDolphin/MaritimeBench](https://modelscope.cn/datasets/HiDolphin/MaritimeBench/summary)
- **数据集描述**:
  > MaritimeBench 是一个用于评估AI模型在 maritime 相关选择题上表现的基准，包含需要模型从给定选项中选出正确答案的 maritime 知识问题。
- **任务类别**: `Chinese`, `Knowledge`, `MCQ`
- **评估指标**: `acc`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `default`

- **提示模板**: 
```text
请回答单选题。要求只输出选项，不输出解释，将选项放在[]里，直接输出答案。示例：

题目：在船舶主推进动力装置中，传动轴系在运转中承受以下复杂的应力和负荷，但不包括______。
选项：
A. 电磁力
B. 压拉应力
C. 弯曲应力
D. 扭应力
答：[A]
 当前题目
 {question}
选项：
{choices}
```

---

### MATH-500

[返回目录](#llm评测集)
- **数据集名称**: `math_500`
- **数据集ID**: [AI-ModelScope/MATH-500](https://modelscope.cn/datasets/AI-ModelScope/MATH-500/summary)
- **数据集描述**:
  > MATH-500 是一个用于评估AI模型数学推理能力的基准，包含500道涵盖五个难度级别的多样化数学题，旨在通过生成逐步解题过程并给出正确最终答案来测试模型解决复杂数学问题的能力。
- **任务类别**: `Math`, `Reasoning`
- **评估指标**: `{'acc': {'numeric': True}}`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `Level 1`, `Level 2`, `Level 3`, `Level 4`, `Level 5`

- **提示模板**: 
```text
{question}
Please reason step by step, and put your final answer within \boxed{{}}.
```

---

### MathQA

[返回目录](#llm评测集)
- **数据集名称**: `math_qa`
- **数据集ID**: [extraordinarylab/math-qa](https://modelscope.cn/datasets/extraordinarylab/math-qa/summary)
- **数据集描述**:
  > MathQA 数据集通过使用一种新的表示语言，对 AQuA-RAT 数据集进行标注，生成完整的操作程序而构建。
- **任务类别**: `MCQ`, `Math`, `Reasoning`
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

### Minerva-Math

[返回目录](#llm评测集)
- **数据集名称**: `minerva_math`
- **数据集ID**: [knoveleng/Minerva-Math](https://modelscope.cn/datasets/knoveleng/Minerva-Math/summary)
- **数据集描述**:
  > Minerva-math 是一个用于评估大语言模型数学与定量推理能力的基准，包含 **272 道题目**，主要来自 **MIT OpenCourseWare** 课程，涵盖固态化学、天文学、微分方程和狭义相对论等 **大学及研究生水平** 的高级 STEM 学科。
- **任务类别**: `Math`, `Reasoning`
- **评估指标**: `{'acc': {'numeric': True}}`
- **需要LLM Judge**: 是
- **默认提示方式**: 0-shot
- **数据集子集**: `default`

- **提示模板**: 
```text
{question}
Please reason step by step, and put your final answer within \boxed{{}}.
```

---

### MIT-Movie-Trivia

[返回目录](#llm评测集)
- **数据集名称**: `mit-movie-trivia`
- **数据集ID**: [extraordinarylab/mit-movie-trivia](https://modelscope.cn/datasets/extraordinarylab/mit-movie-trivia/summary)
- **数据集描述**:
  > MIT-Movie-Trivia 数据集最初用于槽位填充，通过忽略某些槽位类型（如类型、评分）并将其他类型合并（如将导演和演员合并为“人物”，歌曲和电影标题合并为“标题”），以保持所有数据集中命名实体类型的一致性。
- **任务类别**: `Knowledge`, `NER`
- **评估指标**: `accuracy`, `f1_score`, `precision`, `recall`
- **需要LLM Judge**: 否
- **默认提示方式**: 5-shot
- **数据集子集**: `default`

- **提示模板**: 
```text
You are a named entity recognition system that identifies the following entity types:
{entities}

Process the provided text and mark all named entities with XML-style tags.

For example:
<person>John Smith</person> works at <organization>Google</organization> in <location>Mountain View</location>.

Available entity tags: {entity_list}

INSTRUCTIONS:
1. Wrap your entire response in <response>...</response> tags.
2. Inside these tags, include the original text with entity tags inserted.
3. Do not change the original text in any way (preserve spacing, punctuation, case, etc.).
4. Tag ALL entities you can identify using the exact tag names provided.
5. Do not include explanations, just the tagged text.
6. If entity spans overlap, choose the most specific entity type.
7. Ensure every opening tag has a matching closing tag.

Text to process:
{text}

```

---

### MIT-Restaurant

[返回目录](#llm评测集)
- **数据集名称**: `mit-restaurant`
- **数据集ID**: [extraordinarylab/mit-restaurant](https://modelscope.cn/datasets/extraordinarylab/mit-restaurant/summary)
- **数据集描述**:
  > MIT-Restaurant 数据集是一个专门用于训练和测试自然语言处理（NLP）模型的餐厅评论文本集合，尤其适用于命名实体识别（NER）。该数据集包含来自真实评论的句子及其对应的 BIO 格式标签。
- **任务类别**: `Knowledge`, `NER`
- **评估指标**: `accuracy`, `f1_score`, `precision`, `recall`
- **需要LLM Judge**: 否
- **默认提示方式**: 5-shot
- **数据集子集**: `default`

- **提示模板**: 
```text
You are a named entity recognition system that identifies the following entity types:
{entities}

Process the provided text and mark all named entities with XML-style tags.

For example:
<person>John Smith</person> works at <organization>Google</organization> in <location>Mountain View</location>.

Available entity tags: {entity_list}

INSTRUCTIONS:
1. Wrap your entire response in <response>...</response> tags.
2. Inside these tags, include the original text with entity tags inserted.
3. Do not change the original text in any way (preserve spacing, punctuation, case, etc.).
4. Tag ALL entities you can identify using the exact tag names provided.
5. Do not include explanations, just the tagged text.
6. If entity spans overlap, choose the most specific entity type.
7. Ensure every opening tag has a matching closing tag.

Text to process:
{text}

```

---

### MMLU

[返回目录](#llm评测集)
- **数据集名称**: `mmlu`
- **数据集ID**: [cais/mmlu](https://modelscope.cn/datasets/cais/mmlu/summary)
- **数据集描述**:
  > MMLU（大规模多任务语言理解）基准是一个综合评估套件，旨在评估语言模型在广泛主题和任务中的表现。它涵盖历史、科学、数学等多个领域的多项选择题，能够有效衡量模型的理解和推理能力。
- **任务类别**: `Knowledge`, `MCQ`
- **评估指标**: `acc`
- **需要LLM Judge**: 否
- **默认提示方式**: 5-shot
- **数据集子集**: `abstract_algebra`, `anatomy`, `astronomy`, `business_ethics`, `clinical_knowledge`, `college_biology`, `college_chemistry`, `college_computer_science`, `college_mathematics`, `college_medicine`, `college_physics`, `computer_security`, `conceptual_physics`, `econometrics`, `electrical_engineering`, `elementary_mathematics`, `formal_logic`, `global_facts`, `high_school_biology`, `high_school_chemistry`, `high_school_computer_science`, `high_school_european_history`, `high_school_geography`, `high_school_government_and_politics`, `high_school_macroeconomics`, `high_school_mathematics`, `high_school_microeconomics`, `high_school_physics`, `high_school_psychology`, `high_school_statistics`, `high_school_us_history`, `high_school_world_history`, `human_aging`, `human_sexuality`, `international_law`, `jurisprudence`, `logical_fallacies`, `machine_learning`, `management`, `marketing`, `medical_genetics`, `miscellaneous`, `moral_disputes`, `moral_scenarios`, `nutrition`, `philosophy`, `prehistory`, `professional_accounting`, `professional_law`, `professional_medicine`, `professional_psychology`, `public_relations`, `security_studies`, `sociology`, `us_foreign_policy`, `virology`, `world_religions`

- **提示模板**: 
```text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}. Think step by step before answering.

{question}

{choices}
```

---

### MMLU-Pro

[返回目录](#llm评测集)
- **数据集名称**: `mmlu_pro`
- **数据集ID**: [modelscope/MMLU-Pro](https://modelscope.cn/datasets/modelscope/MMLU-Pro/summary)
- **数据集描述**:
  > MMLU-Pro 是一个用于评估语言模型在多个学科选择题上表现的基准，涵盖不同领域的问题，要求模型从给定选项中选出正确答案。
- **任务类别**: `Knowledge`, `MCQ`
- **评估指标**: `acc`
- **需要LLM Judge**: 否
- **默认提示方式**: 5-shot
- **数据集子集**: `biology`, `business`, `chemistry`, `computer science`, `economics`, `engineering`, `health`, `history`, `law`, `math`, `other`, `philosophy`, `physics`, `psychology`

- **提示模板**: 
```text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}. Think step by step before answering.

Question:
{question}
Options:
{choices}

```

---

### MMLU-Redux

[返回目录](#llm评测集)
- **数据集名称**: `mmlu_redux`
- **数据集ID**: [AI-ModelScope/mmlu-redux-2.0](https://modelscope.cn/datasets/AI-ModelScope/mmlu-redux-2.0/summary)
- **数据集描述**:
  > MMLU-Redux 是一个评估语言模型在多个学科选择题上表现的基准，涵盖不同领域的问题，模型需从给定选项中选出正确答案，且错误选项已被修正。
- **任务类别**: `Knowledge`, `MCQ`
- **评估指标**: `{'acc': {'allow_inclusion': True}}`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `abstract_algebra`, `anatomy`, `astronomy`, `business_ethics`, `clinical_knowledge`, `college_biology`, `college_chemistry`, `college_computer_science`, `college_mathematics`, `college_medicine`, `college_physics`, `computer_security`, `conceptual_physics`, `econometrics`, `electrical_engineering`, `elementary_mathematics`, `formal_logic`, `global_facts`, `high_school_biology`, `high_school_chemistry`, `high_school_computer_science`, `high_school_european_history`, `high_school_geography`, `high_school_government_and_politics`, `high_school_macroeconomics`, `high_school_mathematics`, `high_school_microeconomics`, `high_school_physics`, `high_school_psychology`, `high_school_statistics`, `high_school_us_history`, `high_school_world_history`, `human_aging`, `human_sexuality`, `international_law`, `jurisprudence`, `logical_fallacies`, `machine_learning`, `management`, `marketing`, `medical_genetics`, `miscellaneous`, `moral_disputes`, `moral_scenarios`, `nutrition`, `philosophy`, `prehistory`, `professional_accounting`, `professional_law`, `professional_medicine`, `professional_psychology`, `public_relations`, `security_studies`, `sociology`, `us_foreign_policy`, `virology`, `world_religions`

- **提示模板**: 
```text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}. Think step by step before answering.

{question}

{choices}
```

---

### MRI-MCQA

[返回目录](#llm评测集)
- **数据集名称**: `mri_mcqa`
- **数据集ID**: [extraordinarylab/mri-mcqa](https://modelscope.cn/datasets/extraordinarylab/mri-mcqa/summary)
- **数据集描述**:
  > MRI-MCQA 是一个包含磁共振成像（MRI）相关选择题的基准测试。
- **任务类别**: `Knowledge`, `MCQ`
- **评估指标**: `acc`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `default`

- **提示模板**: 
```text
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}.

{question}

{choices}
```

---

### Multi-IF

[返回目录](#llm评测集)
- **数据集名称**: `multi_if`
- **数据集ID**: [facebook/Multi-IF](https://modelscope.cn/datasets/facebook/Multi-IF/summary)
- **数据集描述**:
  > Multi-IF 是一个用于评估大语言模型在多语言环境下多轮指令遵循能力的基准。
- **任务类别**: `InstructionFollowing`, `MultiLingual`, `MultiTurn`
- **评估指标**: `inst_level_loose`, `inst_level_strict`, `prompt_level_loose`, `prompt_level_strict`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `Chinese`, `English`, `French`, `German`, `Hindi`, `Italian`, `Portuguese`, `Russian`, `Spanish`, `Thai`, `Vietnamese`

- **额外参数**: 
```json
{
    "max_turns": 3
}
```

---

### MuSR

[返回目录](#llm评测集)
- **数据集名称**: `musr`
- **数据集ID**: [AI-ModelScope/MuSR](https://modelscope.cn/datasets/AI-ModelScope/MuSR/summary)
- **数据集描述**:
  > MuSR 是一个用于评估 AI 模型在谋杀谜案、物体位置和团队分配等选择题上表现的基准。
- **任务类别**: `MCQ`, `Reasoning`
- **评估指标**: `acc`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `murder_mysteries`, `object_placements`, `team_allocation`

- **提示模板**: 
```text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}. Think step by step before answering.

{question}

{choices}
```

---

### Needle-in-a-Haystack

[返回目录](#llm评测集)
- **数据集名称**: `needle_haystack`
- **数据集ID**: [AI-ModelScope/Needle-in-a-Haystack-Corpus](https://modelscope.cn/datasets/AI-ModelScope/Needle-in-a-Haystack-Corpus/summary)
- **数据集描述**:
  > “大海捞针”是一个专注于信息检索任务的基准，要求模型在大量文本中找出特定信息。[使用示例](https://evalscope.readthedocs.io/zh-cn/latest/third_party/needle_haystack.html)
- **任务类别**: `LongContext`, `Retrieval`
- **评估指标**: `acc`
- **需要LLM Judge**: 是
- **默认提示方式**: 0-shot
- **数据集子集**: `chinese`, `english`

- **额外参数**: 
```json
{
    "retrieval_question": "What is the best thing to do in San Francisco?",
    "needles": [
        "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n"
    ],
    "context_lengths_min": 1000,
    "context_lengths_max": 32000,
    "context_lengths_num_intervals": 10,
    "document_depth_percent_min": 0,
    "document_depth_percent_max": 100,
    "document_depth_percent_intervals": 10,
    "tokenizer_path": "Qwen/Qwen3-0.6B",
    "show_score": false
}
```
- **系统提示词**: 
```text
You are a helpful AI bot that answers questions for a user. Keep your response short and direct
```
- **提示模板**: 
```text
Please read the following text and answer the question below.

<text>
{context}
</text>

<question>
{question}
</question>

Don't give information outside the document or repeat your findings.
```

---

### OntoNotes5

[返回目录](#llm评测集)
- **数据集名称**: `ontonotes5`
- **数据集ID**: [extraordinarylab/ontonotes5](https://modelscope.cn/datasets/extraordinarylab/ontonotes5/summary)
- **数据集描述**:
  > OntoNotes 5.0 是一个大型多语言语料库，包含英语、中文和阿拉伯语的多种体裁文本，如新闻、博客和广播对话。该语料库标注了丰富的语言信息层次，包括句法、谓词-论元结构、词义、命名实体和共指关系，支持自然语言处理的研究与开发。
- **任务类别**: `Knowledge`, `NER`
- **评估指标**: `accuracy`, `f1_score`, `precision`, `recall`
- **需要LLM Judge**: 否
- **默认提示方式**: 5-shot
- **数据集子集**: `default`

- **提示模板**: 
```text
You are a named entity recognition system that identifies the following entity types:
{entities}

Process the provided text and mark all named entities with XML-style tags.

For example:
<person>John Smith</person> works at <organization>Google</organization> in <location>Mountain View</location>.

Available entity tags: {entity_list}

INSTRUCTIONS:
1. Wrap your entire response in <response>...</response> tags.
2. Inside these tags, include the original text with entity tags inserted.
3. Do not change the original text in any way (preserve spacing, punctuation, case, etc.).
4. Tag ALL entities you can identify using the exact tag names provided.
5. Do not include explanations, just the tagged text.
6. If entity spans overlap, choose the most specific entity type.
7. Ensure every opening tag has a matching closing tag.

Text to process:
{text}

```

---

### PIQA

[返回目录](#llm评测集)
- **数据集名称**: `piqa`
- **数据集ID**: [extraordinarylab/piqa](https://modelscope.cn/datasets/extraordinarylab/piqa/summary)
- **数据集描述**:
  > PIQA 旨在解决自然语言中物理常识推理的难题。
- **任务类别**: `Commonsense`, `MCQ`, `Reasoning`
- **评估指标**: `acc`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `default`

- **提示模板**: 
```text
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}.

{question}

{choices}
```

---

### PolyMath

[返回目录](#llm评测集)
- **数据集名称**: `poly_math`
- **数据集ID**: [evalscope/PolyMath](https://modelscope.cn/datasets/evalscope/PolyMath/summary)
- **数据集描述**:
  > PolyMath 是一个涵盖 18 种语言、4 个由易到难难度级别的多语言数学推理基准，包含 9,000 个高质量问题样本。该基准确保了难度全面性、语言多样性和高质量翻译，是推理型大语言模型时代极具区分度的多语言数学评测基准。
- **任务类别**: `Math`, `MultiLingual`, `Reasoning`
- **评估指标**: `{'acc': {'numeric': True}}`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `ar`, `bn`, `de`, `en`, `es`, `fr`, `id`, `it`, `ja`, `ko`, `ms`, `pt`, `ru`, `sw`, `te`, `th`, `vi`, `zh`

- **提示模板**: 
```text
{question}
```

---

### ProcessBench

[返回目录](#llm评测集)
- **数据集名称**: `process_bench`
- **数据集ID**: [Qwen/ProcessBench](https://modelscope.cn/datasets/Qwen/ProcessBench/summary)
- **数据集描述**:
  > ProcessBench 是一个用于评估AI模型数学推理能力的基准测试，包含 GSM8K、Math、OlympiadBench 和 OmniMath 等多个子集，每个子集均提供需逐步推理才能得出正确答案的问题。
- **任务类别**: `Math`, `Reasoning`
- **评估指标**: `correct_acc`, `error_acc`, `simple_f1_score`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `gsm8k`, `math`, `olympiadbench`, `omnimath`

- **提示模板**: 
```text
CThe following is a math problem and a solution (split into paragraphs, enclosed with tags and indexed from 0):

[Math Problem]

{problem}

[Solution]

{tagged_response}

Your task is to review and critique the solution paragraph by paragraph. Once you identify an error in a paragraph, return the index of the paragraph where the earliest error occurs. Otherwise, return the index of -1 (which typically denotes "not found").

Please put your final answer (i.e., the index) in oxed{{}}.

```

---

### PubMedQA

[返回目录](#llm评测集)
- **数据集名称**: `pubmedqa`
- **数据集ID**: [extraordinarylab/pubmed-qa](https://modelscope.cn/datasets/extraordinarylab/pubmed-qa/summary)
- **数据集描述**:
  > PubMedQA 通过推理生物医学研究文本回答多项选择题。
- **任务类别**: `Knowledge`, `Yes/No`
- **评估指标**: `accuracy`, `f1_score`, `maybe_ratio`, `precision`, `recall`, `yes_ratio`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `default`

- **提示模板**: 
```text
{question}
Please answer YES or NO or MAYBE without an explanation.
```

---

### QASC

[返回目录](#llm评测集)
- **数据集名称**: `qasc`
- **数据集ID**: [extraordinarylab/qasc](https://modelscope.cn/datasets/extraordinarylab/qasc/summary)
- **数据集描述**:
  > QASC 是一个专注于句子组合的问答数据集，包含 9,980 道八选一的多项选择题，涉及小学科学内容。
- **任务类别**: `Knowledge`, `MCQ`
- **评估指标**: `acc`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `default`

- **提示模板**: 
```text
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}.

{question}

{choices}
```

---

### RACE

[返回目录](#llm评测集)
- **数据集名称**: `race`
- **数据集ID**: [evalscope/race](https://modelscope.cn/datasets/evalscope/race/summary)
- **数据集描述**:
  > RACE 是一个用于测试神经网络模型阅读理解与推理能力的基准，基于中国初高中考试题目构建。
- **任务类别**: `MCQ`, `Reasoning`
- **评估指标**: `acc`
- **需要LLM Judge**: 否
- **默认提示方式**: 3-shot
- **数据集子集**: `high`, `middle`

- **提示模板**: 
```text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}. Think step by step before answering.

{question}

{choices}
```

---

### SimpleQA

[返回目录](#llm评测集)
- **数据集名称**: `simple_qa`
- **数据集ID**: [AI-ModelScope/SimpleQA](https://modelscope.cn/datasets/AI-ModelScope/SimpleQA/summary)
- **数据集描述**:
  > SimpleQA 是一个用于评估语言模型在简单问答任务上性能的基准，包含一系列需要基本推理和理解能力的直接问题。
- **任务类别**: `Knowledge`, `QA`
- **评估指标**: `is_correct`, `is_incorrect`, `is_not_attempted`
- **需要LLM Judge**: 是
- **默认提示方式**: 0-shot
- **数据集子集**: `default`

- **提示模板**: 
```text
Answer the question:

{question}
```

---

### SIQA

[返回目录](#llm评测集)
- **数据集名称**: `siqa`
- **数据集ID**: [extraordinarylab/siqa](https://modelscope.cn/datasets/extraordinarylab/siqa/summary)
- **数据集描述**:
  > 社交互动问答（SIQA）是一个用于测试社交常识智能的问答基准。与许多关注物理或分类知识的先前基准不同，Social IQa 侧重于推理人们的行为及其社会影响。
- **任务类别**: `Commonsense`, `MCQ`, `Reasoning`
- **评估指标**: `acc`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `default`

- **提示模板**: 
```text
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}.

{question}

{choices}
```

---

### SuperGPQA

[返回目录](#llm评测集)
- **数据集名称**: `super_gpqa`
- **数据集ID**: [m-a-p/SuperGPQA](https://modelscope.cn/datasets/m-a-p/SuperGPQA/summary)
- **数据集描述**:
  > SuperGPQA 是一个大规模多项选择题问答数据集，旨在评估模型在不同领域的泛化能力。它包含来自 50 多个领域的 10 万多个问题，每个问题有 10 个选项。
- **任务类别**: `Knowledge`, `MCQ`
- **评估指标**: `acc`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `Aeronautical and Astronautical Science and Technology`, `Agricultural Engineering`, `Animal Husbandry`, `Applied Economics`, `Aquaculture`, `Architecture`, `Art Studies`, `Astronomy`, `Atmospheric Science`, `Basic Medicine`, `Biology`, `Business Administration`, `Chemical Engineering and Technology`, `Chemistry`, `Civil Engineering`, `Clinical Medicine`, `Computer Science and Technology`, `Control Science and Engineering`, `Crop Science`, `Education`, `Electrical Engineering`, `Electronic Science and Technology`, `Environmental Science and Engineering`, `Food Science and Engineering`, `Forestry Engineering`, `Forestry`, `Geography`, `Geological Resources and Geological Engineering`, `Geology`, `Geophysics`, `History`, `Hydraulic Engineering`, `Information and Communication Engineering`, `Instrument Science and Technology`, `Journalism and Communication`, `Language and Literature`, `Law`, `Library, Information and Archival Management`, `Management Science and Engineering`, `Materials Science and Engineering`, `Mathematics`, `Mechanical Engineering`, `Mechanics`, `Metallurgical Engineering`, `Military Science`, `Mining Engineering`, `Musicology`, `Naval Architecture and Ocean Engineering`, `Nuclear Science and Technology`, `Oceanography`, `Optical Engineering`, `Petroleum and Natural Gas Engineering`, `Pharmacy`, `Philosophy`, `Physical Education`, `Physical Oceanography`, `Physics`, `Political Science`, `Power Engineering and Engineering Thermophysics`, `Psychology`, `Public Administration`, `Public Health and Preventive Medicine`, `Sociology`, `Stomatology`, `Surveying and Mapping Science and Technology`, `Systems Science`, `Textile Science and Engineering`, `Theoretical Economics`, `Traditional Chinese Medicine`, `Transportation Engineering`, `Veterinary Medicine`, `Weapon Science and Technology`

- **提示模板**: 
```text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}. Think step by step before answering.

{question}

{choices}
```

---

### τ-bench

[返回目录](#llm评测集)
- **数据集名称**: `tau_bench`
- **数据集ID**: [tau-bench](https://github.com/sierra-research/tau-bench)
- **数据集描述**:
  > 一个模拟用户（由语言模型模拟）与具备特定领域API工具和策略指南的语言代理之间动态对话的基准测试。评估前请先通过 `pip install git+https://github.com/sierra-research/tau-bench` 安装并设置用户模型。[使用示例](https://evalscope.readthedocs.io/zh-cn/latest/third_party/tau_bench.html)
- **任务类别**: `FunctionCalling`, `Reasoning`
- **评估指标**: `Pass^1`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `airline`, `retail`

- **额外参数**: 
```json
{
    "user_model": "qwen-plus",
    "api_key": "EMPTY",
    "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "generation_config": {
        "temperature": 0.0,
        "max_tokens": 4096
    }
}
```

---

### ToolBench-Static

[返回目录](#llm评测集)
- **数据集名称**: `tool_bench`
- **数据集ID**: [AI-ModelScope/ToolBench-Static](https://modelscope.cn/datasets/AI-ModelScope/ToolBench-Static/summary)
- **数据集描述**:
  > ToolBench 是一个用于评估 AI 模型工具使用能力的基准，包含多个子集（如领域内和领域外），每个子集均提供需逐步推理才能得出正确答案的问题。[使用示例](https://evalscope.readthedocs.io/zh-cn/latest/third_party/toolbench.html)
- **任务类别**: `FunctionCalling`, `Reasoning`
- **评估指标**: `Act.EM`, `F1`, `HalluRate`, `Plan.EM`, `Rouge-L`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `in_domain`, `out_of_domain`


---

### TriviaQA

[返回目录](#llm评测集)
- **数据集名称**: `trivia_qa`
- **数据集ID**: [evalscope/trivia_qa](https://modelscope.cn/datasets/evalscope/trivia_qa/summary)
- **数据集描述**:
  > TriviaQA 是一个大规模阅读理解数据集，包含从 trivia 网站收集的问答对。该数据集中的问题可能有多个正确答案，适用于评估模型基于上下文理解和生成答案的能力。
- **任务类别**: `QA`, `ReadingComprehension`
- **评估指标**: `{'acc': {'allow_inclusion': True}}`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `rc.wikipedia`

- **提示模板**: 
```text
Read the content and answer the following question.

Content: {content}

Question: {question}

Keep your The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

```

---

### TruthfulQA

[返回目录](#llm评测集)
- **数据集名称**: `truthful_qa`
- **数据集ID**: [evalscope/truthful_qa](https://modelscope.cn/datasets/evalscope/truthful_qa/summary)
- **数据集描述**:
  > TruthfulQA 是一个用于评估 AI 模型真实准确回答问题能力的基准，包含多项选择任务，侧重考察模型对事实信息的理解。
- **任务类别**: `Knowledge`
- **评估指标**: `multi_choice_acc`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `multiple_choice`

- **额外参数**: 
```json
{
    "multiple_correct": false
}
```
- **提示模板**: 
```text
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}.

{question}

{choices}
```

---

### Winogrande

[返回目录](#llm评测集)
- **数据集名称**: `winogrande`
- **数据集ID**: [AI-ModelScope/winogrande_val](https://modelscope.cn/datasets/AI-ModelScope/winogrande_val/summary)
- **数据集描述**:
  > Winogrande 是一个用于评估 AI 模型在常识推理任务上表现的基准，专门用于测试模型解决句子中歧义代词的能力。
- **任务类别**: `MCQ`, `Reasoning`
- **评估指标**: `acc`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `default`

- **提示模板**: 
```text
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}.

{question}

{choices}
```

---

### WNUT2017

[返回目录](#llm评测集)
- **数据集名称**: `wnut2017`
- **数据集ID**: [extraordinarylab/wnut2017](https://modelscope.cn/datasets/extraordinarylab/wnut2017/summary)
- **数据集描述**:
  > WNUT2017 数据集包含来自 Twitter 和 YouTube 等社交媒体平台的用户生成文本，专为命名实体识别任务设计。
- **任务类别**: `Knowledge`, `NER`
- **评估指标**: `accuracy`, `f1_score`, `precision`, `recall`
- **需要LLM Judge**: 否
- **默认提示方式**: 5-shot
- **数据集子集**: `default`

- **提示模板**: 
```text
You are a named entity recognition system that identifies the following entity types:
{entities}

Process the provided text and mark all named entities with XML-style tags.

For example:
<person>John Smith</person> works at <organization>Google</organization> in <location>Mountain View</location>.

Available entity tags: {entity_list}

INSTRUCTIONS:
1. Wrap your entire response in <response>...</response> tags.
2. Inside these tags, include the original text with entity tags inserted.
3. Do not change the original text in any way (preserve spacing, punctuation, case, etc.).
4. Tag ALL entities you can identify using the exact tag names provided.
5. Do not include explanations, just the tagged text.
6. If entity spans overlap, choose the most specific entity type.
7. Ensure every opening tag has a matching closing tag.

Text to process:
{text}

```
