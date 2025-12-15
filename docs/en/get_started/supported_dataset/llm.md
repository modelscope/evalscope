# LLM Benchmarks

Below is the list of supported LLM benchmarks. Click on a benchmark name to jump to details.

| Benchmark Name | Pretty Name | Task Categories |
|------------|----------|----------|
| `aa_lcr` | [AA-LCR](#aa-lcr) | `Knowledge`, `LongContext`, `Reasoning` |
| `aime24` | [AIME-2024](#aime-2024) | `Math`, `Reasoning` |
| `aime25` | [AIME-2025](#aime-2025) | `Math`, `Reasoning` |
| `alpaca_eval` | [AlpacaEval2.0](#alpacaeval20) | `Arena`, `InstructionFollowing` |
| `amc` | [AMC](#amc) | `Math`, `Reasoning` |
| `arc` | [ARC](#arc) | `MCQ`, `Reasoning` |
| `arena_hard` | [ArenaHard](#arenahard) | `Arena`, `InstructionFollowing` |
| `bbh` | [BBH](#bbh) | `Reasoning` |
| `biomix_qa` | [BioMixQA](#biomixqa) | `Knowledge`, `MCQ`, `Medical` |
| `broad_twitter_corpus` | [BroadTwitterCorpus](#broadtwittercorpus) | `Knowledge`, `NER` |
| `ceval` | [C-Eval](#c-eval) | `Chinese`, `Knowledge`, `MCQ` |
| `chinese_simpleqa` | [Chinese-SimpleQA](#chinese-simpleqa) | `Chinese`, `Knowledge`, `QA` |
| `cmmlu` | [C-MMLU](#c-mmlu) | `Chinese`, `Knowledge`, `MCQ` |
| `coin_flip` | [CoinFlip](#coinflip) | `Reasoning`, `Yes/No` |
| `commonsense_qa` | [CommonsenseQA](#commonsenseqa) | `Commonsense`, `MCQ`, `Reasoning` |
| `competition_math` | [Competition-MATH](#competition-math) | `Math`, `Reasoning` |
| `conll2003` | [CoNLL2003](#conll2003) | `Knowledge`, `NER` |
| `copious` | [Copious](#copious) | `Knowledge`, `NER` |
| `cross_ner` | [CrossNER](#crossner) | `Knowledge`, `NER` |
| `data_collection` | [Data-Collection](#data-collection) | `Custom` |
| `docmath` | [DocMath](#docmath) | `LongContext`, `Math`, `Reasoning` |
| `drivel_binary` | [DrivelologyBinaryClassification](#drivelologybinaryclassification) | `Yes/No` |
| `drivel_multilabel` | [DrivelologyMultilabelClassification](#drivelologymultilabelclassification) | `MCQ` |
| `drivel_selection` | [DrivelologyNarrativeSelection](#drivelologynarrativeselection) | `MCQ` |
| `drivel_writing` | [DrivelologyNarrativeWriting](#drivelologynarrativewriting) | `Knowledge`, `Reasoning` |
| `drop` | [DROP](#drop) | `Reasoning` |
| `eq_bench` | [EQ-Bench](#eq-bench) | `InstructionFollowing` |
| `frames` | [FRAMES](#frames) | `LongContext`, `Reasoning` |
| `general_arena` | [GeneralArena](#generalarena) | `Arena`, `Custom` |
| `general_mcq` | [General-MCQ](#general-mcq) | `Custom`, `MCQ` |
| `general_qa` | [General-QA](#general-qa) | `Custom`, `QA` |
| `genia_ner` | [GeniaNER](#genianer) | `Knowledge`, `NER` |
| `gpqa_diamond` | [GPQA-Diamond](#gpqa-diamond) | `Knowledge`, `MCQ` |
| `gsm8k` | [GSM8K](#gsm8k) | `Math`, `Reasoning` |
| `halueval` | [HaluEval](#halueval) | `Hallucination`, `Knowledge`, `Yes/No` |
| `harvey_ner` | [HarveyNER](#harveyner) | `Knowledge`, `NER` |
| `health_bench` | [HealthBench](#healthbench) | `Knowledge`, `Medical`, `QA` |
| `hellaswag` | [HellaSwag](#hellaswag) | `Commonsense`, `Knowledge`, `MCQ` |
| `hle` | [Humanity's-Last-Exam](#humanitys-last-exam) | `Knowledge`, `QA` |
| `humaneval` | [HumanEval](#humaneval) | `Coding` |
| `ifbench` | [IFBench](#ifbench) | `InstructionFollowing` |
| `ifeval` | [IFEval](#ifeval) | `InstructionFollowing` |
| `iquiz` | [IQuiz](#iquiz) | `Chinese`, `Knowledge`, `MCQ` |
| `live_code_bench` | [Live-Code-Bench](#live-code-bench) | `Coding` |
| `logi_qa` | [LogiQA](#logiqa) | `MCQ`, `Reasoning` |
| `maritime_bench` | [MaritimeBench](#maritimebench) | `Chinese`, `Knowledge`, `MCQ` |
| `math_500` | [MATH-500](#math-500) | `Math`, `Reasoning` |
| `math_qa` | [MathQA](#mathqa) | `MCQ`, `Math`, `Reasoning` |
| `mbpp` | [MBPP](#mbpp) | `Coding` |
| `med_mcqa` | [Med-MCQA](#med-mcqa) | `Knowledge`, `MCQ` |
| `mgsm` | [MGSM](#mgsm) | `Math`, `MultiLingual`, `Reasoning` |
| `minerva_math` | [Minerva-Math](#minerva-math) | `Math`, `Reasoning` |
| `mit_movie_trivia` | [MIT-Movie-Trivia](#mit-movie-trivia) | `Knowledge`, `NER` |
| `mit_restaurant` | [MIT-Restaurant](#mit-restaurant) | `Knowledge`, `NER` |
| `mmlu` | [MMLU](#mmlu) | `Knowledge`, `MCQ` |
| `mmlu_pro` | [MMLU-Pro](#mmlu-pro) | `Knowledge`, `MCQ` |
| `mmlu_redux` | [MMLU-Redux](#mmlu-redux) | `Knowledge`, `MCQ` |
| `mri_mcqa` | [MRI-MCQA](#mri-mcqa) | `Knowledge`, `MCQ`, `Medical` |
| `multi_if` | [Multi-IF](#multi-if) | `InstructionFollowing`, `MultiLingual`, `MultiTurn` |
| `multiple_humaneval` | [MultiPL-E HumanEval](#multipl-e-humaneval) | `Coding` |
| `multiple_mbpp` | [MultiPL-E MBPP](#multipl-e-mbpp) | `Coding` |
| `music_trivia` | [MusicTrivia](#musictrivia) | `Knowledge`, `MCQ` |
| `musr` | [MuSR](#musr) | `MCQ`, `Reasoning` |
| `needle_haystack` | [Needle-in-a-Haystack](#needle-in-a-haystack) | `LongContext`, `Retrieval` |
| `ontonotes5` | [OntoNotes5](#ontonotes5) | `Knowledge`, `NER` |
| `openai_mrcr` | [OpenAI MRCR](#openai-mrcr) | `LongContext`, `Retrieval` |
| `piqa` | [PIQA](#piqa) | `Commonsense`, `MCQ`, `Reasoning` |
| `poly_math` | [PolyMath](#polymath) | `Math`, `MultiLingual`, `Reasoning` |
| `process_bench` | [ProcessBench](#processbench) | `Math`, `Reasoning` |
| `pubmedqa` | [PubMedQA](#pubmedqa) | `Knowledge`, `Yes/No` |
| `qasc` | [QASC](#qasc) | `Knowledge`, `MCQ` |
| `race` | [RACE](#race) | `MCQ`, `Reasoning` |
| `scicode` | [SciCode](#scicode) | `Coding` |
| `sciq` | [SciQ](#sciq) | `Knowledge`, `MCQ`, `ReadingComprehension` |
| `simple_qa` | [SimpleQA](#simpleqa) | `Knowledge`, `QA` |
| `siqa` | [SIQA](#siqa) | `Commonsense`, `MCQ`, `Reasoning` |
| `super_gpqa` | [SuperGPQA](#supergpqa) | `Knowledge`, `MCQ` |
| `swe_bench_lite` | [SWE-bench_Lite](#swe-bench_lite) | `Coding` |
| `swe_bench_verified` | [SWE-bench_Verified](#swe-bench_verified) | `Coding` |
| `swe_bench_verified_mini` | [SWE-bench_Verified_mini](#swe-bench_verified_mini) | `Coding` |
| `trivia_qa` | [TriviaQA](#triviaqa) | `QA`, `ReadingComprehension` |
| `truthful_qa` | [TruthfulQA](#truthfulqa) | `Knowledge` |
| `winogrande` | [Winogrande](#winogrande) | `MCQ`, `Reasoning` |
| `wmt24pp` | [WMT2024++](#wmt2024) | `MachineTranslation`, `MultiLingual` |
| `wnut2017` | [WNUT2017](#wnut2017) | `Knowledge`, `NER` |
| `zebralogicbench` | [ZebraLogicBench](#zebralogicbench) | `Reasoning` |

---

## Benchmark Details

### AA-LCR

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `aa_lcr`
- **Dataset ID**: [evalscope/AA-LCR](https://modelscope.cn/datasets/evalscope/AA-LCR/summary)
- **Description**:
  > AA-LCR (Artificial Analysis Long Context Retrieval) is a benchmark for evaluating long-context retrieval and reasoning capabilities of language models across multiple documents.
- **Task Categories**: `Knowledge`, `LongContext`, `Reasoning`
- **Evaluation Metrics**: `acc`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: Yes
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `default`

- **Extra Parameters**: 
```json
{
    "text_dir": {
        "type": "str | null",
        "description": "Local directory containing extracted AA-LCR text files; if null will auto-download & extract.",
        "value": null
    }
}
```
- **Prompt Template**:
<details><summary>View</summary>

````text

BEGIN INPUT DOCUMENTS

{documents_text}

END INPUT DOCUMENTS

Answer the following question using the input documents provided above.

START QUESTION

{question}

END QUESTION

````

</details>

---

### AIME-2024

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `aime24`
- **Dataset ID**: [HuggingFaceH4/aime_2024](https://modelscope.cn/datasets/HuggingFaceH4/aime_2024/summary)
- **Description**:
  > The AIME 2024 benchmark is based on problems from the American Invitational Mathematics Examination, a prestigious high school mathematics competition. This benchmark tests a model's ability to solve challenging mathematics problems by generating step-by-step solutions and providing the correct final answer.
- **Task Categories**: `Math`, `Reasoning`
- **Evaluation Metrics**: `{'acc': {'numeric': True}}`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `train`
- **Subsets**: `default`

- **Prompt Template**:
<details><summary>View</summary>

````text
{question}
Please reason step by step, and put your final answer within \boxed{{}}.
````

</details>

---

### AIME-2025

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `aime25`
- **Dataset ID**: [opencompass/AIME2025](https://modelscope.cn/datasets/opencompass/AIME2025/summary)
- **Description**:
  > The AIME 2025 benchmark is based on problems from the American Invitational Mathematics Examination, a prestigious high school mathematics competition. This benchmark tests a model's ability to solve challenging mathematics problems by generating step-by-step solutions and providing the correct final answer.
- **Task Categories**: `Math`, `Reasoning`
- **Evaluation Metrics**: `{'acc': {'numeric': True}}`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `AIME2025-II`, `AIME2025-I`

- **Prompt Template**:
<details><summary>View</summary>

````text

Solve the following math problem step by step. Put your answer inside \boxed{{}}.

{question}

Remember to put your answer inside \boxed{{}}.
````

</details>

---

### AlpacaEval2.0

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `alpaca_eval`
- **Dataset ID**: [AI-ModelScope/alpaca_eval](https://modelscope.cn/datasets/AI-ModelScope/alpaca_eval/summary)
- **Description**:
  > Alpaca Eval 2.0 is an enhanced framework for evaluating instruction-following language models, featuring an improved auto-annotator, updated baselines, and continuous preference calculation to provide more accurate and cost-effective model assessments. Currently not support `length-controlled winrate`; the official Judge model is `gpt-4-1106-preview`, while the baseline model is `gpt-4-turbo`.
- **Task Categories**: `Arena`, `InstructionFollowing`
- **Evaluation Metrics**: `winrate`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: Yes
- **Default Shots**: 0-shot
- **Evaluation Split**: `eval`
- **Subsets**: `alpaca_eval_gpt4_baseline`

- **Prompt Template**:
<details><summary>View</summary>

````text
{question}
````

</details>

---

### AMC

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `amc`
- **Dataset ID**: [evalscope/amc_22-24](https://modelscope.cn/datasets/evalscope/amc_22-24/summary)
- **Description**:
  > AMC (American Mathematics Competitions) is a series of mathematics competitions for high school students.
- **Task Categories**: `Math`, `Reasoning`
- **Evaluation Metrics**: `{'acc': {'numeric': True}}`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Subsets**: `amc22`, `amc23`, `amc24`

- **Prompt Template**:
<details><summary>View</summary>

````text
{question}
Please reason step by step, and put your final answer within \boxed{{}}.
````

</details>

---

### ARC

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `arc`
- **Dataset ID**: [allenai/ai2_arc](https://modelscope.cn/datasets/allenai/ai2_arc/summary)
- **Description**:
  > The ARC (AI2 Reasoning Challenge) benchmark is designed to evaluate the reasoning capabilities of AI models through multiple-choice questions derived from science exams. It includes two subsets: ARC-Easy and ARC-Challenge, which vary in difficulty.
- **Task Categories**: `MCQ`, `Reasoning`
- **Evaluation Metrics**: `acc`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `ARC-Challenge`, `ARC-Easy`

- **Prompt Template**:
<details><summary>View</summary>

````text
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}.

{question}

{choices}
````

</details>

---

### ArenaHard

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `arena_hard`
- **Dataset ID**: [AI-ModelScope/arena-hard-auto-v0.1](https://modelscope.cn/datasets/AI-ModelScope/arena-hard-auto-v0.1/summary)
- **Description**:
  > ArenaHard is a benchmark designed to evaluate the performance of large language models in a competitive setting, where models are pitted against each other in a series of tasks to determine their relative strengths and weaknesses. It includes a set of challenging tasks that require reasoning, understanding, and generation capabilities. Currently not support `style-controlled winrate`; the official Judge model is `gpt-4-1106-preview`, while the baseline model is `gpt-4-0314`.
- **Task Categories**: `Arena`, `InstructionFollowing`
- **Evaluation Metrics**: `winrate`
- **Aggregation Methods**: `elo`
- **Requires LLM Judge**: Yes
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `default`

- **Prompt Template**:
<details><summary>View</summary>

````text
{question}
````

</details>

---

### BBH

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `bbh`
- **Dataset ID**: [evalscope/bbh](https://modelscope.cn/datasets/evalscope/bbh/summary)
- **Description**:
  > The BBH (Big Bench Hard) benchmark is a collection of challenging tasks designed to evaluate the reasoning capabilities of AI models. It includes both free-form and multiple-choice tasks, covering a wide range of reasoning skills.
- **Task Categories**: `Reasoning`
- **Evaluation Metrics**: `acc`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 3-shot
- **Evaluation Split**: `test`
- **Subsets**: `boolean_expressions`, `causal_judgement`, `date_understanding`, `disambiguation_qa`, `dyck_languages`, `formal_fallacies`, `geometric_shapes`, `hyperbaton`, `logical_deduction_five_objects`, `logical_deduction_seven_objects`, `logical_deduction_three_objects`, `movie_recommendation`, `multistep_arithmetic_two`, `navigate`, `object_counting`, `penguins_in_a_table`, `reasoning_about_colored_objects`, `ruin_names`, `salient_translation_error_detection`, `snarks`, `sports_understanding`, `temporal_sequences`, `tracking_shuffled_objects_five_objects`, `tracking_shuffled_objects_seven_objects`, `tracking_shuffled_objects_three_objects`, `web_of_lies`, `word_sorting`

- **Prompt Template**:
<details><summary>View</summary>

````text
Q: {question}
A: Let's think step by step. Put your final answer in the format of "So the answer is [ANSWER]" (without quotes and markdown) where [ANSWER] is the answer to the problem.

````

</details>

---

### BioMixQA

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `biomix_qa`
- **Dataset ID**: [extraordinarylab/biomix-qa](https://modelscope.cn/datasets/extraordinarylab/biomix-qa/summary)
- **Description**:
  > BiomixQA is a curated biomedical question-answering dataset. BiomixQA has been utilized to validate the Knowledge Graph based Retrieval-Augmented Generation (KG-RAG) framework across different LLMs.
- **Task Categories**: `Knowledge`, `MCQ`, `Medical`
- **Evaluation Metrics**: `acc`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `default`

- **Prompt Template**:
<details><summary>View</summary>

````text
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}.

{question}

{choices}
````

</details>

---

### BroadTwitterCorpus

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `broad_twitter_corpus`
- **Dataset ID**: [extraordinarylab/broad-twitter-corpus](https://modelscope.cn/datasets/extraordinarylab/broad-twitter-corpus/summary)
- **Description**:
  > BroadTwitterCorpus is a dataset of tweets collected over stratified times, places and social uses. The goal is to represent a broad range of activities, giving a dataset more representative of the language used in this hardest of social media formats to process.
- **Task Categories**: `Knowledge`, `NER`
- **Evaluation Metrics**: `accuracy`, `f1_score`, `precision`, `recall`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 5-shot
- **Evaluation Split**: `test`
- **Subsets**: `default`

- **Prompt Template**:
<details><summary>View</summary>

````text
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

````

</details>

---

### C-Eval

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `ceval`
- **Dataset ID**: [evalscope/ceval](https://modelscope.cn/datasets/evalscope/ceval/summary)
- **Description**:
  > C-Eval is a benchmark designed to evaluate the performance of AI models on Chinese exams across various subjects, including STEM, social sciences, and humanities. It consists of multiple-choice questions that test knowledge and reasoning abilities in these areas.
- **Task Categories**: `Chinese`, `Knowledge`, `MCQ`
- **Evaluation Metrics**: `acc`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 5-shot
- **Evaluation Split**: `val`
- **Subsets**: `accountant`, `advanced_mathematics`, `art_studies`, `basic_medicine`, `business_administration`, `chinese_language_and_literature`, `civil_servant`, `clinical_medicine`, `college_chemistry`, `college_economics`, `college_physics`, `college_programming`, `computer_architecture`, `computer_network`, `discrete_mathematics`, `education_science`, `electrical_engineer`, `environmental_impact_assessment_engineer`, `fire_engineer`, `high_school_biology`, `high_school_chemistry`, `high_school_chinese`, `high_school_geography`, `high_school_history`, `high_school_mathematics`, `high_school_physics`, `high_school_politics`, `ideological_and_moral_cultivation`, `law`, `legal_professional`, `logic`, `mao_zedong_thought`, `marxism`, `metrology_engineer`, `middle_school_biology`, `middle_school_chemistry`, `middle_school_geography`, `middle_school_history`, `middle_school_mathematics`, `middle_school_physics`, `middle_school_politics`, `modern_chinese_history`, `operating_system`, `physician`, `plant_protection`, `probability_and_statistics`, `professional_tour_guide`, `sports_science`, `tax_accountant`, `teacher_qualification`, `urban_and_rural_planner`, `veterinary_medicine`

- **Prompt Template**:
<details><summary>View</summary>

````text
以下是中国关于{subject}的单项选择题，请选出其中的正确答案。你的回答的最后一行应该是这样的格式："答案：[LETTER]"（不带引号），其中 [LETTER] 是 A、B、C、D 中的一个。

问题：{question}
选项：
{choices}

````

</details>

---

### Chinese-SimpleQA

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `chinese_simpleqa`
- **Dataset ID**: [AI-ModelScope/Chinese-SimpleQA](https://modelscope.cn/datasets/AI-ModelScope/Chinese-SimpleQA/summary)
- **Description**:
  > Chinese SimpleQA is a Chinese question-answering dataset designed to evaluate the performance of language models on simple factual questions. It includes a variety of topics and is structured to test the model's ability to understand and generate correct answers in Chinese.
- **Task Categories**: `Chinese`, `Knowledge`, `QA`
- **Evaluation Metrics**: `is_correct`, `is_incorrect`, `is_not_attempted`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: Yes
- **Default Shots**: 0-shot
- **Evaluation Split**: `train`
- **Subsets**: `中华文化`, `人文与社会科学`, `工程、技术与应用科学`, `生活、艺术与文化`, `社会`, `自然与自然科学`

- **Prompt Template**:
<details><summary>View</summary>

````text
请回答问题：

{question}
````

</details>

---

### C-MMLU

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `cmmlu`
- **Dataset ID**: [evalscope/cmmlu](https://modelscope.cn/datasets/evalscope/cmmlu/summary)
- **Description**:
  > C-MMLU is a benchmark designed to evaluate the performance of AI models on Chinese language tasks, including reading comprehension, text classification, and more.
- **Task Categories**: `Chinese`, `Knowledge`, `MCQ`
- **Evaluation Metrics**: `acc`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `agronomy`, `anatomy`, `ancient_chinese`, `arts`, `astronomy`, `business_ethics`, `chinese_civil_service_exam`, `chinese_driving_rule`, `chinese_food_culture`, `chinese_foreign_policy`, `chinese_history`, `chinese_literature`, `chinese_teacher_qualification`, `clinical_knowledge`, `college_actuarial_science`, `college_education`, `college_engineering_hydrology`, `college_law`, `college_mathematics`, `college_medical_statistics`, `college_medicine`, `computer_science`, `computer_security`, `conceptual_physics`, `construction_project_management`, `economics`, `education`, `electrical_engineering`, `elementary_chinese`, `elementary_commonsense`, `elementary_information_and_technology`, `elementary_mathematics`, `ethnology`, `food_science`, `genetics`, `global_facts`, `high_school_biology`, `high_school_chemistry`, `high_school_geography`, `high_school_mathematics`, `high_school_physics`, `high_school_politics`, `human_sexuality`, `international_law`, `journalism`, `jurisprudence`, `legal_and_moral_basis`, `logical`, `machine_learning`, `management`, `marketing`, `marxist_theory`, `modern_chinese`, `nutrition`, `philosophy`, `professional_accounting`, `professional_law`, `professional_medicine`, `professional_psychology`, `public_relations`, `security_study`, `sociology`, `sports_science`, `traditional_chinese_medicine`, `virology`, `world_history`, `world_religions`

- **Prompt Template**:
<details><summary>View</summary>

````text
回答下面的单项选择题，请选出其中的正确答案。你的回答的最后一行应该是这样的格式："答案：[LETTER]"（不带引号），其中 [LETTER] 是 {letters} 中的一个。请在回答前进行一步步思考。

问题：{question}
选项：
{choices}

````

</details>

---

### CoinFlip

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `coin_flip`
- **Dataset ID**: [extraordinarylab/coin-flip](https://modelscope.cn/datasets/extraordinarylab/coin-flip/summary)
- **Description**:
  > CoinFlip is a symbolic reasoning dataset that tests an LLM's ability to track binary state changes through a sequence of actions. Each example describes whether a coin is flipped or not by different person, requiring logical inference to determine the final state (heads or tails).
- **Task Categories**: `Reasoning`, `Yes/No`
- **Evaluation Metrics**: `accuracy`, `f1_score`, `precision`, `recall`, `yes_ratio`
- **Aggregation Methods**: `f1`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `default`

- **Prompt Template**:
<details><summary>View</summary>

````text

Solve the following coin flip problem step by step. The last line of your response should be of the form "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the answer to the problem.

{question}

Remember to put your answer on its own line at the end in the form "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the answer YES or NO to the problem.

Reasoning:

````

</details>

---

### CommonsenseQA

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `commonsense_qa`
- **Dataset ID**: [extraordinarylab/commonsense-qa](https://modelscope.cn/datasets/extraordinarylab/commonsense-qa/summary)
- **Description**:
  > CommonsenseQA requires different types of commonsense knowledge to predict the correct answers.
- **Task Categories**: `Commonsense`, `MCQ`, `Reasoning`
- **Evaluation Metrics**: `acc`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `validation`
- **Subsets**: `default`

- **Prompt Template**:
<details><summary>View</summary>

````text
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}.

{question}

{choices}
````

</details>

---

### Competition-MATH

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `competition_math`
- **Dataset ID**: [evalscope/competition_math](https://modelscope.cn/datasets/evalscope/competition_math/summary)
- **Description**:
  > The MATH (Mathematics) benchmark is designed to evaluate the mathematical reasoning abilities of AI models through a variety of problem types, including arithmetic, algebra, geometry, and more.
- **Task Categories**: `Math`, `Reasoning`
- **Evaluation Metrics**: `{'acc': {'numeric': True}}`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 4-shot
- **Evaluation Split**: `test`
- **Subsets**: `Level 1`, `Level 2`, `Level 3`, `Level 4`, `Level 5`

- **Prompt Template**:
<details><summary>View</summary>

````text
Problem:
{question}

Please reason step by step, and put your final answer within \boxed{{}}.

````

</details>

---

### CoNLL2003

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `conll2003`
- **Dataset ID**: [evalscope/conll2003](https://modelscope.cn/datasets/evalscope/conll2003/summary)
- **Description**:
  > The ConLL-2003 dataset is for the Named Entity Recognition (NER) task. It was introduced as part of the ConLL-2003 Shared Task conference and contains texts annotated with entities such as people, organizations, places, and various names.
- **Task Categories**: `Knowledge`, `NER`
- **Evaluation Metrics**: `accuracy`, `f1_score`, `precision`, `recall`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 5-shot
- **Evaluation Split**: `test`
- **Subsets**: `default`

- **Prompt Template**:
<details><summary>View</summary>

````text
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

````

</details>

---

### Copious

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `copious`
- **Dataset ID**: [extraordinarylab/copious](https://modelscope.cn/datasets/extraordinarylab/copious/summary)
- **Description**:
  > Copious corpus is a gold standard corpus that covers a wide range of biodiversity entities, consisting of 668 documents downloaded from the Biodiversity Heritage Library with over 26K sentences and more than 28K entities.
- **Task Categories**: `Knowledge`, `NER`
- **Evaluation Metrics**: `accuracy`, `f1_score`, `precision`, `recall`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 5-shot
- **Evaluation Split**: `test`
- **Subsets**: `default`

- **Prompt Template**:
<details><summary>View</summary>

````text
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

````

</details>

---

### CrossNER

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `cross_ner`
- **Dataset ID**: [extraordinarylab/cross-ner](https://modelscope.cn/datasets/extraordinarylab/cross-ner/summary)
- **Description**:
  > CrossNER is a fully-labelled collected of named entity recognition (NER) data spanning over five diverse domains (AI, Literature, Music, Politics, Science).
- **Task Categories**: `Knowledge`, `NER`
- **Evaluation Metrics**: `accuracy`, `f1_score`, `precision`, `recall`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 5-shot
- **Evaluation Split**: `test`
- **Subsets**: `ai`, `literature`, `music`, `politics`, `science`

- **Prompt Template**:
<details><summary>View</summary>

````text
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

````

</details>

---

### Data-Collection

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `data_collection`
- **Dataset ID**: 
- **Description**:
  > Custom Data collection, mixing multiple evaluation datasets for a unified evaluation, aiming to use less data to achieve a more comprehensive assessment of the model's capabilities. [Usage Reference](https://evalscope.readthedocs.io/en/latest/advanced_guides/collection/index.html)
- **Task Categories**: `Custom`
- **Evaluation Metrics**: `acc`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `default`


---

### DocMath

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `docmath`
- **Dataset ID**: [yale-nlp/DocMath-Eval](https://modelscope.cn/datasets/yale-nlp/DocMath-Eval/summary)
- **Description**:
  > DocMath-Eval is a comprehensive benchmark focused on numerical reasoning within specialized domains. It requires the model to comprehend long and specialized documents and perform numerical reasoning to answer the given question.
- **Task Categories**: `LongContext`, `Math`, `Reasoning`
- **Evaluation Metrics**: `acc`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: Yes
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `complong_testmini`, `compshort_testmini`, `simplong_testmini`, `simpshort_testmini`

- **Prompt Template**:
<details><summary>View</summary>

````text
Please read the following text and answer the question below.

<text>
{context}
</text>

{question}

Format your response as follows: "Therefore, the answer is (insert answer here)".
````

</details>

---

### DrivelologyBinaryClassification

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `drivel_binary`
- **Dataset ID**: [extraordinarylab/drivel-hub](https://modelscope.cn/datasets/extraordinarylab/drivel-hub/summary)
- **Description**:
  > Drivelology, a unique linguistic phenomenon characterised as "nonsense with depth" - utterances that are syntactically coherent yet pragmatically paradoxical, emotionally loaded, or rhetorically subversive.
- **Task Categories**: `Yes/No`
- **Evaluation Metrics**: `accuracy`, `f1_score`, `precision`, `recall`, `yes_ratio`
- **Aggregation Methods**: `f1`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `binary-classification`

- **Prompt Template**:
<details><summary>View</summary>

````text
{question}
````

</details>

---

### DrivelologyMultilabelClassification

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `drivel_multilabel`
- **Dataset ID**: [extraordinarylab/drivel-hub](https://modelscope.cn/datasets/extraordinarylab/drivel-hub/summary)
- **Description**:
  > Drivelology, a unique linguistic phenomenon characterised as "nonsense with depth" - utterances that are syntactically coherent yet pragmatically paradoxical, emotionally loaded, or rhetorically subversive.
- **Task Categories**: `MCQ`
- **Evaluation Metrics**: `exact_match`, `f1_macro`, `f1_micro`, `f1_weighted`
- **Aggregation Methods**: `f1_weighted`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `multi-label-classification`

- **Prompt Template**:
<details><summary>View</summary>

````text
{question}
````

</details>

---

### DrivelologyNarrativeSelection

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `drivel_selection`
- **Dataset ID**: [extraordinarylab/drivel-hub](https://modelscope.cn/datasets/extraordinarylab/drivel-hub/summary)
- **Description**:
  > Drivelology, a unique linguistic phenomenon characterised as "nonsense with depth" - utterances that are syntactically coherent yet pragmatically paradoxical, emotionally loaded, or rhetorically subversive.
- **Task Categories**: `MCQ`
- **Evaluation Metrics**: `acc`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `multiple-choice-english-easy`, `multiple-choice-english-hard`

- **Prompt Template**:
<details><summary>View</summary>

````text
Tell me the best option in the following options which represents the underlying narrative of the text?
The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}.

{question}

{choices}
````

</details>

---

### DrivelologyNarrativeWriting

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `drivel_writing`
- **Dataset ID**: [extraordinarylab/drivel-hub](https://modelscope.cn/datasets/extraordinarylab/drivel-hub/summary)
- **Description**:
  > Drivelology, a unique linguistic phenomenon characterised as "nonsense with depth" - utterances that are syntactically coherent yet pragmatically paradoxical, emotionally loaded, or rhetorically subversive.
- **Task Categories**: `Knowledge`, `Reasoning`
- **Evaluation Metrics**: `{'bert_score': {'model_id_or_path': 'AI-ModelScope/roberta-large', 'model_type': 'roberta-large'}}`, `{'gpt_score': {}}`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: Yes
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `narrative-writing-english`

- **Prompt Template**:
<details><summary>View</summary>

````text
You need to first read and understand the text given. Generate a detailed description to illustrate the implicit narrative of the text.

Please provide your response in English, with a clear and comprehensive explanation of the narrative.

Text: {text}
````

</details>

---

### DROP

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `drop`
- **Dataset ID**: [AI-ModelScope/DROP](https://modelscope.cn/datasets/AI-ModelScope/DROP/summary)
- **Description**:
  > The DROP (Discrete Reasoning Over Paragraphs) benchmark is designed to evaluate the reading comprehension and reasoning capabilities of AI models. It includes a variety of tasks that require models to read passages and answer questions based on the content.
- **Task Categories**: `Reasoning`
- **Evaluation Metrics**: `em`, `f1`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 3-shot
- **Evaluation Split**: `validation`
- **Subsets**: `default`

- **Prompt Template**:
<details><summary>View</summary>

````text
You will be asked to read a passage and answer a question. {drop_examples}
# Your Task

---
{query}

Think step by step, then write a line of the form "Answer: [ANSWER]" at the end of your response.
````

</details>

---

### EQ-Bench

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `eq_bench`
- **Dataset ID**: [evalscope/EQ-Bench](https://modelscope.cn/datasets/evalscope/EQ-Bench/summary)
- **Description**:
  > EQ-Bench is a benchmark for evaluating language models on emotional intelligence tasks. It assesses the ability to predict the likely emotional responses of characters in dialogues by rating the intensity of possible emotional responses. [Paper](https://arxiv.org/abs/2312.06281) | [Homepage](https://eqbench.com/)
- **Task Categories**: `InstructionFollowing`
- **Evaluation Metrics**: `eq_bench_score`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `validation`
- **Subsets**: `default`

- **Prompt Template**:
<details><summary>View</summary>

````text
{question}
````

</details>

---

### FRAMES

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `frames`
- **Dataset ID**: [iic/frames](https://modelscope.cn/datasets/iic/frames/summary)
- **Description**:
  > FRAMES is a comprehensive evaluation dataset designed to test the capabilities of Retrieval-Augmented Generation (RAG) systems across factuality, retrieval accuracy, and reasoning.
- **Task Categories**: `LongContext`, `Reasoning`
- **Evaluation Metrics**: `acc`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: Yes
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `default`

- **Prompt Template**:
<details><summary>View</summary>

````text
Please read the following text and answer the question below.

<text>
{context}
</text>

{question}

Format your response as follows: "Therefore, the answer is (insert answer here)".
````

</details>

---

### GeneralArena

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `general_arena`
- **Dataset ID**: general_arena
- **Description**:
  > GeneralArena is a custom benchmark designed to evaluate the performance of large language models in a competitive setting, where models are pitted against each other in custom tasks to determine their relative strengths and weaknesses. You should provide the model outputs in the format of a list of dictionaries, where each dictionary contains the model name and its report path. For detailed instructions on how to use this benchmark, please refer to the [Arena User Guide](https://evalscope.readthedocs.io/en/latest/user_guides/arena.html).
- **Task Categories**: `Arena`, `Custom`
- **Evaluation Metrics**: `winrate`
- **Aggregation Methods**: `elo`
- **Requires LLM Judge**: Yes
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `default`

- **Extra Parameters**: 
```json
{
    "models": {
        "type": "list[dict]",
        "description": "List of model entries with name and report_path for arena comparison.",
        "value": [
            {
                "name": "qwen-plus",
                "report_path": "outputs/20250627_172550/reports/qwen-plus"
            },
            {
                "name": "qwen2.5-7b",
                "report_path": "outputs/20250627_172817/reports/qwen2.5-7b-instruct"
            }
        ]
    },
    "baseline": {
        "type": "str",
        "description": "Baseline model name used for ELO and winrate comparisons.",
        "value": "qwen2.5-7b"
    }
}
```
- **System Prompt**:
<details><summary>View</summary>

````text
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
````

</details>
- **Prompt Template**:
<details><summary>View</summary>

````text
<|User Prompt|>
{question}

<|The Start of Assistant A's Answer|>
{answer_1}
<|The End of Assistant A's Answer|>

<|The Start of Assistant B's Answer|>
{answer_2}
<|The End of Assistant B's Answer|>
````

</details>

---

### General-MCQ

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `general_mcq`
- **Dataset ID**: general_mcq
- **Description**:
  > A general multiple-choice question answering dataset for custom evaluation. For detailed instructions on how to use this benchmark, please refer to the [User Guide](https://evalscope.readthedocs.io/en/latest/advanced_guides/custom_dataset/llm.html#mcq).
- **Task Categories**: `Custom`, `MCQ`
- **Evaluation Metrics**: `acc`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `val`
- **Subsets**: `default`

- **Prompt Template**:
<details><summary>View</summary>

````text
回答下面的单项选择题，请选出其中的正确答案。你的回答的全部内容应该是这样的格式："答案：[LETTER]"（不带引号），其中 [LETTER] 是 {letters} 中的一个。

问题：{question}
选项：
{choices}

````

</details>

---

### General-QA

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `general_qa`
- **Dataset ID**: general_qa
- **Description**:
  > A general question answering dataset for custom evaluation. For detailed instructions on how to use this benchmark, please refer to the [User Guide](https://evalscope.readthedocs.io/en/latest/advanced_guides/custom_dataset/llm.html#qa).
- **Task Categories**: `Custom`, `QA`
- **Evaluation Metrics**: `BLEU`, `Rouge`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `default`

- **Prompt Template**:
<details><summary>View</summary>

````text
请回答问题
{question}
````

</details>

---

### GeniaNER

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `genia_ner`
- **Dataset ID**: [extraordinarylab/genia-ner](https://modelscope.cn/datasets/extraordinarylab/genia-ner/summary)
- **Description**:
  > GeniaNER consisting of 2,000 MEDLINE abstracts has been released with more than 400,000 words and almost 100,000 annotations for biological terms.
- **Task Categories**: `Knowledge`, `NER`
- **Evaluation Metrics**: `accuracy`, `f1_score`, `precision`, `recall`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 5-shot
- **Evaluation Split**: `test`
- **Subsets**: `default`

- **Prompt Template**:
<details><summary>View</summary>

````text
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

````

</details>

---

### GPQA-Diamond

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `gpqa_diamond`
- **Dataset ID**: [AI-ModelScope/gpqa_diamond](https://modelscope.cn/datasets/AI-ModelScope/gpqa_diamond/summary)
- **Description**:
  > GPQA is a dataset for evaluating the reasoning ability of large language models (LLMs) on complex mathematical problems. It contains questions that require step-by-step reasoning to arrive at the correct answer.
- **Task Categories**: `Knowledge`, `MCQ`
- **Evaluation Metrics**: `acc`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `train`
- **Subsets**: `default`

- **Prompt Template**:
<details><summary>View</summary>

````text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

{question}

{choices}
````

</details>

---

### GSM8K

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `gsm8k`
- **Dataset ID**: [AI-ModelScope/gsm8k](https://modelscope.cn/datasets/AI-ModelScope/gsm8k/summary)
- **Description**:
  > GSM8K (Grade School Math 8K) is a dataset of grade school math problems, designed to evaluate the mathematical reasoning abilities of AI models.
- **Task Categories**: `Math`, `Reasoning`
- **Evaluation Metrics**: `{'acc': {'numeric': True}}`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 4-shot
- **Evaluation Split**: `test`
- **Subsets**: `main`

- **Prompt Template**:
<details><summary>View</summary>

````text
{question}
Please reason step by step, and put your final answer within \boxed{{}}.
````

</details>

---

### HaluEval

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `halueval`
- **Dataset ID**: [evalscope/HaluEval](https://modelscope.cn/datasets/evalscope/HaluEval/summary)
- **Description**:
  > HaluEval is a large collection of generated and human-annotated hallucinated samples for evaluating the performance of LLMs in recognizing hallucination.
- **Task Categories**: `Hallucination`, `Knowledge`, `Yes/No`
- **Evaluation Metrics**: `accuracy`, `f1_score`, `precision`, `recall`, `yes_ratio`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `data`
- **Subsets**: `dialogue_samples`, `qa_samples`, `summarization_samples`

- **Prompt Template**:
<details><summary>View</summary>

````text
{question}
````

</details>

---

### HarveyNER

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `harvey_ner`
- **Dataset ID**: [extraordinarylab/harvey-ner](https://modelscope.cn/datasets/extraordinarylab/harvey-ner/summary)
- **Description**:
  > HarveyNER is a dataset with fine-grained locations annotated in tweets. This dataset presents unique challenges and characterizes many complex and long location mentions in informal descriptions.
- **Task Categories**: `Knowledge`, `NER`
- **Evaluation Metrics**: `accuracy`, `f1_score`, `precision`, `recall`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 5-shot
- **Evaluation Split**: `test`
- **Subsets**: `default`

- **Prompt Template**:
<details><summary>View</summary>

````text
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

````

</details>

---

### HealthBench

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `health_bench`
- **Dataset ID**: [openai-mirror/healthbench](https://modelscope.cn/datasets/openai-mirror/healthbench/summary)
- **Description**:
  > HealthBench: a new benchmark designed to better measure capabilities of AI systems for health. Built in partnership with 262 physicians who have practiced in 60 countries, HealthBench includes 5,000 realistic health conversations, each with a custom physician-created rubric to grade model responses.
- **Task Categories**: `Knowledge`, `Medical`, `QA`
- **Evaluation Metrics**: `accuracy`, `communication_quality`, `completeness`, `context_awareness`, `instruction_following`
- **Aggregation Methods**: `clipped_mean`
- **Requires LLM Judge**: Yes
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `communication`, `complex_responses`, `context_seeking`, `emergency_referrals`, `global_health`, `health_data_tasks`, `hedging`

- **Extra Parameters**: 
```json
{
    "version": {
        "type": "str",
        "description": "Dataset file version, choices: ['Consensus', 'Hard', 'All'].",
        "value": "Consensus",
        "choices": [
            "Consensus",
            "Hard",
            "All"
        ]
    }
}
```
- **Prompt Template**:
<details><summary>View</summary>

````text
Answer the question:

{question}
````

</details>

---

### HellaSwag

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `hellaswag`
- **Dataset ID**: [evalscope/hellaswag](https://modelscope.cn/datasets/evalscope/hellaswag/summary)
- **Description**:
  > HellaSwag is a benchmark for commonsense reasoning in natural language understanding tasks. It consists of multiple-choice questions where the model must select the most plausible continuation of a given context.
- **Task Categories**: `Commonsense`, `Knowledge`, `MCQ`
- **Evaluation Metrics**: `acc`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `validation`
- **Subsets**: `default`

- **Prompt Template**:
<details><summary>View</summary>

````text
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}.

{question}

{choices}
````

</details>

---

### Humanity's-Last-Exam

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `hle`
- **Dataset ID**: [cais/hle](https://modelscope.cn/datasets/cais/hle/summary)
- **Description**:
  > Humanity's Last Exam (HLE) is a language model benchmark consisting of 2,500 questions across a broad range of subjects. It was created jointly by the Center for AI Safety and Scale AI. The benchmark classifies the questions into the following broad subjects: mathematics (41%), physics (9%), biology/medicine (11%), humanities/social science (9%), computer science/artificial intelligence (10%), engineering (4%), chemistry (7%), and other (9%). Around 14% of the questions require the ability to understand both text and images, i.e., multi-modality. 24% of the questions are multiple-choice; the rest are short-answer, exact-match questions. 
  > **To evaluate the performance of model without multi-modality capabilities, please set the `extra_params["include_multi_modal"]` to `False`.**
- **Task Categories**: `Knowledge`, `QA`
- **Evaluation Metrics**: `acc`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: Yes
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `Biology/Medicine`, `Chemistry`, `Computer Science/AI`, `Engineering`, `Humanities/Social Science`, `Math`, `Other`, `Physics`

- **Extra Parameters**: 
```json
{
    "include_multi_modal": {
        "type": "bool",
        "description": "Include multi-modal (image) questions during evaluation.",
        "value": true
    }
}
```
- **Prompt Template**:
<details><summary>View</summary>

````text
{question}
````

</details>

---

### HumanEval

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `humaneval`
- **Dataset ID**: [opencompass/humaneval](https://modelscope.cn/datasets/opencompass/humaneval/summary)
- **Description**:
  > HumanEval is a benchmark for evaluating the ability of code generation models to write Python functions based on given specifications. It consists of programming tasks with a defined input-output behavior. **By default the code is executed in local environment. We recommend using sandbox execution to safely run and evaluate the generated code, please refer to the [documentation](https://evalscope.readthedocs.io/en/latest/user_guides/sandbox.html) for more details.**
- **Task Categories**: `Coding`
- **Evaluation Metrics**: `acc`
- **Aggregation Methods**: `mean_and_pass_at_k`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `openai_humaneval`

- **Review Timeout (seconds)**: 4
- **Sandbox Configuration**: 
```json
{
    "image": "python:3.11-slim",
    "tools_config": {
        "shell_executor": {},
        "python_executor": {}
    }
}
```
- **Prompt Template**:
<details><summary>View</summary>

````text
Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.
{question}
````

</details>

---

### IFBench

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `ifbench`
- **Dataset ID**: [allenai/IFBench_test](https://modelscope.cn/datasets/allenai/IFBench_test/summary)
- **Description**:
  > IFBench is a new benchmark designed to evaluate how reliably AI models follow novel, challenging, and diverse verifiable instructions, with a strong focus on out-of-domain generalization. It comprises 58 manually curated verifiable constraints across categories such as counting, formatting, and word usage, aiming to address overfitting and data contamination issues present in existing benchmarks. Developed by AllenAI, IFBench serves as a rigorous test for precise instruction-following capabilities.
- **Task Categories**: `InstructionFollowing`
- **Evaluation Metrics**: `inst_level_loose`, `inst_level_strict`, `prompt_level_loose`, `prompt_level_strict`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `train`
- **Subsets**: `default`


---

### IFEval

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `ifeval`
- **Dataset ID**: [opencompass/ifeval](https://modelscope.cn/datasets/opencompass/ifeval/summary)
- **Description**:
  > IFEval is a benchmark for evaluating instruction-following language models, focusing on their ability to understand and respond to various prompts. It includes a diverse set of tasks and metrics to assess model performance comprehensively.
- **Task Categories**: `InstructionFollowing`
- **Evaluation Metrics**: `inst_level_loose`, `inst_level_strict`, `prompt_level_loose`, `prompt_level_strict`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `train`
- **Subsets**: `default`


---

### IQuiz

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `iquiz`
- **Dataset ID**: [AI-ModelScope/IQuiz](https://modelscope.cn/datasets/AI-ModelScope/IQuiz/summary)
- **Description**:
  > IQuiz is a benchmark for evaluating AI models on IQ and EQ questions. It consists of multiple-choice questions where the model must select the correct answer and provide an explanation.
- **Task Categories**: `Chinese`, `Knowledge`, `MCQ`
- **Evaluation Metrics**: `acc`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `EQ`, `IQ`

- **Prompt Template**:
<details><summary>View</summary>

````text
回答下面的单项选择题，请选出其中的正确答案。你的回答的最后一行应该是这样的格式："答案：[LETTER]"（不带引号），其中 [LETTER] 是 {letters} 中的一个。请在回答前进行一步步思考。

问题：{question}
选项：
{choices}

````

</details>

---

### Live-Code-Bench

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `live_code_bench`
- **Dataset ID**: [AI-ModelScope/code_generation_lite](https://modelscope.cn/datasets/AI-ModelScope/code_generation_lite/summary)
- **Description**:
  > Live Code Bench is a benchmark for evaluating code generation models on real-world coding tasks. It includes a variety of programming problems with test cases to assess the model's ability to generate correct and efficient code solutions. **By default the code is executed in local environment. We recommend using sandbox execution to safely run and evaluate the generated code, please refer to the [documentation](https://evalscope.readthedocs.io/en/latest/user_guides/sandbox.html) for more details.**
- **Task Categories**: `Coding`
- **Evaluation Metrics**: `acc`
- **Aggregation Methods**: `mean_and_pass_at_k`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `release_latest`

- **Review Timeout (seconds)**: 6
- **Extra Parameters**: 
```json
{
    "start_date": {
        "type": "str | null",
        "description": "Filter problems starting from this date (YYYY-MM-DD). Null keeps all.",
        "value": null
    },
    "end_date": {
        "type": "str | null",
        "description": "Filter problems up to this date (YYYY-MM-DD). Null keeps all.",
        "value": null
    },
    "debug": {
        "type": "bool",
        "description": "Enable verbose debug logging and bypass certain safety checks.",
        "value": false
    }
}
```
- **Sandbox Configuration**: 
```json
{
    "image": "python:3.11-slim",
    "tools_config": {
        "shell_executor": {},
        "python_executor": {}
    }
}
```
- **Prompt Template**:
<details><summary>View</summary>

````text
### Question:
{question_content}

{format_prompt} ### Answer: (use the provided format with backticks)


````

</details>

---

### LogiQA

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `logi_qa`
- **Dataset ID**: [extraordinarylab/logiqa](https://modelscope.cn/datasets/extraordinarylab/logiqa/summary)
- **Description**:
  > LogiQA is a dataset sourced from expert-written questions for testing human Logical reasoning.
- **Task Categories**: `MCQ`, `Reasoning`
- **Evaluation Metrics**: `acc`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `default`

- **Prompt Template**:
<details><summary>View</summary>

````text
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}.

{question}

{choices}
````

</details>

---

### MaritimeBench

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `maritime_bench`
- **Dataset ID**: [HiDolphin/MaritimeBench](https://modelscope.cn/datasets/HiDolphin/MaritimeBench/summary)
- **Description**:
  > MaritimeBench is a benchmark for evaluating AI models on maritime-related multiple-choice questions. It consists of questions related to maritime knowledge, where the model must select the correct answer from given options.
- **Task Categories**: `Chinese`, `Knowledge`, `MCQ`
- **Evaluation Metrics**: `acc`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `default`

- **Prompt Template**:
<details><summary>View</summary>

````text
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
````

</details>

---

### MATH-500

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `math_500`
- **Dataset ID**: [AI-ModelScope/MATH-500](https://modelscope.cn/datasets/AI-ModelScope/MATH-500/summary)
- **Description**:
  > MATH-500 is a benchmark for evaluating mathematical reasoning capabilities of AI models. It consists of 500 diverse math problems across five levels of difficulty, designed to test a model's ability to solve complex mathematical problems by generating step-by-step solutions and providing the correct final answer.
- **Task Categories**: `Math`, `Reasoning`
- **Evaluation Metrics**: `{'acc': {'numeric': True}}`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `Level 1`, `Level 2`, `Level 3`, `Level 4`, `Level 5`

- **Prompt Template**:
<details><summary>View</summary>

````text
{question}
Please reason step by step, and put your final answer within \boxed{{}}.
````

</details>

---

### MathQA

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `math_qa`
- **Dataset ID**: [extraordinarylab/math-qa](https://modelscope.cn/datasets/extraordinarylab/math-qa/summary)
- **Description**:
  > MathQA dataset is gathered by using a new representation language to annotate over the AQuA-RAT dataset with fully-specified operational programs.
- **Task Categories**: `MCQ`, `Math`, `Reasoning`
- **Evaluation Metrics**: `acc`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `default`

- **Prompt Template**:
<details><summary>View</summary>

````text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

{question}

{choices}
````

</details>

---

### MBPP

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `mbpp`
- **Dataset ID**: [google-research-datasets/mbpp](https://modelscope.cn/datasets/google-research-datasets/mbpp/summary)
- **Description**:
  > MBPP (Mostly Basic Python Problems Dataset): The benchmark consists of around 1,000 crowd-sourced Python programming problems, designed to be solvable by entry level programmers, covering programming fundamentals, standard library functionality, and so on. Each problem consists of a task description, code solution and 3 automated test cases.**Sandbox environment is needed for execution to safely run and evaluate the generated code, please refer to the [documentation](https://evalscope.readthedocs.io/en/latest/user_guides/sandbox.html) for more details.**
- **Task Categories**: `Coding`
- **Evaluation Metrics**: `acc`
- **Aggregation Methods**: `mean_and_pass_at_k`
- **Requires LLM Judge**: No
- **Default Shots**: 3-shot
- **Evaluation Split**: `test`
- **Subsets**: `full`

- **Review Timeout (seconds)**: 20
- **Sandbox Configuration**: 
```json
{
    "image": "python:3.11-slim",
    "tools_config": {
        "shell_executor": {},
        "python_executor": {}
    }
}
```
- **Prompt Template**:
<details><summary>View</summary>

````text
You are an expert Python programmer, and here is your task: {question} Your code should pass these tests:

{tests}
````

</details>

---

### Med-MCQA

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `med_mcqa`
- **Dataset ID**: [extraordinarylab/medmcqa](https://modelscope.cn/datasets/extraordinarylab/medmcqa/summary)
- **Description**:
  > MedMCQA is a large-scale MCQA dataset designed to address real-world medical entrance exam questions.
- **Task Categories**: `Knowledge`, `MCQ`
- **Evaluation Metrics**: `acc`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `validation`
- **Subsets**: `default`

- **Prompt Template**:
<details><summary>View</summary>

````text
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}.

{question}

{choices}
````

</details>

---

### MGSM

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `mgsm`
- **Dataset ID**: [evalscope/mgsm](https://modelscope.cn/datasets/evalscope/mgsm/summary)
- **Description**:
  > Multilingual Grade School Math Benchmark (MGSM) is a benchmark of grade-school math problems, proposed in the paper Language models are multilingual chain-of-thought reasoners.
- **Task Categories**: `Math`, `MultiLingual`, `Reasoning`
- **Evaluation Metrics**: `{'acc': {'numeric': True}}`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 4-shot
- **Evaluation Split**: `test`
- **Subsets**: `bn`, `de`, `en`, `es`, `fr`, `ja`, `ru`, `sw`, `te`, `th`, `zh`

- **Prompt Template**:
<details><summary>View</summary>

````text
{question}
Please reason step by step, and put your final answer within \boxed{{}}.


````

</details>

---

### Minerva-Math

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `minerva_math`
- **Dataset ID**: [knoveleng/Minerva-Math](https://modelscope.cn/datasets/knoveleng/Minerva-Math/summary)
- **Description**:
  > Minerva-math is a benchmark designed to evaluate the mathematical and quantitative reasoning capabilities of LLMs. It consists of **272 problems** sourced primarily from **MIT OpenCourseWare** courses, covering advanced STEM subjects such as solid-state chemistry, astronomy, differential equations, and special relativity at the **university and graduate level**.
- **Task Categories**: `Math`, `Reasoning`
- **Evaluation Metrics**: `{'acc': {'numeric': True}}`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: Yes
- **Default Shots**: 0-shot
- **Evaluation Split**: `train`
- **Subsets**: `default`

- **Prompt Template**:
<details><summary>View</summary>

````text
{question}
Please reason step by step, and put your final answer within \boxed{{}}.
````

</details>

---

### MIT-Movie-Trivia

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `mit_movie_trivia`
- **Dataset ID**: [extraordinarylab/mit-movie-trivia](https://modelscope.cn/datasets/extraordinarylab/mit-movie-trivia/summary)
- **Description**:
  > The MIT-Movie-Trivia dataset, originally created for slot filling, is modified by ignoring some slot types (e.g. genre, rating) and merging others (e.g. director and actor in person, and song and movie title in title) in order to keep consistent named entity types across all datasets.
- **Task Categories**: `Knowledge`, `NER`
- **Evaluation Metrics**: `accuracy`, `f1_score`, `precision`, `recall`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 5-shot
- **Evaluation Split**: `test`
- **Subsets**: `default`

- **Prompt Template**:
<details><summary>View</summary>

````text
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

````

</details>

---

### MIT-Restaurant

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `mit_restaurant`
- **Dataset ID**: [extraordinarylab/mit-restaurant](https://modelscope.cn/datasets/extraordinarylab/mit-restaurant/summary)
- **Description**:
  > The MIT-Restaurant dataset is a collection of restaurant review text specifically curated for training and testing Natural Language Processing (NLP) models, particularly for Named Entity Recognition (NER). It contains sentences from real reviews, along with corresponding labels in the BIO format.
- **Task Categories**: `Knowledge`, `NER`
- **Evaluation Metrics**: `accuracy`, `f1_score`, `precision`, `recall`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 5-shot
- **Evaluation Split**: `test`
- **Subsets**: `default`

- **Prompt Template**:
<details><summary>View</summary>

````text
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

````

</details>

---

### MMLU

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `mmlu`
- **Dataset ID**: [cais/mmlu](https://modelscope.cn/datasets/cais/mmlu/summary)
- **Description**:
  > The MMLU (Massive Multitask Language Understanding) benchmark is a comprehensive evaluation suite designed to assess the performance of language models across a wide range of subjects and tasks. It includes multiple-choice questions from various domains, such as history, science, mathematics, and more, providing a robust measure of a model's understanding and reasoning capabilities.
- **Task Categories**: `Knowledge`, `MCQ`
- **Evaluation Metrics**: `acc`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 5-shot
- **Evaluation Split**: `test`
- **Subsets**: `abstract_algebra`, `anatomy`, `astronomy`, `business_ethics`, `clinical_knowledge`, `college_biology`, `college_chemistry`, `college_computer_science`, `college_mathematics`, `college_medicine`, `college_physics`, `computer_security`, `conceptual_physics`, `econometrics`, `electrical_engineering`, `elementary_mathematics`, `formal_logic`, `global_facts`, `high_school_biology`, `high_school_chemistry`, `high_school_computer_science`, `high_school_european_history`, `high_school_geography`, `high_school_government_and_politics`, `high_school_macroeconomics`, `high_school_mathematics`, `high_school_microeconomics`, `high_school_physics`, `high_school_psychology`, `high_school_statistics`, `high_school_us_history`, `high_school_world_history`, `human_aging`, `human_sexuality`, `international_law`, `jurisprudence`, `logical_fallacies`, `machine_learning`, `management`, `marketing`, `medical_genetics`, `miscellaneous`, `moral_disputes`, `moral_scenarios`, `nutrition`, `philosophy`, `prehistory`, `professional_accounting`, `professional_law`, `professional_medicine`, `professional_psychology`, `public_relations`, `security_studies`, `sociology`, `us_foreign_policy`, `virology`, `world_religions`

- **Prompt Template**:
<details><summary>View</summary>

````text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

{question}

{choices}
````

</details>

---

### MMLU-Pro

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `mmlu_pro`
- **Dataset ID**: [TIGER-Lab/MMLU-Pro](https://modelscope.cn/datasets/TIGER-Lab/MMLU-Pro/summary)
- **Description**:
  > MMLU-Pro is a benchmark for evaluating language models on multiple-choice questions across various subjects. It includes questions from different domains, where the model must select the correct answer from given options.
- **Task Categories**: `Knowledge`, `MCQ`
- **Evaluation Metrics**: `acc`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 5-shot
- **Evaluation Split**: `test`
- **Subsets**: `biology`, `business`, `chemistry`, `computer science`, `economics`, `engineering`, `health`, `history`, `law`, `math`, `other`, `philosophy`, `physics`, `psychology`

- **Prompt Template**:
<details><summary>View</summary>

````text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

Question:
{question}
Options:
{choices}

````

</details>

---

### MMLU-Redux

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `mmlu_redux`
- **Dataset ID**: [AI-ModelScope/mmlu-redux-2.0](https://modelscope.cn/datasets/AI-ModelScope/mmlu-redux-2.0/summary)
- **Description**:
  > MMLU-Redux is a benchmark for evaluating language models on multiple-choice questions across various subjects. It includes questions from different domains, where the model must select the correct answer from given options. The bad answers are corrected.
- **Task Categories**: `Knowledge`, `MCQ`
- **Evaluation Metrics**: `{'acc': {'allow_inclusion': True}}`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `abstract_algebra`, `anatomy`, `astronomy`, `business_ethics`, `clinical_knowledge`, `college_biology`, `college_chemistry`, `college_computer_science`, `college_mathematics`, `college_medicine`, `college_physics`, `computer_security`, `conceptual_physics`, `econometrics`, `electrical_engineering`, `elementary_mathematics`, `formal_logic`, `global_facts`, `high_school_biology`, `high_school_chemistry`, `high_school_computer_science`, `high_school_european_history`, `high_school_geography`, `high_school_government_and_politics`, `high_school_macroeconomics`, `high_school_mathematics`, `high_school_microeconomics`, `high_school_physics`, `high_school_psychology`, `high_school_statistics`, `high_school_us_history`, `high_school_world_history`, `human_aging`, `human_sexuality`, `international_law`, `jurisprudence`, `logical_fallacies`, `machine_learning`, `management`, `marketing`, `medical_genetics`, `miscellaneous`, `moral_disputes`, `moral_scenarios`, `nutrition`, `philosophy`, `prehistory`, `professional_accounting`, `professional_law`, `professional_medicine`, `professional_psychology`, `public_relations`, `security_studies`, `sociology`, `us_foreign_policy`, `virology`, `world_religions`

- **Prompt Template**:
<details><summary>View</summary>

````text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

{question}

{choices}
````

</details>

---

### MRI-MCQA

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `mri_mcqa`
- **Dataset ID**: [extraordinarylab/mri-mcqa](https://modelscope.cn/datasets/extraordinarylab/mri-mcqa/summary)
- **Description**:
  > MRI-MCQA is a benchmark composed by multiple-choice questions related to Magnetic Resonance Imaging (MRI).
- **Task Categories**: `Knowledge`, `MCQ`, `Medical`
- **Evaluation Metrics**: `acc`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `default`

- **Prompt Template**:
<details><summary>View</summary>

````text
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}.

{question}

{choices}
````

</details>

---

### Multi-IF

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `multi_if`
- **Dataset ID**: [facebook/Multi-IF](https://modelscope.cn/datasets/facebook/Multi-IF/summary)
- **Description**:
  > Multi-IF is a benchmark designed to evaluate the performance of LLM models' capabilities in multi-turn instruction following within a multilingual environment.
- **Task Categories**: `InstructionFollowing`, `MultiLingual`, `MultiTurn`
- **Evaluation Metrics**: `inst_level_loose`, `inst_level_strict`, `prompt_level_loose`, `prompt_level_strict`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `train`
- **Subsets**: `Chinese`, `English`, `French`, `German`, `Hindi`, `Italian`, `Portuguese`, `Russian`, `Spanish`, `Thai`, `Vietnamese`

- **Extra Parameters**: 
```json
{
    "max_turns": {
        "type": "int",
        "description": "Maximum number of interactive turns to evaluate (1-3).",
        "value": 3,
        "choices": [
            1,
            2,
            3
        ]
    }
}
```

---

### MultiPL-E HumanEval

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `multiple_humaneval`
- **Dataset ID**: [evalscope/MultiPL-E](https://modelscope.cn/datasets/evalscope/MultiPL-E/summary)
- **Description**:
  > This multilingual HumanEval was from MultiPL-E. 18 languages were implemented and tested. **Sandbox environment is needed for execution to safely run and evaluate the generated code, please refer to the [documentation](https://evalscope.readthedocs.io/en/latest/user_guides/sandbox.html) for more details.**
- **Task Categories**: `Coding`
- **Evaluation Metrics**: `acc`
- **Aggregation Methods**: `mean_and_pass_at_k`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `humaneval-cpp`, `humaneval-cs`, `humaneval-d`, `humaneval-go`, `humaneval-java`, `humaneval-jl`, `humaneval-js`, `humaneval-lua`, `humaneval-php`, `humaneval-pl`, `humaneval-r`, `humaneval-rb`, `humaneval-rkt`, `humaneval-rs`, `humaneval-scala`, `humaneval-sh`, `humaneval-swift`, `humaneval-ts`

- **Review Timeout (seconds)**: 30
- **Sandbox Configuration**: 
```json
{
    "image": "volcengine/sandbox-fusion:server-20250609",
    "tools_config": {
        "shell_executor": {},
        "python_executor": {},
        "multi_code_executor": {}
    },
    "memory_limit": "2g",
    "cpu_limit": "2.0"
}
```
- **Prompt Template**:
<details><summary>View</summary>

````text
{prompt}
````

</details>

---

### MultiPL-E MBPP

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `multiple_mbpp`
- **Dataset ID**: [evalscope/MultiPL-E](https://modelscope.cn/datasets/evalscope/MultiPL-E/summary)
- **Description**:
  > This multilingual MBPP was from MultiPL-E. 18 languages were implemented and tested. **Sandbox environment is needed for execution to safely run and evaluate the generated code, please refer to the [documentation](https://evalscope.readthedocs.io/en/latest/user_guides/sandbox.html) for more details.**
- **Task Categories**: `Coding`
- **Evaluation Metrics**: `acc`
- **Aggregation Methods**: `mean_and_pass_at_k`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `mbpp-cpp`, `mbpp-cs`, `mbpp-d`, `mbpp-go`, `mbpp-java`, `mbpp-jl`, `mbpp-js`, `mbpp-lua`, `mbpp-php`, `mbpp-pl`, `mbpp-r`, `mbpp-rb`, `mbpp-rkt`, `mbpp-rs`, `mbpp-scala`, `mbpp-sh`, `mbpp-swift`, `mbpp-ts`

- **Review Timeout (seconds)**: 30
- **Sandbox Configuration**: 
```json
{
    "image": "volcengine/sandbox-fusion:server-20250609",
    "tools_config": {
        "shell_executor": {},
        "python_executor": {},
        "multi_code_executor": {}
    },
    "memory_limit": "2g",
    "cpu_limit": "2.0"
}
```
- **Prompt Template**:
<details><summary>View</summary>

````text
{prompt}
````

</details>

---

### MusicTrivia

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `music_trivia`
- **Dataset ID**: [extraordinarylab/music-trivia](https://modelscope.cn/datasets/extraordinarylab/music-trivia/summary)
- **Description**:
  > MusicTrivia is a curated dataset of multiple-choice questions covering both classical and modern music topics. It includes questions about composers, musical periods, and popular artists, designed for evaluating factual recall and domain-specific music knowledge.
- **Task Categories**: `Knowledge`, `MCQ`
- **Evaluation Metrics**: `acc`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `default`

- **Prompt Template**:
<details><summary>View</summary>

````text
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}.

{question}

{choices}
````

</details>

---

### MuSR

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `musr`
- **Dataset ID**: [AI-ModelScope/MuSR](https://modelscope.cn/datasets/AI-ModelScope/MuSR/summary)
- **Description**:
  > MuSR is a benchmark for evaluating AI models on multiple-choice questions related to murder mysteries, object placements, and team allocation.
- **Task Categories**: `MCQ`, `Reasoning`
- **Evaluation Metrics**: `acc`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `murder_mysteries`, `object_placements`, `team_allocation`

- **Prompt Template**:
<details><summary>View</summary>

````text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

{question}

{choices}
````

</details>

---

### Needle-in-a-Haystack

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `needle_haystack`
- **Dataset ID**: [AI-ModelScope/Needle-in-a-Haystack-Corpus](https://modelscope.cn/datasets/AI-ModelScope/Needle-in-a-Haystack-Corpus/summary)
- **Description**:
  > Needle in a Haystack is a benchmark focused on information retrieval tasks. It requires the model to find specific information within a large corpus of text. [Usage Example](https://evalscope.readthedocs.io/en/latest/third_party/needle_haystack.html)
- **Task Categories**: `LongContext`, `Retrieval`
- **Evaluation Metrics**: `acc`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: Yes
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `chinese`, `english`

- **Extra Parameters**: 
```json
{
    "retrieval_question": {
        "type": "str",
        "description": "Question used for retrieval evaluation.",
        "value": "What is the best thing to do in San Francisco?"
    },
    "needles": {
        "type": "list[str]",
        "description": "List of factual needle strings inserted into the context.",
        "value": [
            "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n"
        ]
    },
    "context_lengths_min": {
        "type": "int",
        "description": "Minimum context length (tokens) to generate synthetic samples.",
        "value": 1000
    },
    "context_lengths_max": {
        "type": "int",
        "description": "Maximum context length (tokens) to generate synthetic samples.",
        "value": 32000
    },
    "context_lengths_num_intervals": {
        "type": "int",
        "description": "Number of intervals between min and max context lengths.",
        "value": 10
    },
    "document_depth_percent_min": {
        "type": "int",
        "description": "Minimum insertion depth percentage for needles.",
        "value": 0
    },
    "document_depth_percent_max": {
        "type": "int",
        "description": "Maximum insertion depth percentage for needles.",
        "value": 100
    },
    "document_depth_percent_intervals": {
        "type": "int",
        "description": "Number of intervals between min and max depth percentages.",
        "value": 10
    },
    "tokenizer_path": {
        "type": "str",
        "description": "Tokenizer checkpoint path used for tokenization.",
        "value": "Qwen/Qwen3-0.6B"
    },
    "show_score": {
        "type": "bool",
        "description": "Render numerical scores on heatmap output images.",
        "value": false
    }
}
```
- **System Prompt**:
<details><summary>View</summary>

````text
You are a helpful AI bot that answers questions for a user. Keep your response short and direct
````

</details>
- **Prompt Template**:
<details><summary>View</summary>

````text
Please read the following text and answer the question below.

<text>
{context}
</text>

<question>
{question}
</question>

Don't give information outside the document or repeat your findings.
````

</details>

---

### OntoNotes5

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `ontonotes5`
- **Dataset ID**: [extraordinarylab/ontonotes5](https://modelscope.cn/datasets/extraordinarylab/ontonotes5/summary)
- **Description**:
  > OntoNotes Release 5.0 is a large, multilingual corpus containing text in English, Chinese, and Arabic across various genres like news, weblogs, and broadcast conversations. It is richly annotated with multiple layers of linguistic information, including syntax, predicate-argument structure, word sense, named entities, and coreference to support research and development in natural language processing.
- **Task Categories**: `Knowledge`, `NER`
- **Evaluation Metrics**: `accuracy`, `f1_score`, `precision`, `recall`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 5-shot
- **Evaluation Split**: `test`
- **Subsets**: `default`

- **Prompt Template**:
<details><summary>View</summary>

````text
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

````

</details>

---

### OpenAI MRCR

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `openai_mrcr`
- **Dataset ID**: [openai-mirror/mrcr](https://modelscope.cn/datasets/openai-mirror/mrcr/summary)
- **Description**:
  > Memory-Recall with Contextual Retrieval (MRCR). Evaluates retrieval and recall in long contexts by placing 2, 4 or 8 needles in the prompt. Measures whether the model can correctly extract and use them. 
- **Task Categories**: `LongContext`, `Retrieval`
- **Evaluation Metrics**: `mrcr_score`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `train`
- **Subsets**: `default`

- **Extra Parameters**: 
```json
{
    "max_context_size": {
        "type": "int | null",
        "description": "Maximum context tokens; samples exceeding are skipped. Defaults to None (no limit).",
        "value": null
    },
    "needle_count": {
        "type": "list[int] | null",
        "description": "Needle count filter (allowed: 2,4,8). Must be a list, e.g., [2], [4], or [2, 4, 8].  None keeps all.",
        "value": null
    },
    "tik_enc": {
        "type": "str",
        "description": "tiktoken encoding name used for token counting.",
        "value": "o200k_base"
    }
}
```

---

### PIQA

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `piqa`
- **Dataset ID**: [extraordinarylab/piqa](https://modelscope.cn/datasets/extraordinarylab/piqa/summary)
- **Description**:
  > PIQA addresses the challenging task of reasoning about physical commonsense in natural language.
- **Task Categories**: `Commonsense`, `MCQ`, `Reasoning`
- **Evaluation Metrics**: `acc`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `validation`
- **Subsets**: `default`

- **Prompt Template**:
<details><summary>View</summary>

````text
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}.

{question}

{choices}
````

</details>

---

### PolyMath

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `poly_math`
- **Dataset ID**: [evalscope/PolyMath](https://modelscope.cn/datasets/evalscope/PolyMath/summary)
- **Description**:
  > PolyMath is a multilingual mathematical reasoning benchmark covering 18 languages and 4 easy-to-hard difficulty levels, with 9,000 high-quality problem samples. Our benchmark ensures difficulty comprehensiveness, language diversity, and high-quality translation, making it a highly discriminative multilingual mathematical benchmark in the era of reasoning LLMs.
- **Task Categories**: `Math`, `MultiLingual`, `Reasoning`
- **Evaluation Metrics**: `{'acc': {'numeric': True}}`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `ar`, `bn`, `de`, `en`, `es`, `fr`, `id`, `it`, `ja`, `ko`, `ms`, `pt`, `ru`, `sw`, `te`, `th`, `vi`, `zh`

- **Prompt Template**:
<details><summary>View</summary>

````text
{question}
````

</details>

---

### ProcessBench

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `process_bench`
- **Dataset ID**: [Qwen/ProcessBench](https://modelscope.cn/datasets/Qwen/ProcessBench/summary)
- **Description**:
  > ProcessBench is a benchmark for evaluating AI models on mathematical reasoning tasks. It includes various subsets such as GSM8K, Math, OlympiadBench, and OmniMath, each with its own set of problems that require step-by-step reasoning to arrive at the correct answer.
- **Task Categories**: `Math`, `Reasoning`
- **Evaluation Metrics**: `correct_acc`, `error_acc`, `simple_f1_score`
- **Aggregation Methods**: `f1`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `gsm8k`, `math`, `olympiadbench`, `omnimath`

- **Prompt Template**:
<details><summary>View</summary>

````text
CThe following is a math problem and a solution (split into paragraphs, enclosed with tags and indexed from 0):

[Math Problem]

{problem}

[Solution]

{tagged_response}

Your task is to review and critique the solution paragraph by paragraph. Once you identify an error in a paragraph, return the index of the paragraph where the earliest error occurs. Otherwise, return the index of -1 (which typically denotes "not found").

Please put your final answer (i.e., the index) in oxed{{}}.

````

</details>

---

### PubMedQA

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `pubmedqa`
- **Dataset ID**: [extraordinarylab/pubmed-qa](https://modelscope.cn/datasets/extraordinarylab/pubmed-qa/summary)
- **Description**:
  > PubMedQA reasons over biomedical research texts to answer the multiple-choice questions.
- **Task Categories**: `Knowledge`, `Yes/No`
- **Evaluation Metrics**: `accuracy`, `f1_score`, `maybe_ratio`, `precision`, `recall`, `yes_ratio`
- **Aggregation Methods**: `f1`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `default`

- **Prompt Template**:
<details><summary>View</summary>

````text
{question}
Please answer YES or NO or MAYBE without an explanation.
````

</details>

---

### QASC

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `qasc`
- **Dataset ID**: [extraordinarylab/qasc](https://modelscope.cn/datasets/extraordinarylab/qasc/summary)
- **Description**:
  > QASC is a question-answering dataset with a focus on sentence composition. It consists of 9,980 8-way multiple-choice questions about grade school science.
- **Task Categories**: `Knowledge`, `MCQ`
- **Evaluation Metrics**: `acc`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `validation`
- **Subsets**: `default`

- **Prompt Template**:
<details><summary>View</summary>

````text
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}.

{question}

{choices}
````

</details>

---

### RACE

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `race`
- **Dataset ID**: [evalscope/race](https://modelscope.cn/datasets/evalscope/race/summary)
- **Description**:
  > RACE is a benchmark for testing reading comprehension and reasoning abilities of neural models. It is constructed from Chinese middle and high school examinations.
- **Task Categories**: `MCQ`, `Reasoning`
- **Evaluation Metrics**: `acc`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 3-shot
- **Evaluation Split**: `test`
- **Subsets**: `high`, `middle`

- **Prompt Template**:
<details><summary>View</summary>

````text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

{question}

{choices}
````

</details>

---

### SciCode

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `scicode`
- **Dataset ID**: [evalscope/SciCode](https://modelscope.cn/datasets/evalscope/SciCode/summary)
- **Description**:
  > SciCode is a challenging benchmark designed to evaluate the capabilities of language models (LMs) in generating code for solving realistic scientific research problems. It has a diverse coverage of 16 subdomains from 5 domains: Physics, Math, Material Science, Biology, and Chemistry. Unlike previous benchmarks that consist of exam-like question-answer pairs, SciCode is converted from real research problems. SciCode problems naturally factorize into multiple subproblems, each involving knowledge recall, reasoning, and code synthesis. **Sandbox environment is needed for execution to safely run and evaluate the generated code, please refer to the [documentation](https://evalscope.readthedocs.io/en/latest/user_guides/sandbox.html) for more details.**
- **Task Categories**: `Coding`
- **Evaluation Metrics**: `main_problem_pass_rate`, `subproblem_pass_rate`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `default`

- **Review Timeout (seconds)**: 300
- **Extra Parameters**: 
```json
{
    "provide_background": {
        "type": "bool",
        "value": false,
        "description": "Include scientific background information written by scientists for the problem in the model's prompt."
    }
}
```
- **Sandbox Configuration**: 
```json
{
    "image": "scicode-benchmark:latest",
    "tools_config": {
        "shell_executor": {},
        "python_executor": {}
    }
}
```
- **System Prompt**:
<details><summary>View</summary>

````text

PROBLEM DESCRIPTION:
You will be provided with a description of a scientific problem. You will solve these problems by solving a sequence of *subproblems*. The solution to each subproblem may be implemented using your solutions to earlier subproblems. Each subproblem should be solved by providing a Python function that meets the specifications provided.

For each subproblem, you will be provided with the following
 1. a description of the subproblem
 2. a function header, which you must use in your solution implementation
 3. a return line, which you must use in your solution implementation

You must only use the following dependencies to implement your solution:
{required_dependencies}

You MUST NOT import these dependencies anywhere in the code you generate.

For each subproblem provided you must solve it as follows:
 1. Generate scientific background required for the next step, in a comment
 2. Implement a function to solve the problem provided, using the provided header and return line

The response must be formatted as ```python```

````

</details>
- **Prompt Template**:
<details><summary>View</summary>

````text

Implement code to solve the following subproblem, using the description, function header, and return line provided.

Remember that you may use functions that you generated previously as solutions to previous subproblems to implement your answer.

Remember that you MUST NOT include code to import dependencies.

Remember to ensure your response is in the format of ```python``` and includes necessary background as a comment at the top.

SUBPROBLEM DESCRIPTION:
{step_description_prompt}

FUNCTION HEADER:
{function_header}

RETURN LINE:
{return_line}

Example:
```python
# Background: [Here, insert the necessary scientific knowledge required for the next step.]

[Insert the Python code here based on the provided function header and dependencies.]
```

````

</details>

---

### SciQ

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `sciq`
- **Dataset ID**: [extraordinarylab/sciq](https://modelscope.cn/datasets/extraordinarylab/sciq/summary)
- **Description**:
  > The SciQ dataset contains crowdsourced science exam questions about Physics, Chemistry and Biology, among others. For the majority of the questions, an additional paragraph with supporting evidence for the correct answer is provided.
- **Task Categories**: `Knowledge`, `MCQ`, `ReadingComprehension`
- **Evaluation Metrics**: `acc`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `default`

- **Prompt Template**:
<details><summary>View</summary>

````text
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}.

{question}

{choices}
````

</details>

---

### SimpleQA

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `simple_qa`
- **Dataset ID**: [evalscope/SimpleQA](https://modelscope.cn/datasets/evalscope/SimpleQA/summary)
- **Description**:
  > SimpleQA is a benchmark designed to evaluate the performance of language models on simple question-answering tasks. It includes a set of straightforward questions that require basic reasoning and understanding capabilities.
- **Task Categories**: `Knowledge`, `QA`
- **Evaluation Metrics**: `is_correct`, `is_incorrect`, `is_not_attempted`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: Yes
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `default`

- **Prompt Template**:
<details><summary>View</summary>

````text
Answer the question:

{question}
````

</details>

---

### SIQA

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `siqa`
- **Dataset ID**: [extraordinarylab/siqa](https://modelscope.cn/datasets/extraordinarylab/siqa/summary)
- **Description**:
  > Social Interaction QA (SIQA) is a question-answering benchmark for testing social commonsense intelligence. Contrary to many prior benchmarks that focus on physical or taxonomic knowledge, Social IQa focuses on reasoning about people's actions and their social implications.
- **Task Categories**: `Commonsense`, `MCQ`, `Reasoning`
- **Evaluation Metrics**: `acc`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `validation`
- **Subsets**: `default`

- **Prompt Template**:
<details><summary>View</summary>

````text
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}.

{question}

{choices}
````

</details>

---

### SuperGPQA

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `super_gpqa`
- **Dataset ID**: [m-a-p/SuperGPQA](https://modelscope.cn/datasets/m-a-p/SuperGPQA/summary)
- **Description**:
  > SuperGPQA is a large-scale multiple-choice question answering dataset, designed to evaluate the generalization ability of models across different fields. It contains 26,000+ questions from 50+ fields, with each question having 10 options.
- **Task Categories**: `Knowledge`, `MCQ`
- **Evaluation Metrics**: `acc`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `train`
- **Subsets**: `Aeronautical and Astronautical Science and Technology`, `Agricultural Engineering`, `Animal Husbandry`, `Applied Economics`, `Aquaculture`, `Architecture`, `Art Studies`, `Astronomy`, `Atmospheric Science`, `Basic Medicine`, `Biology`, `Business Administration`, `Chemical Engineering and Technology`, `Chemistry`, `Civil Engineering`, `Clinical Medicine`, `Computer Science and Technology`, `Control Science and Engineering`, `Crop Science`, `Education`, `Electrical Engineering`, `Electronic Science and Technology`, `Environmental Science and Engineering`, `Food Science and Engineering`, `Forestry Engineering`, `Forestry`, `Geography`, `Geological Resources and Geological Engineering`, `Geology`, `Geophysics`, `History`, `Hydraulic Engineering`, `Information and Communication Engineering`, `Instrument Science and Technology`, `Journalism and Communication`, `Language and Literature`, `Law`, `Library, Information and Archival Management`, `Management Science and Engineering`, `Materials Science and Engineering`, `Mathematics`, `Mechanical Engineering`, `Mechanics`, `Metallurgical Engineering`, `Military Science`, `Mining Engineering`, `Musicology`, `Naval Architecture and Ocean Engineering`, `Nuclear Science and Technology`, `Oceanography`, `Optical Engineering`, `Petroleum and Natural Gas Engineering`, `Pharmacy`, `Philosophy`, `Physical Education`, `Physical Oceanography`, `Physics`, `Political Science`, `Power Engineering and Engineering Thermophysics`, `Psychology`, `Public Administration`, `Public Health and Preventive Medicine`, `Sociology`, `Stomatology`, `Surveying and Mapping Science and Technology`, `Systems Science`, `Textile Science and Engineering`, `Theoretical Economics`, `Traditional Chinese Medicine`, `Transportation Engineering`, `Veterinary Medicine`, `Weapon Science and Technology`

- **Prompt Template**:
<details><summary>View</summary>

````text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

{question}

{choices}
````

</details>

---

### SWE-bench_Lite

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `swe_bench_lite`
- **Dataset ID**: [princeton-nlp/SWE-bench_Lite](https://modelscope.cn/datasets/princeton-nlp/SWE-bench_Lite/summary)
- **Description**:
  > SWE-bench Lite is subset of SWE-bench, a dataset that tests systems' ability to solve GitHub issues automatically. The dataset collects 300 test Issue-Pull Request pairs from 11 popular Python. Evaluation is performed by unit test verification using post-PR behavior as the reference solution. Need to run `pip install swebench==4.1.0` before evaluating. [Usage Example](https://evalscope.readthedocs.io/en/latest/third_party/swe_bench.html)
- **Task Categories**: `Coding`
- **Evaluation Metrics**: `acc`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `default`

- **Extra Parameters**: 
```json
{
    "build_docker_images": {
        "type": "bool",
        "description": "Build Docker images locally for each sample.",
        "value": true
    },
    "pull_remote_images_if_available": {
        "type": "bool",
        "description": "Attempt to pull existing remote Docker images before building.",
        "value": true
    },
    "inference_dataset_id": {
        "type": "str",
        "description": "Oracle dataset ID used to fetch inference context.",
        "value": "princeton-nlp/SWE-bench_oracle"
    }
}
```
- **Prompt Template**:
<details><summary>View</summary>

````text
{question}
````

</details>

---

### SWE-bench_Verified

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `swe_bench_verified`
- **Dataset ID**: [princeton-nlp/SWE-bench_Verified](https://modelscope.cn/datasets/princeton-nlp/SWE-bench_Verified/summary)
- **Description**:
  > SWE-bench Verified is a subset of 500 samples from the SWE-bench test set, which have been human-validated for quality. SWE-bench is a dataset that tests systems' ability to solve GitHub issues automatically. Need to run `pip install swebench==4.1.0` before evaluating. [Usage Example](https://evalscope.readthedocs.io/en/latest/third_party/swe_bench.html)
- **Task Categories**: `Coding`
- **Evaluation Metrics**: `acc`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `default`

- **Extra Parameters**: 
```json
{
    "inference_dataset_id": {
        "type": "str",
        "description": "Oracle dataset ID used to fetch inference context.",
        "value": "princeton-nlp/SWE-bench_oracle"
    },
    "build_docker_images": {
        "type": "bool",
        "description": "Build Docker images locally for each sample.",
        "value": true
    },
    "pull_remote_images_if_available": {
        "type": "bool",
        "description": "Attempt to pull existing remote Docker images before building.",
        "value": true
    }
}
```
- **Prompt Template**:
<details><summary>View</summary>

````text
{question}
````

</details>

---

### SWE-bench_Verified_mini

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `swe_bench_verified_mini`
- **Dataset ID**: [evalscope/swe-bench-verified-mini](https://modelscope.cn/datasets/evalscope/swe-bench-verified-mini/summary)
- **Description**:
  > SWEBench-verified-mini is a subset of SWEBench-verified that uses 50 instead of 500 datapoints, requires 5GB instead of 130GB of storage and has approximately the same distribution of performance, test pass rates and difficulty as the original dataset. Need to run `pip install swebench==4.1.0` before evaluating. [Usage Example](https://evalscope.readthedocs.io/en/latest/third_party/swe_bench.html)
- **Task Categories**: `Coding`
- **Evaluation Metrics**: `acc`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `default`

- **Extra Parameters**: 
```json
{
    "build_docker_images": {
        "type": "bool",
        "description": "Build Docker images locally for each sample.",
        "value": true
    },
    "pull_remote_images_if_available": {
        "type": "bool",
        "description": "Attempt to pull existing remote Docker images before building.",
        "value": true
    },
    "inference_dataset_id": {
        "type": "str",
        "description": "Oracle dataset ID used to fetch inference context.",
        "value": "princeton-nlp/SWE-bench_oracle"
    }
}
```
- **Prompt Template**:
<details><summary>View</summary>

````text
{question}
````

</details>

---

### TriviaQA

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `trivia_qa`
- **Dataset ID**: [evalscope/trivia_qa](https://modelscope.cn/datasets/evalscope/trivia_qa/summary)
- **Description**:
  > TriviaQA is a large-scale reading comprehension dataset consisting of question-answer pairs collected from trivia websites. It includes questions with multiple possible answers, making it suitable for evaluating the ability of models to understand and generate answers based on context.
- **Task Categories**: `QA`, `ReadingComprehension`
- **Evaluation Metrics**: `{'acc': {'allow_inclusion': True}}`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `validation`
- **Subsets**: `rc.wikipedia`

- **Prompt Template**:
<details><summary>View</summary>

````text
Read the content and answer the following question.

Content: {content}

Question: {question}

Keep your The last line of your response should be of the form "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the answer to the problem.

````

</details>

---

### TruthfulQA

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `truthful_qa`
- **Dataset ID**: [evalscope/truthful_qa](https://modelscope.cn/datasets/evalscope/truthful_qa/summary)
- **Description**:
  > TruthfulQA is a benchmark designed to evaluate the ability of AI models to answer questions truthfully and accurately. It includes multiple-choice tasks, focusing on the model's understanding of factual information.
- **Task Categories**: `Knowledge`
- **Evaluation Metrics**: `multi_choice_acc`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `validation`
- **Subsets**: `multiple_choice`

- **Extra Parameters**: 
```json
{
    "multiple_correct": {
        "type": "bool",
        "description": "Use multiple-answer format (MC2) if True; otherwise single-answer (MC1).",
        "value": false
    }
}
```
- **Prompt Template**:
<details><summary>View</summary>

````text
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}.

{question}

{choices}
````

</details>

---

### Winogrande

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `winogrande`
- **Dataset ID**: [AI-ModelScope/winogrande_val](https://modelscope.cn/datasets/AI-ModelScope/winogrande_val/summary)
- **Description**:
  > Winogrande is a benchmark for evaluating AI models on commonsense reasoning tasks, specifically designed to test the ability to resolve ambiguous pronouns in sentences.
- **Task Categories**: `MCQ`, `Reasoning`
- **Evaluation Metrics**: `acc`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `validation`
- **Subsets**: `default`

- **Prompt Template**:
<details><summary>View</summary>

````text
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}.

{question}

{choices}
````

</details>

---

### WMT2024++

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `wmt24pp`
- **Dataset ID**: [extraordinarylab/wmt24pp](https://modelscope.cn/datasets/extraordinarylab/wmt24pp/summary)
- **Description**:
  > WMT2024 news translation benchmark supporting multiple language pairs. Each subset represents a specific translation direction
- **Task Categories**: `MachineTranslation`, `MultiLingual`
- **Evaluation Metrics**: `{'bert_score': {'model_id_or_path': 'AI-ModelScope/xlm-roberta-large', 'model_type': 'xlm-roberta-large'}}`, `{'bleu': {}}`, `{'comet': {'model_id_or_path': 'evalscope/wmt22-comet-da'}}`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `en-ar_eg`, `en-ar_sa`, `en-bg_bg`, `en-bn_in`, `en-ca_es`, `en-cs_cz`, `en-da_dk`, `en-de_de`, `en-el_gr`, `en-es_mx`, `en-et_ee`, `en-fa_ir`, `en-fi_fi`, `en-fil_ph`, `en-fr_ca`, `en-fr_fr`, `en-gu_in`, `en-he_il`, `en-hi_in`, `en-hr_hr`, `en-hu_hu`, `en-id_id`, `en-is_is`, `en-it_it`, `en-ja_jp`, `en-kn_in`, `en-ko_kr`, `en-lt_lt`, `en-lv_lv`, `en-ml_in`, `en-mr_in`, `en-nl_nl`, `en-no_no`, `en-pa_in`, `en-pl_pl`, `en-pt_br`, `en-pt_pt`, `en-ro_ro`, `en-ru_ru`, `en-sk_sk`, `en-sl_si`, `en-sr_rs`, `en-sv_se`, `en-sw_ke`, `en-sw_tz`, `en-ta_in`, `en-te_in`, `en-th_th`, `en-tr_tr`, `en-uk_ua`, `en-ur_pk`, `en-vi_vn`, `en-zh_cn`, `en-zh_tw`, `en-zu_za`

- **Prompt Template**:
<details><summary>View</summary>

````text
Translate the following {source_language} sentence into {target_language}:

{source_language}: {source_text}
{target_language}:
````

</details>

---

### WNUT2017

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `wnut2017`
- **Dataset ID**: [extraordinarylab/wnut2017](https://modelscope.cn/datasets/extraordinarylab/wnut2017/summary)
- **Description**:
  > The WNUT2017 dataset is a collection of user-generated text from various social media platforms, like Twitter and YouTube, specifically designed for a named-entity recognition task.
- **Task Categories**: `Knowledge`, `NER`
- **Evaluation Metrics**: `accuracy`, `f1_score`, `precision`, `recall`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 5-shot
- **Evaluation Split**: `test`
- **Subsets**: `default`

- **Prompt Template**:
<details><summary>View</summary>

````text
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

````

</details>

---

### ZebraLogicBench

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `zebralogicbench`
- **Dataset ID**: [allenai/ZebraLogicBench-private](https://modelscope.cn/datasets/allenai/ZebraLogicBench-private/summary)
- **Description**:
  > ZebraLogic, a comprehensive evaluation framework for assessing LLM reasoning performance on logic grid puzzles derived from constraint satisfaction problems (CSPs).
- **Task Categories**: `Reasoning`
- **Evaluation Metrics**: `avg_reason_lens`, `cell_acc`, `easy_puzzle_acc`, `hard_puzzle_acc`, `large_puzzle_acc`, `medium_puzzle_acc`, `no_answer_num`, `puzzle_acc`, `small_puzzle_acc`, `xl_puzzle_acc`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `grid_mode`

- **Prompt Template**:
<details><summary>View</summary>

````text
# Example Puzzle

There are 3 houses, numbered 1 to 3 from left to right, as seen from across the street. Each house is occupied by a different person. Each house has a unique attribute for each of the following characteristics:
 - Each person has a unique name: `Peter`, `Eric`, `Arnold`.
 - Each person has a unique favorite drink: `tea`, `water`, `milk`

## Clues for the Example Puzzle

1. Peter is in the second house.
2. Arnold is directly left of the one who only drinks water.
3. The one who only drinks water is directly left of the person who likes milk.

## Answer to the Example Puzzle

{{
    "reasoning": "Given Clue 1, we know Peter is in House 2. According to Clue 2, Arnold is directly left of the one who only drinks water. The person in House 3 cannot be on the left of anyone, so Arnold must be in House 1. Thus, Peter drinks water, and Eric lives in House 3. Then, according to Clue 3, Eric drinks milk. Therefore, Arnold drinks tea.",
    "solution": {{
        "House 1": {{
            "Name": "Arnold",
            "Drink": "tea"
        }},
        "House 2": {{
            "Name": "Peter",
            "Drink": "water"
        }},
        "House 3": {{
            "Name": "Eric",
            "Drink": "milk"
        }}
    }}
}}

# Puzzle to Solve

{question}


# Instruction

Now please solve the above puzzle. Present your reasoning and solution in the following json format:

{json_template}


````

</details>
