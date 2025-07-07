# LLM评测集

以下是支持的LLM评测集列表，点击数据集标准名称可跳转详细信息。

| 数据集名称 | 标准名称 | 任务类别 |
|------------|----------|----------|
| `aime24` | [AIME-2024](#aime-2024) | `Mathematics` |
| `aime25` | [AIME-2025](#aime-2025) | `Mathematics` |
| `alpaca_eval` | [AlpacaEval2.0](#alpacaeval20) | `Arena`, `Instruction-Following` |
| `arc` | [ARC](#arc) | `MCQ`, `Reasoning` |
| `arena_hard` | [ArenaHard](#arenahard) | `Arena`, `Instruction-Following` |
| `bbh` | [BBH](#bbh) | `Reasoning` |
| `bfcl_v3` | [BFCL-v3](#bfcl-v3) | `Agent` |
| `ceval` | [C-Eval](#c-eval) | `Chinese`, `Knowledge`, `MCQ` |
| `chinese_simpleqa` | [Chinese-SimpleQA](#chinese-simpleqa) | `Chinese`, `Knowledge`, `QA` |
| `cmmlu` | [C-MMLU](#c-mmlu) | `Chinese`, `Knowledge`, `MCQ` |
| `competition_math` | [MATH](#math) | `Mathematics` |
| `docmath` | [DocMath](#docmath) | `Long Context`, `Mathematics`, `Reasoning` |
| `drop` | [DROP](#drop) | `Reasoning` |
| `frames` | [FRAMES](#frames) | `Long Context`, `Reasoning` |
| `general_arena` | [GeneralArena](#generalarena) | `Arena`, `Custom` |
| `general_mcq` | [General-MCQ](#general-mcq) | `Custom`, `MCQ` |
| `general_qa` | [General-QA](#general-qa) | `Custom`, `QA` |
| `gpqa` | [GPQA](#gpqa) | `Knowledge`, `MCQ` |
| `gsm8k` | [GSM8K](#gsm8k) | `Mathematics` |
| `hellaswag` | [HellaSwag](#hellaswag) | `Commonsense`, `Knowledge`, `MCQ` |
| `humaneval` | [HumanEval](#humaneval) | `Coding` |
| `ifeval` | [IFEval](#ifeval) | `Instruction-Following` |
| `iquiz` | [IQuiz](#iquiz) | `Chinese`, `Knowledge`, `MCQ` |
| `live_code_bench` | [Live-Code-Bench](#live-code-bench) | `Coding` |
| `maritime_bench` | [MaritimeBench](#maritimebench) | `Knowledge`, `MCQ`, `Maritime` |
| `math_500` | [MATH-500](#math-500) | `Mathematics` |
| `mmlu` | [MMLU](#mmlu) | `Knowledge`, `MCQ` |
| `mmlu_pro` | [MMLU-Pro](#mmlu-pro) | `Knowledge`, `MCQ` |
| `mmlu_redux` | [MMLU-Redux](#mmlu-redux) | `Knowledge`, `MCQ` |
| `musr` | [MuSR](#musr) | `MCQ`, `Reasoning` |
| `needle_haystack` | [Needle-in-a-Haystack](#needle-in-a-haystack) | `Long Context`, `Retrieval` |
| `process_bench` | [ProcessBench](#processbench) | `Mathematical`, `Reasoning` |
| `race` | [RACE](#race) | `MCQ`, `Reasoning` |
| `simple_qa` | [SimpleQA](#simpleqa) | `Knowledge`, `QA` |
| `super_gpqa` | [SuperGPQA](#supergpqa) | `Knowledge`, `MCQ` |
| `tool_bench` | [ToolBench-Static](#toolbench-static) | `Agent`, `Reasoning` |
| `trivia_qa` | [TriviaQA](#triviaqa) | `QA`, `Reading Comprehension` |
| `truthful_qa` | [TruthfulQA](#truthfulqa) | `Knowledge` |
| `winogrande` | [Winogrande](#winogrande) | `MCQ`, `Reasoning` |

---

## 数据集详情

### AIME-2024

[返回目录](#llm评测集)
- **数据集名称**: `aime24`
- **数据集ID**: [HuggingFaceH4/aime_2024](https://modelscope.cn/datasets/HuggingFaceH4/aime_2024/summary)
- **数据集描述**:  
  > The AIME 2024 benchmark is based on problems from the American Invitational Mathematics Examination, a prestigious high school mathematics competition. This benchmark tests a model’s ability to solve challenging mathematics problems by generating step-by-step solutions and providing the correct final answer.
- **任务类别**: `Mathematics`
- **评估指标**: `AveragePass@1`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `default`

- **支持输出格式**: `generation`
- **提示模板**: 
```text
{query}
Please reason step by step, and put your final answer within \boxed{{}}.
```

---

### AIME-2025

[返回目录](#llm评测集)
- **数据集名称**: `aime25`
- **数据集ID**: [opencompass/AIME2025](https://modelscope.cn/datasets/opencompass/AIME2025/summary)
- **数据集描述**:  
  > The AIME 2025 benchmark is based on problems from the American Invitational Mathematics Examination, a prestigious high school mathematics competition. This benchmark tests a model’s ability to solve challenging mathematics problems by generating step-by-step solutions and providing the correct final answer.
- **任务类别**: `Mathematics`
- **评估指标**: `AveragePass@1`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `AIME2025-II`, `AIME2025-I`

- **支持输出格式**: `generation`
- **提示模板**: 
```text
{query}
Please reason step by step, and put your final answer within \boxed{{}}.
```

---

### AlpacaEval2.0

[返回目录](#llm评测集)
- **数据集名称**: `alpaca_eval`
- **数据集ID**: [AI-ModelScope/alpaca_eval](https://modelscope.cn/datasets/AI-ModelScope/alpaca_eval/summary)
- **数据集描述**:  
  > Alpaca Eval 2.0 is an enhanced framework for evaluating instruction-following language models, featuring an improved auto-annotator, updated baselines, and continuous preference calculation to provide more accurate and cost-effective model assessments. Currently not support `length-controlled winrate`; the official Judge model is `gpt-4-1106-preview`, while the baseline model is `gpt-4-turbo`.
- **任务类别**: `Arena`, `Instruction-Following`
- **评估指标**: `winrate`
- **需要LLM Judge**: 是
- **默认提示方式**: 0-shot
- **数据集子集**: `alpaca_eval_gpt4_baseline`

- **支持输出格式**: `generation`

---

### ARC

[返回目录](#llm评测集)
- **数据集名称**: `arc`
- **数据集ID**: [modelscope/ai2_arc](https://modelscope.cn/datasets/modelscope/ai2_arc/summary)
- **数据集描述**:  
  > The ARC (AI2 Reasoning Challenge) benchmark is designed to evaluate the reasoning capabilities of AI models through multiple-choice questions derived from science exams. It includes two subsets: ARC-Easy and ARC-Challenge, which vary in difficulty.
- **任务类别**: `MCQ`, `Reasoning`
- **评估指标**: `AverageAccuracy`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `ARC-Challenge`, `ARC-Easy`

- **支持输出格式**: `generation`, `multiple_choice_logits`
- **提示模板**: 
```text
Given the following question and four candidate answers (A, B, C and D), choose the best answer.
{query}
Your response should end with "The best answer is [the_answer_letter]" where the [the_answer_letter] is one of A, B, C or D.
```

---

### ArenaHard

[返回目录](#llm评测集)
- **数据集名称**: `arena_hard`
- **数据集ID**: [AI-ModelScope/arena-hard-auto-v0.1](https://modelscope.cn/datasets/AI-ModelScope/arena-hard-auto-v0.1/summary)
- **数据集描述**:  
  > ArenaHard is a benchmark designed to evaluate the performance of large language models in a competitive setting, where models are pitted against each other in a series of tasks to determine their relative strengths and weaknesses. It includes a set of challenging tasks that require reasoning, understanding, and generation capabilities. Currently not support `style-controlled winrate`; the official Judge model is `gpt-4-1106-preview`, while the baseline model is `gpt-4-0314`.
- **任务类别**: `Arena`, `Instruction-Following`
- **评估指标**: `winrate`
- **需要LLM Judge**: 是
- **默认提示方式**: 0-shot
- **数据集子集**: `default`

- **支持输出格式**: `generation`

---

### BBH

[返回目录](#llm评测集)
- **数据集名称**: `bbh`
- **数据集ID**: [modelscope/bbh](https://modelscope.cn/datasets/modelscope/bbh/summary)
- **数据集描述**:  
  > The BBH (Big Bench Hard) benchmark is a collection of challenging tasks designed to evaluate the reasoning capabilities of AI models. It includes both free-form and multiple-choice tasks, covering a wide range of reasoning skills.
- **任务类别**: `Reasoning`
- **评估指标**: `AverageAccuracy`
- **需要LLM Judge**: 否
- **默认提示方式**: 3-shot
- **数据集子集**: `boolean_expressions`, `causal_judgement`, `date_understanding`, `disambiguation_qa`, `dyck_languages`, `formal_fallacies`, `geometric_shapes`, `hyperbaton`, `logical_deduction_five_objects`, `logical_deduction_seven_objects`, `logical_deduction_three_objects`, `movie_recommendation`, `multistep_arithmetic_two`, `navigate`, `object_counting`, `penguins_in_a_table`, `reasoning_about_colored_objects`, `ruin_names`, `salient_translation_error_detection`, `snarks`, `sports_understanding`, `temporal_sequences`, `tracking_shuffled_objects_five_objects`, `tracking_shuffled_objects_seven_objects`, `tracking_shuffled_objects_three_objects`, `web_of_lies`, `word_sorting`

- **支持输出格式**: `generation`
- **提示模板**: 
```text
Q: {query}
A: Let's think step by step.
```

---

### BFCL-v3

[返回目录](#llm评测集)
- **数据集名称**: `bfcl_v3`
- **数据集ID**: [AI-ModelScope/bfcl_v3](https://modelscope.cn/datasets/AI-ModelScope/bfcl_v3/summary)
- **数据集描述**:  
  > Berkeley Function Calling Leaderboard (BFCL), the **first comprehensive and executable function call evaluation** dedicated to assessing Large Language Models' (LLMs) ability to invoke functions. Unlike previous evaluations, BFCL accounts for various forms of function calls, diverse scenarios, and executability. Need to run `pip install bfcl-eval` before evaluating. [Usage Example](https://evalscope.readthedocs.io/zh-cn/latest/third_party/bfcl_v3.html)
- **任务类别**: `Agent`
- **评估指标**: `AverageAccuracy`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `irrelevance`, `java`, `javascript`, `live_irrelevance`, `live_multiple`, `live_parallel_multiple`, `live_parallel`, `live_relevance`, `live_simple`, `multi_turn_base`, `multi_turn_long_context`, `multi_turn_miss_func`, `multi_turn_miss_param`, `multiple`, `parallel_multiple`, `parallel`, `simple`

- **支持输出格式**: `generation`
- **额外参数**: 
```json
{
    "underscore_to_dot": true,
    "is_fc_model": true
}
```

---

### C-Eval

[返回目录](#llm评测集)
- **数据集名称**: `ceval`
- **数据集ID**: [modelscope/ceval-exam](https://modelscope.cn/datasets/modelscope/ceval-exam/summary)
- **数据集描述**:  
  > C-Eval is a benchmark designed to evaluate the performance of AI models on Chinese exams across various subjects, including STEM, social sciences, and humanities. It consists of multiple-choice questions that test knowledge and reasoning abilities in these areas.
- **任务类别**: `Chinese`, `Knowledge`, `MCQ`
- **评估指标**: `AverageAccuracy`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `accountant`, `advanced_mathematics`, `art_studies`, `basic_medicine`, `business_administration`, `chinese_language_and_literature`, `civil_servant`, `clinical_medicine`, `college_chemistry`, `college_economics`, `college_physics`, `college_programming`, `computer_architecture`, `computer_network`, `discrete_mathematics`, `education_science`, `electrical_engineer`, `environmental_impact_assessment_engineer`, `fire_engineer`, `high_school_biology`, `high_school_chemistry`, `high_school_chinese`, `high_school_geography`, `high_school_history`, `high_school_mathematics`, `high_school_physics`, `high_school_politics`, `ideological_and_moral_cultivation`, `law`, `legal_professional`, `logic`, `mao_zedong_thought`, `marxism`, `metrology_engineer`, `middle_school_biology`, `middle_school_chemistry`, `middle_school_geography`, `middle_school_history`, `middle_school_mathematics`, `middle_school_physics`, `middle_school_politics`, `modern_chinese_history`, `operating_system`, `physician`, `plant_protection`, `probability_and_statistics`, `professional_tour_guide`, `sports_science`, `tax_accountant`, `teacher_qualification`, `urban_and_rural_planner`, `veterinary_medicine`

- **支持输出格式**: `generation`, `multiple_choice_logits`
- **提示模板**: 
```text
以下是中国关于{subset_name}考试的单项选择题，请选出其中的正确答案。你的回答的最后一行应该是这样的格式：“答案是：LETTER”（不带引号），其中 LETTER 是 A、B、C、D 中的一个。
{query}
```

---

### Chinese-SimpleQA

[返回目录](#llm评测集)
- **数据集名称**: `chinese_simpleqa`
- **数据集ID**: [AI-ModelScope/Chinese-SimpleQA](https://modelscope.cn/datasets/AI-ModelScope/Chinese-SimpleQA/summary)
- **数据集描述**:  
  > Chinese SimpleQA is a Chinese question-answering dataset designed to evaluate the performance of language models on simple factual questions. It includes a variety of topics and is structured to test the model's ability to understand and generate correct answers in Chinese.
- **任务类别**: `Chinese`, `Knowledge`, `QA`
- **评估指标**: `is_correct`, `is_incorrect`, `is_not_attempted`
- **需要LLM Judge**: 是
- **默认提示方式**: 0-shot
- **数据集子集**: `中华文化`, `人文与社会科学`, `工程、技术与应用科学`, `生活、艺术与文化`, `社会`, `自然与自然科学`

- **支持输出格式**: `generation`

---

### C-MMLU

[返回目录](#llm评测集)
- **数据集名称**: `cmmlu`
- **数据集ID**: [modelscope/cmmlu](https://modelscope.cn/datasets/modelscope/cmmlu/summary)
- **数据集描述**:  
  > C-MMLU is a benchmark designed to evaluate the performance of AI models on Chinese language tasks, including reading comprehension, text classification, and more.
- **任务类别**: `Chinese`, `Knowledge`, `MCQ`
- **评估指标**: `AverageAccuracy`
- **需要LLM Judge**: 否
- **默认提示方式**: 5-shot
- **数据集子集**: `agronomy`, `anatomy`, `ancient_chinese`, `arts`, `astronomy`, `business_ethics`, `chinese_civil_service_exam`, `chinese_driving_rule`, `chinese_food_culture`, `chinese_foreign_policy`, `chinese_history`, `chinese_literature`, `chinese_teacher_qualification`, `clinical_knowledge`, `college_actuarial_science`, `college_education`, `college_engineering_hydrology`, `college_law`, `college_mathematics`, `college_medical_statistics`, `college_medicine`, `computer_science`, `computer_security`, `conceptual_physics`, `construction_project_management`, `economics`, `education`, `electrical_engineering`, `elementary_chinese`, `elementary_commonsense`, `elementary_information_and_technology`, `elementary_mathematics`, `ethnology`, `food_science`, `genetics`, `global_facts`, `high_school_biology`, `high_school_chemistry`, `high_school_geography`, `high_school_mathematics`, `high_school_physics`, `high_school_politics`, `human_sexuality`, `international_law`, `journalism`, `jurisprudence`, `legal_and_moral_basis`, `logical`, `machine_learning`, `management`, `marketing`, `marxist_theory`, `modern_chinese`, `nutrition`, `philosophy`, `professional_accounting`, `professional_law`, `professional_medicine`, `professional_psychology`, `public_relations`, `security_study`, `sociology`, `sports_science`, `traditional_chinese_medicine`, `virology`, `world_history`, `world_religions`

- **支持输出格式**: `generation`, `multiple_choice_logits`
- **提示模板**: 
```text
以下是关于{subset_name}的单项选择题，请给出正确答案的选项。你的回答的最后一行应该是这样的格式：“答案：LETTER”（不带引号），其中 LETTER 是 A、B、C、D 中的一个。
{query}
```

---

### MATH

[返回目录](#llm评测集)
- **数据集名称**: `competition_math`
- **数据集ID**: [modelscope/competition_math](https://modelscope.cn/datasets/modelscope/competition_math/summary)
- **数据集描述**:  
  > The MATH (Mathematics) benchmark is designed to evaluate the mathematical reasoning abilities of AI models through a variety of problem types, including arithmetic, algebra, geometry, and more.
- **任务类别**: `Mathematics`
- **评估指标**: `AveragePass@1`
- **需要LLM Judge**: 否
- **默认提示方式**: 4-shot
- **数据集子集**: `Level 1`, `Level 2`, `Level 3`, `Level 4`, `Level 5`

- **支持输出格式**: `generation`
- **提示模板**: 
```text
{query}
Please reason step by step, and put your final answer within \boxed{{}}.
```

---

### DocMath

[返回目录](#llm评测集)
- **数据集名称**: `docmath`
- **数据集ID**: [yale-nlp/DocMath-Eval](https://modelscope.cn/datasets/yale-nlp/DocMath-Eval/summary)
- **数据集描述**:  
  > DocMath-Eval is a comprehensive benchmark focused on numerical reasoning within specialized domains. It requires the model to comprehend long and specialized documents and perform numerical reasoning to answer the given question.
- **任务类别**: `Long Context`, `Mathematics`, `Reasoning`
- **评估指标**: `AverageAccuracy`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `complong_testmini`, `compshort_testmini`, `simplong_testmini`, `simpshort_testmini`

- **支持输出格式**: `generation`
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
  > The DROP (Discrete Reasoning Over Paragraphs) benchmark is designed to evaluate the reading comprehension and reasoning capabilities of AI models. It includes a variety of tasks that require models to read passages and answer questions based on the content.
- **任务类别**: `Reasoning`
- **评估指标**: `AverageAccuracy`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `default`

- **支持输出格式**: `generation`
- **提示模板**: 
```text
You will be asked to read a passage and answer a question.{drop_examples}# Your Task

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
  > FRAMES is a comprehensive evaluation dataset designed to test the capabilities of Retrieval-Augmented Generation (RAG) systems across factuality, retrieval accuracy, and reasoning.
- **任务类别**: `Long Context`, `Reasoning`
- **评估指标**: `AverageAccuracy`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `default`

- **支持输出格式**: `generation`
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
  > GeneralArena is a custom benchmark designed to evaluate the performance of large language models in a competitive setting, where models are pitted against each other in custom tasks to determine their relative strengths and weaknesses. You should provide the model outputs in the format of a list of dictionaries, where each dictionary contains the model name and its report path. For detailed instructions on how to use this benchmark, please refer to the [Arena User Guide](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/arena.html).
- **任务类别**: `Arena`, `Custom`
- **评估指标**: `winrate`
- **需要LLM Judge**: 是
- **默认提示方式**: 0-shot
- **数据集子集**: `default`

- **支持输出格式**: `generation`
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
  > A general multiple-choice question answering dataset for custom evaluation. For detailed instructions on how to use this benchmark, please refer to the [User Guide](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/custom_dataset/llm.html#mcq).
- **任务类别**: `Custom`, `MCQ`
- **评估指标**: `AverageAccuracy`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `default`

- **支持输出格式**: `generation`, `multiple_choice_logits`
- **提示模板**: 
```text
请回答问题，并选出其中的正确答案。你的回答的最后一行应该是这样的格式：“答案是：LETTER”（不带引号），其中 LETTER 是 A、B、C、D 中的一个。
{query}
```

---

### General-QA

[返回目录](#llm评测集)
- **数据集名称**: `general_qa`
- **数据集ID**: general_qa
- **数据集描述**:  
  > A general question answering dataset for custom evaluation. For detailed instructions on how to use this benchmark, please refer to the [User Guide](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/custom_dataset/llm.html#qa).
- **任务类别**: `Custom`, `QA`
- **评估指标**: `AverageBLEU`, `AverageRouge`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `default`

- **支持输出格式**: `generation`
- **提示模板**: 
```text
请回答问题
{query}
```

---

### GPQA

[返回目录](#llm评测集)
- **数据集名称**: `gpqa`
- **数据集ID**: [modelscope/gpqa](https://modelscope.cn/datasets/modelscope/gpqa/summary)
- **数据集描述**:  
  > GPQA is a dataset for evaluating the reasoning ability of large language models (LLMs) on complex mathematical problems. It contains questions that require step-by-step reasoning to arrive at the correct answer.
- **任务类别**: `Knowledge`, `MCQ`
- **评估指标**: `AveragePass@1`
- **需要LLM Judge**: 否
- **默认提示方式**: 5-shot
- **数据集子集**: `gpqa_diamond`, `gpqa_extended`, `gpqa_main`

- **支持输出格式**: `generation`, `multiple_choice_logits`
- **提示模板**: 
```text
{query}
Please reason step by step, and put your final answer within \boxed{{}}.
```

---

### GSM8K

[返回目录](#llm评测集)
- **数据集名称**: `gsm8k`
- **数据集ID**: [modelscope/gsm8k](https://modelscope.cn/datasets/modelscope/gsm8k/summary)
- **数据集描述**:  
  > GSM8K (Grade School Math 8K) is a dataset of grade school math problems, designed to evaluate the mathematical reasoning abilities of AI models.
- **任务类别**: `Mathematics`
- **评估指标**: `AverageAccuracy`
- **需要LLM Judge**: 否
- **默认提示方式**: 4-shot
- **数据集子集**: `main`

- **支持输出格式**: `generation`
- **提示模板**: 
```text
Question: {query}
Let's think step by step
Answer:
```

---

### HellaSwag

[返回目录](#llm评测集)
- **数据集名称**: `hellaswag`
- **数据集ID**: [modelscope/hellaswag](https://modelscope.cn/datasets/modelscope/hellaswag/summary)
- **数据集描述**:  
  > HellaSwag is a benchmark for commonsense reasoning in natural language understanding tasks. It consists of multiple-choice questions where the model must select the most plausible continuation of a given context.
- **任务类别**: `Commonsense`, `Knowledge`, `MCQ`
- **评估指标**: `AverageAccuracy`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `default`

- **支持输出格式**: `generation`, `multiple_choice_logits`
- **提示模板**: 
```text
{query}
```

---

### HumanEval

[返回目录](#llm评测集)
- **数据集名称**: `humaneval`
- **数据集ID**: [modelscope/humaneval](https://modelscope.cn/datasets/modelscope/humaneval/summary)
- **数据集描述**:  
  > HumanEval is a benchmark for evaluating the ability of code generation models to write Python functions based on given specifications. It consists of programming tasks with a defined input-output behavior.
- **任务类别**: `Coding`
- **评估指标**: `Pass@1`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `openai_humaneval`

- **支持输出格式**: `generation`
- **额外参数**: 
```json
{
    "num_workers": 4,
    "timeout": 4
}
```
- **提示模板**: 
```text
Complete the following python code:
{query}
```

---

### IFEval

[返回目录](#llm评测集)
- **数据集名称**: `ifeval`
- **数据集ID**: [opencompass/ifeval](https://modelscope.cn/datasets/opencompass/ifeval/summary)
- **数据集描述**:  
  > IFEval is a benchmark for evaluating instruction-following language models, focusing on their ability to understand and respond to various prompts. It includes a diverse set of tasks and metrics to assess model performance comprehensively.
- **任务类别**: `Instruction-Following`
- **评估指标**: `inst_level_loose_acc`, `inst_level_strict_acc`, `prompt_level_loose_acc`, `prompt_level_strict_acc`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `default`

- **支持输出格式**: `generation`

---

### IQuiz

[返回目录](#llm评测集)
- **数据集名称**: `iquiz`
- **数据集ID**: [AI-ModelScope/IQuiz](https://modelscope.cn/datasets/AI-ModelScope/IQuiz/summary)
- **数据集描述**:  
  > IQuiz is a benchmark for evaluating AI models on IQ and EQ questions. It consists of multiple-choice questions where the model must select the correct answer and provide an explanation.
- **任务类别**: `Chinese`, `Knowledge`, `MCQ`
- **评估指标**: `AverageAccuracy`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `EQ`, `IQ`

- **支持输出格式**: `generation`, `multiple_choice_logits`
- **系统提示词**: 
```text
你是一个高智商和高情商的专家，你被要求回答一个选择题，并选出一个正确的选项，解释原因，最终输出格式为：`答案是(选项)`。
```

---

### Live-Code-Bench

[返回目录](#llm评测集)
- **数据集名称**: `live_code_bench`
- **数据集ID**: [AI-ModelScope/code_generation_lite](https://modelscope.cn/datasets/AI-ModelScope/code_generation_lite/summary)
- **数据集描述**:  
  > Live Code Bench is a benchmark for evaluating code generation models on real-world coding tasks. It includes a variety of programming problems with test cases to assess the model's ability to generate correct and efficient code solutions.
- **任务类别**: `Coding`
- **评估指标**: `Pass@1`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `release_latest`

- **支持输出格式**: `generation`
- **额外参数**: 
```json
{
    "start_date": null,
    "end_date": null,
    "timeout": 6,
    "debug": false
}
```
- **系统提示词**: 
```text
You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.
```
- **提示模板**: 
```text
### Question:
{question_content}

{format_prompt} ### Answer: (use the provided format with backticks)


```

---

### MaritimeBench

[返回目录](#llm评测集)
- **数据集名称**: `maritime_bench`
- **数据集ID**: [HiDolphin/MaritimeBench](https://modelscope.cn/datasets/HiDolphin/MaritimeBench/summary)
- **数据集描述**:  
  > MaritimeBench is a benchmark for evaluating AI models on maritime-related multiple-choice questions. It consists of questions related to maritime knowledge, where the model must select the correct answer from given options.
- **任务类别**: `Knowledge`, `MCQ`, `Maritime`
- **评估指标**: `AverageAccuracy`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `default`

- **支持输出格式**: `generation`, `multiple_choice_logits`
- **提示模板**: 
```text
题目来自于{subset_name}请回答单选题。要求只输出选项，不输出解释，将选项放在<>里，直接输出答案。示例：

题目：在船舶主推进动力装置中，传动轴系在运转中承受以下复杂的应力和负荷，但不包括______。
选项：
A. 电磁力
B. 压拉应力
C. 弯曲应力
D. 扭应力
答：<A> 当前题目
 {query}
```

---

### MATH-500

[返回目录](#llm评测集)
- **数据集名称**: `math_500`
- **数据集ID**: [AI-ModelScope/MATH-500](https://modelscope.cn/datasets/AI-ModelScope/MATH-500/summary)
- **数据集描述**:  
  > MATH-500 is a benchmark for evaluating mathematical reasoning capabilities of AI models. It consists of 500 diverse math problems across five levels of difficulty, designed to test a model's ability to solve complex mathematical problems by generating step-by-step solutions and providing the correct final answer.
- **任务类别**: `Mathematics`
- **评估指标**: `AveragePass@1`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `Level 1`, `Level 2`, `Level 3`, `Level 4`, `Level 5`

- **支持输出格式**: `generation`
- **提示模板**: 
```text
{query}
Please reason step by step, and put your final answer within \boxed{{}}.
```

---

### MMLU

[返回目录](#llm评测集)
- **数据集名称**: `mmlu`
- **数据集ID**: [modelscope/mmlu](https://modelscope.cn/datasets/modelscope/mmlu/summary)
- **数据集描述**:  
  > The MMLU (Massive Multitask Language Understanding) benchmark is a comprehensive evaluation suite designed to assess the performance of language models across a wide range of subjects and tasks. It includes multiple-choice questions from various domains, such as history, science, mathematics, and more, providing a robust measure of a model's understanding and reasoning capabilities.
- **任务类别**: `Knowledge`, `MCQ`
- **评估指标**: `AverageAccuracy`
- **需要LLM Judge**: 否
- **默认提示方式**: 5-shot
- **数据集子集**: `abstract_algebra`, `anatomy`, `astronomy`, `business_ethics`, `clinical_knowledge`, `college_biology`, `college_chemistry`, `college_computer_science`, `college_mathematics`, `college_medicine`, `college_physics`, `computer_security`, `conceptual_physics`, `econometrics`, `electrical_engineering`, `elementary_mathematics`, `formal_logic`, `global_facts`, `high_school_biology`, `high_school_chemistry`, `high_school_computer_science`, `high_school_european_history`, `high_school_geography`, `high_school_government_and_politics`, `high_school_macroeconomics`, `high_school_mathematics`, `high_school_microeconomics`, `high_school_physics`, `high_school_psychology`, `high_school_statistics`, `high_school_us_history`, `high_school_world_history`, `human_aging`, `human_sexuality`, `international_law`, `jurisprudence`, `logical_fallacies`, `machine_learning`, `management`, `marketing`, `medical_genetics`, `miscellaneous`, `moral_disputes`, `moral_scenarios`, `nutrition`, `philosophy`, `prehistory`, `professional_accounting`, `professional_law`, `professional_medicine`, `professional_psychology`, `public_relations`, `security_studies`, `sociology`, `us_foreign_policy`, `virology`, `world_religions`

- **支持输出格式**: `generation`, `multiple_choice_logits`
- **提示模板**: 
```text
Answer the following multiple choice question about {subset_name}. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{query}
```

---

### MMLU-Pro

[返回目录](#llm评测集)
- **数据集名称**: `mmlu_pro`
- **数据集ID**: [modelscope/MMLU-Pro](https://modelscope.cn/datasets/modelscope/MMLU-Pro/summary)
- **数据集描述**:  
  > MMLU-Pro is a benchmark for evaluating language models on multiple-choice questions across various subjects. It includes questions from different domains, where the model must select the correct answer from given options.
- **任务类别**: `Knowledge`, `MCQ`
- **评估指标**: `AverageAccuracy`
- **需要LLM Judge**: 否
- **默认提示方式**: 5-shot
- **数据集子集**: `biology`, `business`, `chemistry`, `computer science`, `economics`, `engineering`, `health`, `history`, `law`, `math`, `other`, `philosophy`, `physics`, `psychology`

- **支持输出格式**: `generation`, `multiple_choice_logits`
- **提示模板**: 
```text
The following are multiple choice questions (with answers) about {subset_name}. Think step by step and then finish your answer with "the answer is (X)" where X is the correct letter choice.
{query}
```

---

### MMLU-Redux

[返回目录](#llm评测集)
- **数据集名称**: `mmlu_redux`
- **数据集ID**: [AI-ModelScope/mmlu-redux-2.0](https://modelscope.cn/datasets/AI-ModelScope/mmlu-redux-2.0/summary)
- **数据集描述**:  
  > MMLU-Redux is a benchmark for evaluating language models on multiple-choice questions across various subjects. It includes questions from different domains, where the model must select the correct answer from given options.
- **任务类别**: `Knowledge`, `MCQ`
- **评估指标**: `AverageAccuracy`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `abstract_algebra`, `anatomy`, `astronomy`, `business_ethics`, `clinical_knowledge`, `college_biology`, `college_chemistry`, `college_computer_science`, `college_mathematics`, `college_medicine`, `college_physics`, `computer_security`, `conceptual_physics`, `econometrics`, `electrical_engineering`, `elementary_mathematics`, `formal_logic`, `global_facts`, `high_school_biology`, `high_school_chemistry`, `high_school_computer_science`, `high_school_european_history`, `high_school_geography`, `high_school_government_and_politics`, `high_school_macroeconomics`, `high_school_mathematics`, `high_school_microeconomics`, `high_school_physics`, `high_school_psychology`, `high_school_statistics`, `high_school_us_history`, `high_school_world_history`, `human_aging`, `human_sexuality`, `international_law`, `jurisprudence`, `logical_fallacies`, `machine_learning`, `management`, `marketing`, `medical_genetics`, `miscellaneous`, `moral_disputes`, `moral_scenarios`, `nutrition`, `philosophy`, `prehistory`, `professional_accounting`, `professional_law`, `professional_medicine`, `professional_psychology`, `public_relations`, `security_studies`, `sociology`, `us_foreign_policy`, `virology`, `world_religions`

- **支持输出格式**: `generation`, `multiple_choice_logits`
- **提示模板**: 
```text
The following are multiple choice questions (with answers) about {subset_name}. Think step by step and then finish your answer with "the answer is (X)" where X is the correct letter choice.
{query}
```

---

### MuSR

[返回目录](#llm评测集)
- **数据集名称**: `musr`
- **数据集ID**: [AI-ModelScope/MuSR](https://modelscope.cn/datasets/AI-ModelScope/MuSR/summary)
- **数据集描述**:  
  > MuSR is a benchmark for evaluating AI models on multiple-choice questions related to murder mysteries, object placements, and team allocation.
- **任务类别**: `MCQ`, `Reasoning`
- **评估指标**: `AverageAccuracy`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `murder_mysteries`, `object_placements`, `team_allocation`

- **支持输出格式**: `generation`, `multiple_choice_logits`
- **提示模板**: 
```text
{narrative}

{question}

{choices}
Think step by step and then finish your answer with "the answer is (X)" where X is the correct letter choice.
```

---

### Needle-in-a-Haystack

[返回目录](#llm评测集)
- **数据集名称**: `needle_haystack`
- **数据集ID**: [AI-ModelScope/Needle-in-a-Haystack-Corpus](https://modelscope.cn/datasets/AI-ModelScope/Needle-in-a-Haystack-Corpus/summary)
- **数据集描述**:  
  > Needle in a Haystack is a benchmark focused on information retrieval tasks. It requires the model to find specific information within a large corpus of text. [Usage Example](https://evalscope.readthedocs.io/zh-cn/latest/third_party/needle_haystack.html)
- **任务类别**: `Long Context`, `Retrieval`
- **评估指标**: `AverageAccuracy`
- **需要LLM Judge**: 是
- **默认提示方式**: 0-shot
- **数据集子集**: `chinese`, `english`

- **支持输出格式**: `generation`
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

### ProcessBench

[返回目录](#llm评测集)
- **数据集名称**: `process_bench`
- **数据集ID**: [Qwen/ProcessBench](https://modelscope.cn/datasets/Qwen/ProcessBench/summary)
- **数据集描述**:  
  > ProcessBench is a benchmark for evaluating AI models on mathematical reasoning tasks. It includes various subsets such as GSM8K, Math, OlympiadBench, and OmniMath, each with its own set of problems that require step-by-step reasoning to arrive at the correct answer.
- **任务类别**: `Mathematical`, `Reasoning`
- **评估指标**: `correct_acc`, `error_acc`, `simple_f1_score`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `gsm8k`, `math`, `olympiadbench`, `omnimath`

- **支持输出格式**: `generation`
- **提示模板**: 
```text
The following is a math problem and a solution (split into paragraphs, enclosed with tags and indexed from 0):

[Math Problem]

{problem}

[Solution]

{tagged_response}

Your task is to review and critique the solution paragraph by paragraph. Once you identify an error in a paragraph, return the index of the paragraph where the earliest error occurs. Otherwise, return the index of -1 (which typically denotes "not found").

Please put your final answer (i.e., the index) in \boxed{{}}.

```

---

### RACE

[返回目录](#llm评测集)
- **数据集名称**: `race`
- **数据集ID**: [modelscope/race](https://modelscope.cn/datasets/modelscope/race/summary)
- **数据集描述**:  
  > RACE is a benchmark for testing reading comprehension and reasoning abilities of neural models. It is constructed from Chinese middle and high school examinations.
- **任务类别**: `MCQ`, `Reasoning`
- **评估指标**: `AverageAccuracy`
- **需要LLM Judge**: 否
- **默认提示方式**: 3-shot
- **数据集子集**: `high`, `middle`

- **支持输出格式**: `generation`, `multiple_choice_logits`

---

### SimpleQA

[返回目录](#llm评测集)
- **数据集名称**: `simple_qa`
- **数据集ID**: [AI-ModelScope/SimpleQA](https://modelscope.cn/datasets/AI-ModelScope/SimpleQA/summary)
- **数据集描述**:  
  > SimpleQA is a benchmark designed to evaluate the performance of language models on simple question-answering tasks. It includes a set of straightforward questions that require basic reasoning and understanding capabilities.
- **任务类别**: `Knowledge`, `QA`
- **评估指标**: `is_correct`, `is_incorrect`, `is_not_attempted`
- **需要LLM Judge**: 是
- **默认提示方式**: 0-shot
- **数据集子集**: `default`

- **支持输出格式**: `generation`

---

### SuperGPQA

[返回目录](#llm评测集)
- **数据集名称**: `super_gpqa`
- **数据集ID**: [m-a-p/SuperGPQA](https://modelscope.cn/datasets/m-a-p/SuperGPQA/summary)
- **数据集描述**:  
  > SuperGPQA is a large-scale multiple-choice question answering dataset, designed to evaluate the generalization ability of models across different fields. It contains 100,000+ questions from 50+ fields, with each question having 10 options.
- **任务类别**: `Knowledge`, `MCQ`
- **评估指标**: `AverageAccuracy`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `Aeronautical and Astronautical Science and Technology`, `Agricultural Engineering`, `Animal Husbandry`, `Applied Economics`, `Aquaculture`, `Architecture`, `Art Studies`, `Astronomy`, `Atmospheric Science`, `Basic Medicine`, `Biology`, `Business Administration`, `Chemical Engineering and Technology`, `Chemistry`, `Civil Engineering`, `Clinical Medicine`, `Computer Science and Technology`, `Control Science and Engineering`, `Crop Science`, `Education`, `Electrical Engineering`, `Electronic Science and Technology`, `Environmental Science and Engineering`, `Food Science and Engineering`, `Forestry Engineering`, `Forestry`, `Geography`, `Geological Resources and Geological Engineering`, `Geology`, `Geophysics`, `History`, `Hydraulic Engineering`, `Information and Communication Engineering`, `Instrument Science and Technology`, `Journalism and Communication`, `Language and Literature`, `Law`, `Library, Information and Archival Management`, `Management Science and Engineering`, `Materials Science and Engineering`, `Mathematics`, `Mechanical Engineering`, `Mechanics`, `Metallurgical Engineering`, `Military Science`, `Mining Engineering`, `Musicology`, `Naval Architecture and Ocean Engineering`, `Nuclear Science and Technology`, `Oceanography`, `Optical Engineering`, `Petroleum and Natural Gas Engineering`, `Pharmacy`, `Philosophy`, `Physical Education`, `Physical Oceanography`, `Physics`, `Political Science`, `Power Engineering and Engineering Thermophysics`, `Psychology`, `Public Administration`, `Public Health and Preventive Medicine`, `Sociology`, `Stomatology`, `Surveying and Mapping Science and Technology`, `Systems Science`, `Textile Science and Engineering`, `Theoretical Economics`, `Traditional Chinese Medicine`, `Transportation Engineering`, `Veterinary Medicine`, `Weapon Science and Technology`

- **支持输出格式**: `generation`, `multiple_choice_logits`

---

### ToolBench-Static

[返回目录](#llm评测集)
- **数据集名称**: `tool_bench`
- **数据集ID**: [AI-ModelScope/ToolBench-Static](https://modelscope.cn/datasets/AI-ModelScope/ToolBench-Static/summary)
- **数据集描述**:  
  > ToolBench is a benchmark for evaluating AI models on tool use tasks. It includes various subsets such as in-domain and out-of-domain, each with its own set of problems that require step-by-step reasoning to arrive at the correct answer. [Usage Example](https://evalscope.readthedocs.io/zh-cn/latest/third_party/toolbench.html)
- **任务类别**: `Agent`, `Reasoning`
- **评估指标**: `Act.EM`, `F1`, `HalluRate`, `Plan.EM`, `Rouge-L`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `in_domain`, `out_of_domain`

- **支持输出格式**: `generation`

---

### TriviaQA

[返回目录](#llm评测集)
- **数据集名称**: `trivia_qa`
- **数据集ID**: [modelscope/trivia_qa](https://modelscope.cn/datasets/modelscope/trivia_qa/summary)
- **数据集描述**:  
  > TriviaQA is a large-scale reading comprehension dataset consisting of question-answer pairs collected from trivia websites. It includes questions with multiple possible answers, making it suitable for evaluating the ability of models to understand and generate answers based on context.
- **任务类别**: `QA`, `Reading Comprehension`
- **评估指标**: `AverageAccuracy`
- **需要LLM Judge**: 否
- **默认提示方式**: 5-shot
- **数据集子集**: `default`

- **支持输出格式**: `generation`

---

### TruthfulQA

[返回目录](#llm评测集)
- **数据集名称**: `truthful_qa`
- **数据集ID**: [modelscope/truthful_qa](https://modelscope.cn/datasets/modelscope/truthful_qa/summary)
- **数据集描述**:  
  > TruthfulQA is a benchmark designed to evaluate the ability of AI models to answer questions truthfully and accurately. It includes multiple-choice and generation tasks, focusing on the model's understanding of factual information and its ability to generate coherent responses.
- **任务类别**: `Knowledge`
- **评估指标**: `AverageAccuracy`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `multiple_choice`

- **支持输出格式**: `continuous_logits`, `generation`

---

### Winogrande

[返回目录](#llm评测集)
- **数据集名称**: `winogrande`
- **数据集ID**: [AI-ModelScope/winogrande_val](https://modelscope.cn/datasets/AI-ModelScope/winogrande_val/summary)
- **数据集描述**:  
  > Winogrande is a benchmark for evaluating AI models on commonsense reasoning tasks, specifically designed to test the ability to resolve ambiguous pronouns in sentences.
- **任务类别**: `MCQ`, `Reasoning`
- **评估指标**: `AverageAccuracy`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `default`

- **支持输出格式**: `generation`, `multiple_choice_logits`
- **提示模板**: 
```text
Question: {query}
A. {option1}
B. {option2}
Answer:
```
