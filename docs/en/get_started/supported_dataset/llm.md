# LLM Benchmarks

Below is the list of supported LLM benchmarks. Click on a benchmark name to jump to details.

| Benchmark Name | Pretty Name | Task Categories |
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

## Benchmark Details

### AIME-2024

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `aime24`
- **Dataset ID**: [HuggingFaceH4/aime_2024](https://modelscope.cn/datasets/HuggingFaceH4/aime_2024/summary)
- **Description**:  
  > The AIME 2024 benchmark is based on problems from the American Invitational Mathematics Examination, a prestigious high school mathematics competition. This benchmark tests a model’s ability to solve challenging mathematics problems by generating step-by-step solutions and providing the correct final answer.
- **Task Categories**: `Mathematics`
- **Evaluation Metrics**: `AveragePass@1`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Subsets**: `default`

- **Supported Output Formats**: `generation`
- **Prompt Template**: 
```text
{query}
Please reason step by step, and put your final answer within \boxed{{}}.
```

---

### AIME-2025

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `aime25`
- **Dataset ID**: [opencompass/AIME2025](https://modelscope.cn/datasets/opencompass/AIME2025/summary)
- **Description**:  
  > The AIME 2025 benchmark is based on problems from the American Invitational Mathematics Examination, a prestigious high school mathematics competition. This benchmark tests a model’s ability to solve challenging mathematics problems by generating step-by-step solutions and providing the correct final answer.
- **Task Categories**: `Mathematics`
- **Evaluation Metrics**: `AveragePass@1`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Subsets**: `AIME2025-II`, `AIME2025-I`

- **Supported Output Formats**: `generation`
- **Prompt Template**: 
```text
{query}
Please reason step by step, and put your final answer within \boxed{{}}.
```

---

### AlpacaEval2.0

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `alpaca_eval`
- **Dataset ID**: [AI-ModelScope/alpaca_eval](https://modelscope.cn/datasets/AI-ModelScope/alpaca_eval/summary)
- **Description**:  
  > Alpaca Eval 2.0 is an enhanced framework for evaluating instruction-following language models, featuring an improved auto-annotator, updated baselines, and continuous preference calculation to provide more accurate and cost-effective model assessments. Currently not support `length-controlled winrate`; the official Judge model is `gpt-4-1106-preview`, while the baseline model is `gpt-4-turbo`.
- **Task Categories**: `Arena`, `Instruction-Following`
- **Evaluation Metrics**: `winrate`
- **Requires LLM Judge**: Yes
- **Default Shots**: 0-shot
- **Subsets**: `alpaca_eval_gpt4_baseline`

- **Supported Output Formats**: `generation`

---

### ARC

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `arc`
- **Dataset ID**: [modelscope/ai2_arc](https://modelscope.cn/datasets/modelscope/ai2_arc/summary)
- **Description**:  
  > The ARC (AI2 Reasoning Challenge) benchmark is designed to evaluate the reasoning capabilities of AI models through multiple-choice questions derived from science exams. It includes two subsets: ARC-Easy and ARC-Challenge, which vary in difficulty.
- **Task Categories**: `MCQ`, `Reasoning`
- **Evaluation Metrics**: `AverageAccuracy`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Subsets**: `ARC-Challenge`, `ARC-Easy`

- **Supported Output Formats**: `generation`, `multiple_choice_logits`
- **Prompt Template**: 
```text
Given the following question and four candidate answers (A, B, C and D), choose the best answer.
{query}
Your response should end with "The best answer is [the_answer_letter]" where the [the_answer_letter] is one of A, B, C or D.
```

---

### ArenaHard

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `arena_hard`
- **Dataset ID**: [AI-ModelScope/arena-hard-auto-v0.1](https://modelscope.cn/datasets/AI-ModelScope/arena-hard-auto-v0.1/summary)
- **Description**:  
  > ArenaHard is a benchmark designed to evaluate the performance of large language models in a competitive setting, where models are pitted against each other in a series of tasks to determine their relative strengths and weaknesses. It includes a set of challenging tasks that require reasoning, understanding, and generation capabilities. Currently not support `style-controlled winrate`; the official Judge model is `gpt-4-1106-preview`, while the baseline model is `gpt-4-0314`.
- **Task Categories**: `Arena`, `Instruction-Following`
- **Evaluation Metrics**: `winrate`
- **Requires LLM Judge**: Yes
- **Default Shots**: 0-shot
- **Subsets**: `default`

- **Supported Output Formats**: `generation`

---

### BBH

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `bbh`
- **Dataset ID**: [modelscope/bbh](https://modelscope.cn/datasets/modelscope/bbh/summary)
- **Description**:  
  > The BBH (Big Bench Hard) benchmark is a collection of challenging tasks designed to evaluate the reasoning capabilities of AI models. It includes both free-form and multiple-choice tasks, covering a wide range of reasoning skills.
- **Task Categories**: `Reasoning`
- **Evaluation Metrics**: `AverageAccuracy`
- **Requires LLM Judge**: No
- **Default Shots**: 3-shot
- **Subsets**: `boolean_expressions`, `causal_judgement`, `date_understanding`, `disambiguation_qa`, `dyck_languages`, `formal_fallacies`, `geometric_shapes`, `hyperbaton`, `logical_deduction_five_objects`, `logical_deduction_seven_objects`, `logical_deduction_three_objects`, `movie_recommendation`, `multistep_arithmetic_two`, `navigate`, `object_counting`, `penguins_in_a_table`, `reasoning_about_colored_objects`, `ruin_names`, `salient_translation_error_detection`, `snarks`, `sports_understanding`, `temporal_sequences`, `tracking_shuffled_objects_five_objects`, `tracking_shuffled_objects_seven_objects`, `tracking_shuffled_objects_three_objects`, `web_of_lies`, `word_sorting`

- **Supported Output Formats**: `generation`
- **Prompt Template**: 
```text
Q: {query}
A: Let's think step by step.
```

---

### BFCL-v3

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `bfcl_v3`
- **Dataset ID**: [AI-ModelScope/bfcl_v3](https://modelscope.cn/datasets/AI-ModelScope/bfcl_v3/summary)
- **Description**:  
  > Berkeley Function Calling Leaderboard (BFCL), the **first comprehensive and executable function call evaluation** dedicated to assessing Large Language Models' (LLMs) ability to invoke functions. Unlike previous evaluations, BFCL accounts for various forms of function calls, diverse scenarios, and executability. Need to run `pip install bfcl-eval` before evaluating. [Usage Example](https://evalscope.readthedocs.io/zh-cn/latest/third_party/bfcl_v3.html)
- **Task Categories**: `Agent`
- **Evaluation Metrics**: `AverageAccuracy`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Subsets**: `irrelevance`, `java`, `javascript`, `live_irrelevance`, `live_multiple`, `live_parallel_multiple`, `live_parallel`, `live_relevance`, `live_simple`, `multi_turn_base`, `multi_turn_long_context`, `multi_turn_miss_func`, `multi_turn_miss_param`, `multiple`, `parallel_multiple`, `parallel`, `simple`

- **Supported Output Formats**: `generation`
- **Extra Parameters**: 
```json
{
    "underscore_to_dot": true,
    "is_fc_model": true
}
```

---

### C-Eval

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `ceval`
- **Dataset ID**: [modelscope/ceval-exam](https://modelscope.cn/datasets/modelscope/ceval-exam/summary)
- **Description**:  
  > C-Eval is a benchmark designed to evaluate the performance of AI models on Chinese exams across various subjects, including STEM, social sciences, and humanities. It consists of multiple-choice questions that test knowledge and reasoning abilities in these areas.
- **Task Categories**: `Chinese`, `Knowledge`, `MCQ`
- **Evaluation Metrics**: `AverageAccuracy`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Subsets**: `accountant`, `advanced_mathematics`, `art_studies`, `basic_medicine`, `business_administration`, `chinese_language_and_literature`, `civil_servant`, `clinical_medicine`, `college_chemistry`, `college_economics`, `college_physics`, `college_programming`, `computer_architecture`, `computer_network`, `discrete_mathematics`, `education_science`, `electrical_engineer`, `environmental_impact_assessment_engineer`, `fire_engineer`, `high_school_biology`, `high_school_chemistry`, `high_school_chinese`, `high_school_geography`, `high_school_history`, `high_school_mathematics`, `high_school_physics`, `high_school_politics`, `ideological_and_moral_cultivation`, `law`, `legal_professional`, `logic`, `mao_zedong_thought`, `marxism`, `metrology_engineer`, `middle_school_biology`, `middle_school_chemistry`, `middle_school_geography`, `middle_school_history`, `middle_school_mathematics`, `middle_school_physics`, `middle_school_politics`, `modern_chinese_history`, `operating_system`, `physician`, `plant_protection`, `probability_and_statistics`, `professional_tour_guide`, `sports_science`, `tax_accountant`, `teacher_qualification`, `urban_and_rural_planner`, `veterinary_medicine`

- **Supported Output Formats**: `generation`, `multiple_choice_logits`
- **Prompt Template**: 
```text
以下是中国关于{subset_name}考试的单项选择题，请选出其中的正确答案。你的回答的最后一行应该是这样的格式：“答案是：LETTER”（不带引号），其中 LETTER 是 A、B、C、D 中的一个。
{query}
```

---

### Chinese-SimpleQA

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `chinese_simpleqa`
- **Dataset ID**: [AI-ModelScope/Chinese-SimpleQA](https://modelscope.cn/datasets/AI-ModelScope/Chinese-SimpleQA/summary)
- **Description**:  
  > Chinese SimpleQA is a Chinese question-answering dataset designed to evaluate the performance of language models on simple factual questions. It includes a variety of topics and is structured to test the model's ability to understand and generate correct answers in Chinese.
- **Task Categories**: `Chinese`, `Knowledge`, `QA`
- **Evaluation Metrics**: `is_correct`, `is_incorrect`, `is_not_attempted`
- **Requires LLM Judge**: Yes
- **Default Shots**: 0-shot
- **Subsets**: `中华文化`, `人文与社会科学`, `工程、技术与应用科学`, `生活、艺术与文化`, `社会`, `自然与自然科学`

- **Supported Output Formats**: `generation`

---

### C-MMLU

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `cmmlu`
- **Dataset ID**: [modelscope/cmmlu](https://modelscope.cn/datasets/modelscope/cmmlu/summary)
- **Description**:  
  > C-MMLU is a benchmark designed to evaluate the performance of AI models on Chinese language tasks, including reading comprehension, text classification, and more.
- **Task Categories**: `Chinese`, `Knowledge`, `MCQ`
- **Evaluation Metrics**: `AverageAccuracy`
- **Requires LLM Judge**: No
- **Default Shots**: 5-shot
- **Subsets**: `agronomy`, `anatomy`, `ancient_chinese`, `arts`, `astronomy`, `business_ethics`, `chinese_civil_service_exam`, `chinese_driving_rule`, `chinese_food_culture`, `chinese_foreign_policy`, `chinese_history`, `chinese_literature`, `chinese_teacher_qualification`, `clinical_knowledge`, `college_actuarial_science`, `college_education`, `college_engineering_hydrology`, `college_law`, `college_mathematics`, `college_medical_statistics`, `college_medicine`, `computer_science`, `computer_security`, `conceptual_physics`, `construction_project_management`, `economics`, `education`, `electrical_engineering`, `elementary_chinese`, `elementary_commonsense`, `elementary_information_and_technology`, `elementary_mathematics`, `ethnology`, `food_science`, `genetics`, `global_facts`, `high_school_biology`, `high_school_chemistry`, `high_school_geography`, `high_school_mathematics`, `high_school_physics`, `high_school_politics`, `human_sexuality`, `international_law`, `journalism`, `jurisprudence`, `legal_and_moral_basis`, `logical`, `machine_learning`, `management`, `marketing`, `marxist_theory`, `modern_chinese`, `nutrition`, `philosophy`, `professional_accounting`, `professional_law`, `professional_medicine`, `professional_psychology`, `public_relations`, `security_study`, `sociology`, `sports_science`, `traditional_chinese_medicine`, `virology`, `world_history`, `world_religions`

- **Supported Output Formats**: `generation`, `multiple_choice_logits`
- **Prompt Template**: 
```text
以下是关于{subset_name}的单项选择题，请给出正确答案的选项。你的回答的最后一行应该是这样的格式：“答案：LETTER”（不带引号），其中 LETTER 是 A、B、C、D 中的一个。
{query}
```

---

### MATH

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `competition_math`
- **Dataset ID**: [modelscope/competition_math](https://modelscope.cn/datasets/modelscope/competition_math/summary)
- **Description**:  
  > The MATH (Mathematics) benchmark is designed to evaluate the mathematical reasoning abilities of AI models through a variety of problem types, including arithmetic, algebra, geometry, and more.
- **Task Categories**: `Mathematics`
- **Evaluation Metrics**: `AveragePass@1`
- **Requires LLM Judge**: No
- **Default Shots**: 4-shot
- **Subsets**: `Level 1`, `Level 2`, `Level 3`, `Level 4`, `Level 5`

- **Supported Output Formats**: `generation`
- **Prompt Template**: 
```text
{query}
Please reason step by step, and put your final answer within \boxed{{}}.
```

---

### DocMath

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `docmath`
- **Dataset ID**: [yale-nlp/DocMath-Eval](https://modelscope.cn/datasets/yale-nlp/DocMath-Eval/summary)
- **Description**:  
  > DocMath-Eval is a comprehensive benchmark focused on numerical reasoning within specialized domains. It requires the model to comprehend long and specialized documents and perform numerical reasoning to answer the given question.
- **Task Categories**: `Long Context`, `Mathematics`, `Reasoning`
- **Evaluation Metrics**: `AverageAccuracy`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Subsets**: `complong_testmini`, `compshort_testmini`, `simplong_testmini`, `simpshort_testmini`

- **Supported Output Formats**: `generation`
- **Prompt Template**: 
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

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `drop`
- **Dataset ID**: [AI-ModelScope/DROP](https://modelscope.cn/datasets/AI-ModelScope/DROP/summary)
- **Description**:  
  > The DROP (Discrete Reasoning Over Paragraphs) benchmark is designed to evaluate the reading comprehension and reasoning capabilities of AI models. It includes a variety of tasks that require models to read passages and answer questions based on the content.
- **Task Categories**: `Reasoning`
- **Evaluation Metrics**: `AverageAccuracy`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Subsets**: `default`

- **Supported Output Formats**: `generation`
- **Prompt Template**: 
```text
You will be asked to read a passage and answer a question.{drop_examples}# Your Task

---
{query}

Think step by step, then write a line of the form "Answer: $ANSWER" at the end of your response.
```

---

### FRAMES

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `frames`
- **Dataset ID**: [iic/frames](https://modelscope.cn/datasets/iic/frames/summary)
- **Description**:  
  > FRAMES is a comprehensive evaluation dataset designed to test the capabilities of Retrieval-Augmented Generation (RAG) systems across factuality, retrieval accuracy, and reasoning.
- **Task Categories**: `Long Context`, `Reasoning`
- **Evaluation Metrics**: `AverageAccuracy`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Subsets**: `default`

- **Supported Output Formats**: `generation`
- **Prompt Template**: 
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

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `general_arena`
- **Dataset ID**: general_arena
- **Description**:  
  > GeneralArena is a custom benchmark designed to evaluate the performance of large language models in a competitive setting, where models are pitted against each other in custom tasks to determine their relative strengths and weaknesses. You should provide the model outputs in the format of a list of dictionaries, where each dictionary contains the model name and its report path. For detailed instructions on how to use this benchmark, please refer to the [Arena User Guide](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/arena.html).
- **Task Categories**: `Arena`, `Custom`
- **Evaluation Metrics**: `winrate`
- **Requires LLM Judge**: Yes
- **Default Shots**: 0-shot
- **Subsets**: `default`

- **Supported Output Formats**: `generation`
- **Extra Parameters**: 
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
- **System Prompt**: 
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
- **Prompt Template**: 
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

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `general_mcq`
- **Dataset ID**: general_mcq
- **Description**:  
  > A general multiple-choice question answering dataset for custom evaluation. For detailed instructions on how to use this benchmark, please refer to the [User Guide](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/custom_dataset/llm.html#mcq).
- **Task Categories**: `Custom`, `MCQ`
- **Evaluation Metrics**: `AverageAccuracy`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Subsets**: `default`

- **Supported Output Formats**: `generation`, `multiple_choice_logits`
- **Prompt Template**: 
```text
请回答问题，并选出其中的正确答案。你的回答的最后一行应该是这样的格式：“答案是：LETTER”（不带引号），其中 LETTER 是 A、B、C、D 中的一个。
{query}
```

---

### General-QA

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `general_qa`
- **Dataset ID**: general_qa
- **Description**:  
  > A general question answering dataset for custom evaluation. For detailed instructions on how to use this benchmark, please refer to the [User Guide](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/custom_dataset/llm.html#qa).
- **Task Categories**: `Custom`, `QA`
- **Evaluation Metrics**: `AverageBLEU`, `AverageRouge`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Subsets**: `default`

- **Supported Output Formats**: `generation`
- **Prompt Template**: 
```text
请回答问题
{query}
```

---

### GPQA

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `gpqa`
- **Dataset ID**: [modelscope/gpqa](https://modelscope.cn/datasets/modelscope/gpqa/summary)
- **Description**:  
  > GPQA is a dataset for evaluating the reasoning ability of large language models (LLMs) on complex mathematical problems. It contains questions that require step-by-step reasoning to arrive at the correct answer.
- **Task Categories**: `Knowledge`, `MCQ`
- **Evaluation Metrics**: `AveragePass@1`
- **Requires LLM Judge**: No
- **Default Shots**: 5-shot
- **Subsets**: `gpqa_diamond`, `gpqa_extended`, `gpqa_main`

- **Supported Output Formats**: `generation`, `multiple_choice_logits`
- **Prompt Template**: 
```text
{query}
Please reason step by step, and put your final answer within \boxed{{}}.
```

---

### GSM8K

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `gsm8k`
- **Dataset ID**: [modelscope/gsm8k](https://modelscope.cn/datasets/modelscope/gsm8k/summary)
- **Description**:  
  > GSM8K (Grade School Math 8K) is a dataset of grade school math problems, designed to evaluate the mathematical reasoning abilities of AI models.
- **Task Categories**: `Mathematics`
- **Evaluation Metrics**: `AverageAccuracy`
- **Requires LLM Judge**: No
- **Default Shots**: 4-shot
- **Subsets**: `main`

- **Supported Output Formats**: `generation`
- **Prompt Template**: 
```text
Question: {query}
Let's think step by step
Answer:
```

---

### HellaSwag

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `hellaswag`
- **Dataset ID**: [modelscope/hellaswag](https://modelscope.cn/datasets/modelscope/hellaswag/summary)
- **Description**:  
  > HellaSwag is a benchmark for commonsense reasoning in natural language understanding tasks. It consists of multiple-choice questions where the model must select the most plausible continuation of a given context.
- **Task Categories**: `Commonsense`, `Knowledge`, `MCQ`
- **Evaluation Metrics**: `AverageAccuracy`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Subsets**: `default`

- **Supported Output Formats**: `generation`, `multiple_choice_logits`
- **Prompt Template**: 
```text
{query}
```

---

### HumanEval

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `humaneval`
- **Dataset ID**: [modelscope/humaneval](https://modelscope.cn/datasets/modelscope/humaneval/summary)
- **Description**:  
  > HumanEval is a benchmark for evaluating the ability of code generation models to write Python functions based on given specifications. It consists of programming tasks with a defined input-output behavior.
- **Task Categories**: `Coding`
- **Evaluation Metrics**: `Pass@1`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Subsets**: `openai_humaneval`

- **Supported Output Formats**: `generation`
- **Extra Parameters**: 
```json
{
    "num_workers": 4,
    "timeout": 4
}
```
- **Prompt Template**: 
```text
Complete the following python code:
{query}
```

---

### IFEval

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `ifeval`
- **Dataset ID**: [opencompass/ifeval](https://modelscope.cn/datasets/opencompass/ifeval/summary)
- **Description**:  
  > IFEval is a benchmark for evaluating instruction-following language models, focusing on their ability to understand and respond to various prompts. It includes a diverse set of tasks and metrics to assess model performance comprehensively.
- **Task Categories**: `Instruction-Following`
- **Evaluation Metrics**: `inst_level_loose_acc`, `inst_level_strict_acc`, `prompt_level_loose_acc`, `prompt_level_strict_acc`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Subsets**: `default`

- **Supported Output Formats**: `generation`

---

### IQuiz

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `iquiz`
- **Dataset ID**: [AI-ModelScope/IQuiz](https://modelscope.cn/datasets/AI-ModelScope/IQuiz/summary)
- **Description**:  
  > IQuiz is a benchmark for evaluating AI models on IQ and EQ questions. It consists of multiple-choice questions where the model must select the correct answer and provide an explanation.
- **Task Categories**: `Chinese`, `Knowledge`, `MCQ`
- **Evaluation Metrics**: `AverageAccuracy`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Subsets**: `EQ`, `IQ`

- **Supported Output Formats**: `generation`, `multiple_choice_logits`
- **System Prompt**: 
```text
你是一个高智商和高情商的专家，你被要求回答一个选择题，并选出一个正确的选项，解释原因，最终输出格式为：`答案是(选项)`。
```

---

### Live-Code-Bench

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `live_code_bench`
- **Dataset ID**: [AI-ModelScope/code_generation_lite](https://modelscope.cn/datasets/AI-ModelScope/code_generation_lite/summary)
- **Description**:  
  > Live Code Bench is a benchmark for evaluating code generation models on real-world coding tasks. It includes a variety of programming problems with test cases to assess the model's ability to generate correct and efficient code solutions.
- **Task Categories**: `Coding`
- **Evaluation Metrics**: `Pass@1`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Subsets**: `release_latest`

- **Supported Output Formats**: `generation`
- **Extra Parameters**: 
```json
{
    "start_date": null,
    "end_date": null,
    "timeout": 6,
    "debug": false
}
```
- **System Prompt**: 
```text
You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.
```
- **Prompt Template**: 
```text
### Question:
{question_content}

{format_prompt} ### Answer: (use the provided format with backticks)


```

---

### MaritimeBench

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `maritime_bench`
- **Dataset ID**: [HiDolphin/MaritimeBench](https://modelscope.cn/datasets/HiDolphin/MaritimeBench/summary)
- **Description**:  
  > MaritimeBench is a benchmark for evaluating AI models on maritime-related multiple-choice questions. It consists of questions related to maritime knowledge, where the model must select the correct answer from given options.
- **Task Categories**: `Knowledge`, `MCQ`, `Maritime`
- **Evaluation Metrics**: `AverageAccuracy`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Subsets**: `default`

- **Supported Output Formats**: `generation`, `multiple_choice_logits`
- **Prompt Template**: 
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

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `math_500`
- **Dataset ID**: [AI-ModelScope/MATH-500](https://modelscope.cn/datasets/AI-ModelScope/MATH-500/summary)
- **Description**:  
  > MATH-500 is a benchmark for evaluating mathematical reasoning capabilities of AI models. It consists of 500 diverse math problems across five levels of difficulty, designed to test a model's ability to solve complex mathematical problems by generating step-by-step solutions and providing the correct final answer.
- **Task Categories**: `Mathematics`
- **Evaluation Metrics**: `AveragePass@1`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Subsets**: `Level 1`, `Level 2`, `Level 3`, `Level 4`, `Level 5`

- **Supported Output Formats**: `generation`
- **Prompt Template**: 
```text
{query}
Please reason step by step, and put your final answer within \boxed{{}}.
```

---

### MMLU

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `mmlu`
- **Dataset ID**: [modelscope/mmlu](https://modelscope.cn/datasets/modelscope/mmlu/summary)
- **Description**:  
  > The MMLU (Massive Multitask Language Understanding) benchmark is a comprehensive evaluation suite designed to assess the performance of language models across a wide range of subjects and tasks. It includes multiple-choice questions from various domains, such as history, science, mathematics, and more, providing a robust measure of a model's understanding and reasoning capabilities.
- **Task Categories**: `Knowledge`, `MCQ`
- **Evaluation Metrics**: `AverageAccuracy`
- **Requires LLM Judge**: No
- **Default Shots**: 5-shot
- **Subsets**: `abstract_algebra`, `anatomy`, `astronomy`, `business_ethics`, `clinical_knowledge`, `college_biology`, `college_chemistry`, `college_computer_science`, `college_mathematics`, `college_medicine`, `college_physics`, `computer_security`, `conceptual_physics`, `econometrics`, `electrical_engineering`, `elementary_mathematics`, `formal_logic`, `global_facts`, `high_school_biology`, `high_school_chemistry`, `high_school_computer_science`, `high_school_european_history`, `high_school_geography`, `high_school_government_and_politics`, `high_school_macroeconomics`, `high_school_mathematics`, `high_school_microeconomics`, `high_school_physics`, `high_school_psychology`, `high_school_statistics`, `high_school_us_history`, `high_school_world_history`, `human_aging`, `human_sexuality`, `international_law`, `jurisprudence`, `logical_fallacies`, `machine_learning`, `management`, `marketing`, `medical_genetics`, `miscellaneous`, `moral_disputes`, `moral_scenarios`, `nutrition`, `philosophy`, `prehistory`, `professional_accounting`, `professional_law`, `professional_medicine`, `professional_psychology`, `public_relations`, `security_studies`, `sociology`, `us_foreign_policy`, `virology`, `world_religions`

- **Supported Output Formats**: `generation`, `multiple_choice_logits`
- **Prompt Template**: 
```text
Answer the following multiple choice question about {subset_name}. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{query}
```

---

### MMLU-Pro

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `mmlu_pro`
- **Dataset ID**: [modelscope/MMLU-Pro](https://modelscope.cn/datasets/modelscope/MMLU-Pro/summary)
- **Description**:  
  > MMLU-Pro is a benchmark for evaluating language models on multiple-choice questions across various subjects. It includes questions from different domains, where the model must select the correct answer from given options.
- **Task Categories**: `Knowledge`, `MCQ`
- **Evaluation Metrics**: `AverageAccuracy`
- **Requires LLM Judge**: No
- **Default Shots**: 5-shot
- **Subsets**: `biology`, `business`, `chemistry`, `computer science`, `economics`, `engineering`, `health`, `history`, `law`, `math`, `other`, `philosophy`, `physics`, `psychology`

- **Supported Output Formats**: `generation`, `multiple_choice_logits`
- **Prompt Template**: 
```text
The following are multiple choice questions (with answers) about {subset_name}. Think step by step and then finish your answer with "the answer is (X)" where X is the correct letter choice.
{query}
```

---

### MMLU-Redux

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `mmlu_redux`
- **Dataset ID**: [AI-ModelScope/mmlu-redux-2.0](https://modelscope.cn/datasets/AI-ModelScope/mmlu-redux-2.0/summary)
- **Description**:  
  > MMLU-Redux is a benchmark for evaluating language models on multiple-choice questions across various subjects. It includes questions from different domains, where the model must select the correct answer from given options.
- **Task Categories**: `Knowledge`, `MCQ`
- **Evaluation Metrics**: `AverageAccuracy`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Subsets**: `abstract_algebra`, `anatomy`, `astronomy`, `business_ethics`, `clinical_knowledge`, `college_biology`, `college_chemistry`, `college_computer_science`, `college_mathematics`, `college_medicine`, `college_physics`, `computer_security`, `conceptual_physics`, `econometrics`, `electrical_engineering`, `elementary_mathematics`, `formal_logic`, `global_facts`, `high_school_biology`, `high_school_chemistry`, `high_school_computer_science`, `high_school_european_history`, `high_school_geography`, `high_school_government_and_politics`, `high_school_macroeconomics`, `high_school_mathematics`, `high_school_microeconomics`, `high_school_physics`, `high_school_psychology`, `high_school_statistics`, `high_school_us_history`, `high_school_world_history`, `human_aging`, `human_sexuality`, `international_law`, `jurisprudence`, `logical_fallacies`, `machine_learning`, `management`, `marketing`, `medical_genetics`, `miscellaneous`, `moral_disputes`, `moral_scenarios`, `nutrition`, `philosophy`, `prehistory`, `professional_accounting`, `professional_law`, `professional_medicine`, `professional_psychology`, `public_relations`, `security_studies`, `sociology`, `us_foreign_policy`, `virology`, `world_religions`

- **Supported Output Formats**: `generation`, `multiple_choice_logits`
- **Prompt Template**: 
```text
The following are multiple choice questions (with answers) about {subset_name}. Think step by step and then finish your answer with "the answer is (X)" where X is the correct letter choice.
{query}
```

---

### MuSR

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `musr`
- **Dataset ID**: [AI-ModelScope/MuSR](https://modelscope.cn/datasets/AI-ModelScope/MuSR/summary)
- **Description**:  
  > MuSR is a benchmark for evaluating AI models on multiple-choice questions related to murder mysteries, object placements, and team allocation.
- **Task Categories**: `MCQ`, `Reasoning`
- **Evaluation Metrics**: `AverageAccuracy`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Subsets**: `murder_mysteries`, `object_placements`, `team_allocation`

- **Supported Output Formats**: `generation`, `multiple_choice_logits`
- **Prompt Template**: 
```text
{narrative}

{question}

{choices}
Think step by step and then finish your answer with "the answer is (X)" where X is the correct letter choice.
```

---

### Needle-in-a-Haystack

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `needle_haystack`
- **Dataset ID**: [AI-ModelScope/Needle-in-a-Haystack-Corpus](https://modelscope.cn/datasets/AI-ModelScope/Needle-in-a-Haystack-Corpus/summary)
- **Description**:  
  > Needle in a Haystack is a benchmark focused on information retrieval tasks. It requires the model to find specific information within a large corpus of text. [Usage Example](https://evalscope.readthedocs.io/zh-cn/latest/third_party/needle_haystack.html)
- **Task Categories**: `Long Context`, `Retrieval`
- **Evaluation Metrics**: `AverageAccuracy`
- **Requires LLM Judge**: Yes
- **Default Shots**: 0-shot
- **Subsets**: `chinese`, `english`

- **Supported Output Formats**: `generation`
- **Extra Parameters**: 
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
- **System Prompt**: 
```text
You are a helpful AI bot that answers questions for a user. Keep your response short and direct
```
- **Prompt Template**: 
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

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `process_bench`
- **Dataset ID**: [Qwen/ProcessBench](https://modelscope.cn/datasets/Qwen/ProcessBench/summary)
- **Description**:  
  > ProcessBench is a benchmark for evaluating AI models on mathematical reasoning tasks. It includes various subsets such as GSM8K, Math, OlympiadBench, and OmniMath, each with its own set of problems that require step-by-step reasoning to arrive at the correct answer.
- **Task Categories**: `Mathematical`, `Reasoning`
- **Evaluation Metrics**: `correct_acc`, `error_acc`, `simple_f1_score`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Subsets**: `gsm8k`, `math`, `olympiadbench`, `omnimath`

- **Supported Output Formats**: `generation`
- **Prompt Template**: 
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

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `race`
- **Dataset ID**: [modelscope/race](https://modelscope.cn/datasets/modelscope/race/summary)
- **Description**:  
  > RACE is a benchmark for testing reading comprehension and reasoning abilities of neural models. It is constructed from Chinese middle and high school examinations.
- **Task Categories**: `MCQ`, `Reasoning`
- **Evaluation Metrics**: `AverageAccuracy`
- **Requires LLM Judge**: No
- **Default Shots**: 3-shot
- **Subsets**: `high`, `middle`

- **Supported Output Formats**: `generation`, `multiple_choice_logits`

---

### SimpleQA

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `simple_qa`
- **Dataset ID**: [AI-ModelScope/SimpleQA](https://modelscope.cn/datasets/AI-ModelScope/SimpleQA/summary)
- **Description**:  
  > SimpleQA is a benchmark designed to evaluate the performance of language models on simple question-answering tasks. It includes a set of straightforward questions that require basic reasoning and understanding capabilities.
- **Task Categories**: `Knowledge`, `QA`
- **Evaluation Metrics**: `is_correct`, `is_incorrect`, `is_not_attempted`
- **Requires LLM Judge**: Yes
- **Default Shots**: 0-shot
- **Subsets**: `default`

- **Supported Output Formats**: `generation`

---

### SuperGPQA

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `super_gpqa`
- **Dataset ID**: [m-a-p/SuperGPQA](https://modelscope.cn/datasets/m-a-p/SuperGPQA/summary)
- **Description**:  
  > SuperGPQA is a large-scale multiple-choice question answering dataset, designed to evaluate the generalization ability of models across different fields. It contains 100,000+ questions from 50+ fields, with each question having 10 options.
- **Task Categories**: `Knowledge`, `MCQ`
- **Evaluation Metrics**: `AverageAccuracy`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Subsets**: `Aeronautical and Astronautical Science and Technology`, `Agricultural Engineering`, `Animal Husbandry`, `Applied Economics`, `Aquaculture`, `Architecture`, `Art Studies`, `Astronomy`, `Atmospheric Science`, `Basic Medicine`, `Biology`, `Business Administration`, `Chemical Engineering and Technology`, `Chemistry`, `Civil Engineering`, `Clinical Medicine`, `Computer Science and Technology`, `Control Science and Engineering`, `Crop Science`, `Education`, `Electrical Engineering`, `Electronic Science and Technology`, `Environmental Science and Engineering`, `Food Science and Engineering`, `Forestry Engineering`, `Forestry`, `Geography`, `Geological Resources and Geological Engineering`, `Geology`, `Geophysics`, `History`, `Hydraulic Engineering`, `Information and Communication Engineering`, `Instrument Science and Technology`, `Journalism and Communication`, `Language and Literature`, `Law`, `Library, Information and Archival Management`, `Management Science and Engineering`, `Materials Science and Engineering`, `Mathematics`, `Mechanical Engineering`, `Mechanics`, `Metallurgical Engineering`, `Military Science`, `Mining Engineering`, `Musicology`, `Naval Architecture and Ocean Engineering`, `Nuclear Science and Technology`, `Oceanography`, `Optical Engineering`, `Petroleum and Natural Gas Engineering`, `Pharmacy`, `Philosophy`, `Physical Education`, `Physical Oceanography`, `Physics`, `Political Science`, `Power Engineering and Engineering Thermophysics`, `Psychology`, `Public Administration`, `Public Health and Preventive Medicine`, `Sociology`, `Stomatology`, `Surveying and Mapping Science and Technology`, `Systems Science`, `Textile Science and Engineering`, `Theoretical Economics`, `Traditional Chinese Medicine`, `Transportation Engineering`, `Veterinary Medicine`, `Weapon Science and Technology`

- **Supported Output Formats**: `generation`, `multiple_choice_logits`

---

### ToolBench-Static

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `tool_bench`
- **Dataset ID**: [AI-ModelScope/ToolBench-Static](https://modelscope.cn/datasets/AI-ModelScope/ToolBench-Static/summary)
- **Description**:  
  > ToolBench is a benchmark for evaluating AI models on tool use tasks. It includes various subsets such as in-domain and out-of-domain, each with its own set of problems that require step-by-step reasoning to arrive at the correct answer. [Usage Example](https://evalscope.readthedocs.io/zh-cn/latest/third_party/toolbench.html)
- **Task Categories**: `Agent`, `Reasoning`
- **Evaluation Metrics**: `Act.EM`, `F1`, `HalluRate`, `Plan.EM`, `Rouge-L`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Subsets**: `in_domain`, `out_of_domain`

- **Supported Output Formats**: `generation`

---

### TriviaQA

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `trivia_qa`
- **Dataset ID**: [modelscope/trivia_qa](https://modelscope.cn/datasets/modelscope/trivia_qa/summary)
- **Description**:  
  > TriviaQA is a large-scale reading comprehension dataset consisting of question-answer pairs collected from trivia websites. It includes questions with multiple possible answers, making it suitable for evaluating the ability of models to understand and generate answers based on context.
- **Task Categories**: `QA`, `Reading Comprehension`
- **Evaluation Metrics**: `AverageAccuracy`
- **Requires LLM Judge**: No
- **Default Shots**: 5-shot
- **Subsets**: `default`

- **Supported Output Formats**: `generation`

---

### TruthfulQA

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `truthful_qa`
- **Dataset ID**: [modelscope/truthful_qa](https://modelscope.cn/datasets/modelscope/truthful_qa/summary)
- **Description**:  
  > TruthfulQA is a benchmark designed to evaluate the ability of AI models to answer questions truthfully and accurately. It includes multiple-choice and generation tasks, focusing on the model's understanding of factual information and its ability to generate coherent responses.
- **Task Categories**: `Knowledge`
- **Evaluation Metrics**: `AverageAccuracy`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Subsets**: `multiple_choice`

- **Supported Output Formats**: `continuous_logits`, `generation`

---

### Winogrande

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `winogrande`
- **Dataset ID**: [AI-ModelScope/winogrande_val](https://modelscope.cn/datasets/AI-ModelScope/winogrande_val/summary)
- **Description**:  
  > Winogrande is a benchmark for evaluating AI models on commonsense reasoning tasks, specifically designed to test the ability to resolve ambiguous pronouns in sentences.
- **Task Categories**: `MCQ`, `Reasoning`
- **Evaluation Metrics**: `AverageAccuracy`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Subsets**: `default`

- **Supported Output Formats**: `generation`, `multiple_choice_logits`
- **Prompt Template**: 
```text
Question: {query}
A. {option1}
B. {option2}
Answer:
```
