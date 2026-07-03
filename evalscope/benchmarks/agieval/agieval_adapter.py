# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger
from .utils import (
    ALL_SUBSETS,
    build_prompt,
    extract_math_answer,
    extract_multiple_answers,
    extract_single_answer_en,
    extract_single_answer_zh,
    is_chinese_qa,
    is_cloze,
    is_english_qa,
    is_multi_choice,
    is_qa,
    score_math,
    score_multiple_choice,
    score_single_choice,
)

logger = get_logger()

DESCRIPTION = """
## Overview

AGIEval is a human-centric benchmark designed to evaluate foundation models in the context of human cognition and problem-solving. It uses official, standard, and authoritative admission and qualification exams intended for general human test-takers, such as college entrance exams (GaoKao), law school admission tests (LSAT), math competitions, and lawyer qualification exams.

## Task Description

- **Task Type**: Mixed (Multiple-Choice QA + Open-ended Math)
- **Input**: Questions from standardized exams with optional passages and answer choices
- **Output**: Answer letter(s) for MCQ, or numerical/mathematical answer for open-ended
- **Languages**: English and Chinese

## Key Features

- 21 subsets covering diverse exam types across two languages
- English MCQ: LSAT (AR/LR/RC), SAT (Math/English), AQuA-RAT, LogiQA, GaoKao-English
- Chinese MCQ: GaoKao (Chinese/Geography/History/Biology/Chemistry/Physics/MathQA), LogiQA-zh, JEC-QA
- Open-ended math: MATH (English), GaoKao-MathCloze (Chinese)
- Multi-select subsets: JEC-QA-KD, JEC-QA-CA, GaoKao-Physics
- Includes passage-based reading comprehension questions

## Evaluation Notes

- MCQ subsets use exact letter match (single or multi-select)
- Math/cloze subsets use mathematical equivalence checking
- Supports few-shot evaluation using dev split examples
- Prompt format follows official AGIEval conventions
"""


@register_benchmark(
    BenchmarkMeta(
        name='agieval',
        pretty_name='AGIEval',
        dataset_id='opencompass/agieval',
        tags=[Tags.REASONING, Tags.KNOWLEDGE, Tags.MATH],
        description=DESCRIPTION,
        subset_list=ALL_SUBSETS,
        metric_list=['acc'],
        few_shot_num=0,
        train_split='dev',
        eval_split='test',
        prompt_template='{question}',
    )
)
class AGIEvalAdapter(DefaultDataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        subset = self.current_subset_name
        input_text = build_prompt(record, subset)
        label = record.get('label', '')
        target = label if isinstance(label, str) else ''.join(sorted(label))

        return Sample(
            input=input_text,
            target=target,
            choices=record.get('options') if is_qa(subset) else None,
            metadata={'subset': subset},
        )

    def sample_to_fewshot(self, sample: Sample) -> str:
        return f'{sample.input}\n{sample.target}'

    def format_fewshot_template(self, fewshot: str, sample: Sample) -> str:
        if fewshot:
            return f'{fewshot}\n\n{sample.input}'
        return sample.input

    def format_prompt_template(self, sample: Sample) -> str:
        return sample.input

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        """Extract answer based on subset type."""
        subset = task_state.metadata.get('subset', '') if task_state.metadata else ''

        if is_multi_choice(subset):
            return extract_multiple_answers(prediction)
        elif is_english_qa(subset):
            return extract_single_answer_en(prediction)
        elif is_chinese_qa(subset):
            return extract_single_answer_zh(prediction)
        elif is_cloze(subset):
            return extract_math_answer(prediction)
        else:
            return extract_single_answer_en(prediction)

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        """Score based on subset type."""
        score = Score(extracted_prediction=filtered_prediction, prediction=original_prediction)
        subset = task_state.metadata.get('subset', '') if task_state.metadata else ''

        if is_multi_choice(subset):
            correct = score_multiple_choice(filtered_prediction, reference)
        elif is_cloze(subset):
            correct = score_math(filtered_prediction, reference)
        else:
            correct = score_single_choice(filtered_prediction, reference)

        score.value = {'acc': correct}
        score.main_score_name = 'acc'
        return score
