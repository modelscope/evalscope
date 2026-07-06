# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import Choices, TaskState
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import (
    MultipleChoiceTemplate,
    parse_answers,
    parse_answers_zh,
    prompt,
    valid_template,
)
from .utils import ALL_SUBSETS, is_chinese_qa, is_cloze, is_multi_choice, is_qa

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

- MCQ subsets use evalscope's standard MultiChoice template and extraction
- Multi-select subsets use Chinese multi-answer template
- Math/cloze subsets use mathematical equivalence checking
- CoT (Chain-of-Thought) prompting enabled by default
"""

MATH_PROMPT_TEMPLATE = '{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.'.lstrip()


@register_benchmark(
    BenchmarkMeta(
        name='agieval',
        pretty_name='AGIEval',
        dataset_id='opencompass/agieval',
        tags=[Tags.REASONING, Tags.KNOWLEDGE, Tags.MATH, Tags.MULTIPLE_CHOICE],
        description=DESCRIPTION,
        subset_list=ALL_SUBSETS,
        metric_list=['acc'],
        few_shot_num=0,
        train_split='dev',
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class AGIEvalAdapter(MultiChoiceAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        subset = self.current_subset_name
        raw_label = record.get('label', '')
        passage = record.get('passage') or ''
        question = record['question']

        # Parse label: may be a string like "['A', 'B']" for multi-select
        label = self._parse_label(raw_label)

        # Prepend passage to question if present
        full_question = f'{passage}\n\n{question}' if passage else question

        if is_qa(subset):
            options = record.get('options') or []
            target = ''.join(sorted(label)) if isinstance(label, list) else label
            return Sample(
                input=full_question,
                choices=options,
                target=target,
                subset_key=subset,
                metadata={'subset': subset},
            )
        else:
            # Cloze/Math: no choices
            target = label if isinstance(label, str) else str(label)
            return Sample(
                input=full_question,
                target=target,
                subset_key=subset,
                metadata={'subset': subset},
            )

    @staticmethod
    def _parse_label(label) -> Any:
        """Parse label that may be a JSON-encoded list string like \"['A', 'B']\"."""
        if isinstance(label, list):
            return label
        if isinstance(label, str) and label.startswith('['):
            import ast
            try:
                parsed = ast.literal_eval(label)
                if isinstance(parsed, list):
                    return parsed
            except (ValueError, SyntaxError):
                pass
        return label

    def format_prompt_template(self, sample: Sample) -> str:
        """Dispatch prompt formatting based on subset type."""
        subset = sample.metadata.get('subset', '') if sample.metadata else ''

        if is_cloze(subset):
            return MATH_PROMPT_TEMPLATE.format(question=sample.input)

        # MCQ: select template based on language and multi-select
        if is_multi_choice(subset):
            template = MultipleChoiceTemplate.CHINESE_MULTIPLE_ANSWER_TEMPLATE_COT
        elif is_chinese_qa(subset):
            template = MultipleChoiceTemplate.CHINESE_SINGLE_ANSWER_TEMPLATE_COT
        else:
            template = MultipleChoiceTemplate.SINGLE_ANSWER_COT

        return prompt(
            question=sample.input,
            choices=Choices(sample.choices),
            template=template,
        )

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        """Dispatch extraction based on subset type."""
        subset = task_state.metadata.get('subset', '') if task_state.metadata else ''

        if is_cloze(subset):
            from evalscope.metrics.math.parser import extract_answer
            return extract_answer(prediction)

        # MCQ: use evalscope's built-in parsers
        multiple = is_multi_choice(subset)
        if is_chinese_qa(subset):
            answers = parse_answers_zh(task_state, multiple_correct=multiple)
        else:
            answers = parse_answers(task_state, multiple_correct=multiple)
        return ''.join(sorted(list(answers)))

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        """Score based on subset type."""
        score = Score(extracted_prediction=filtered_prediction, prediction=original_prediction)
        subset = task_state.metadata.get('subset', '') if task_state.metadata else ''

        if is_cloze(subset):
            from evalscope.metrics.math.parser import math_equal
            correct = 1.0 if math_equal(filtered_prediction, reference) else 0.0
        else:
            # MCQ: exact match on extracted letters
            correct = 1.0 if filtered_prediction.upper() == reference.upper() else 0.0

        score.value = {'acc': correct}
        score.main_score_name = 'acc'
        return score
