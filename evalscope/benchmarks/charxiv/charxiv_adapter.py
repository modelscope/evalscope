# flake8: noqa: E501
from typing import Any, Dict, List, Union

from pydantic import BaseModel, Field

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.judge import JudgeCase, JudgeContext, JudgeDefinition, JudgeRequest, OutputContract, ReducedVerdict
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.metric.scorer import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import ScoringPolicy, Tags
from evalscope.utils.logger import get_logger

logger = get_logger()


class DescriptiveGrade(BaseModel):
    """Official descriptive-question reply keys."""

    extract_answer_T1: str = ''
    score_T1: int = Field(ge=0, le=1)

    @property
    def score(self) -> int:
        return self.score_T1


class ReasoningGrade(BaseModel):
    """Official reasoning-question reply keys."""

    extract_answer: str = ''
    score: int = Field(ge=0, le=1)


DESCRIPTIVE_CONTRACT = OutputContract(schema_model=DescriptiveGrade)
REASONING_CONTRACT = OutputContract(schema_model=ReasoningGrade)

DESCRIPTION = """
## Overview

CharXiv is a comprehensive chart understanding benchmark from NeurIPS 2024 that evaluates multimodal
large language models on realistic scientific charts from arXiv papers. It tests both low-level
chart element perception (descriptive) and high-level reasoning about chart data.

## Task Description

- **Task Type**: Chart Understanding (Descriptive + Reasoning)
- **Input**: Scientific chart image + question
- **Output**: Free-form text answer
- **Domains**: cs, physics, math, eess, q-bio, q-fin, stat, econ

## Key Features

- 2,323 real scientific charts from arXiv papers across 8 disciplines
- Two question types:
  - **Descriptive** (4 per chart): Basic element identification (titles, axes, legends, trends, etc.)
  - **Reasoning** (1 per chart): Higher-order reasoning requiring data synthesis
- 19 descriptive question templates covering information extraction, enumeration, pattern recognition, counting, and compositionality
- 4 reasoning answer types: text-in-chart, text-in-general, number-in-chart, number-in-general
- Validation set (1,000 charts) and test set (1,323 charts)
- Evaluation via LLM judge following the official CharXiv grading protocol

## Evaluation Notes

- Default evaluation uses the **validation** split (1,000 charts, 5,000 questions)
- Each chart yields 5 samples: 4 descriptive + 1 reasoning
- Primary metric: **Accuracy** via LLM-as-judge
- Subsets: `descriptive` and `reasoning` (also by category)
- Requires an LLM judge configured through `judge.models`
- [Paper](https://arxiv.org/abs/2406.18521) | [GitHub](https://github.com/princeton-nlp/CharXiv)
"""

SUBSET_LIST = ['descriptive', 'reasoning']


@register_benchmark(
    BenchmarkMeta(
        name='charxiv',
        pretty_name='CharXiv',
        dataset_id='princeton-nlp/CharXiv',
        tags=[Tags.MULTI_MODAL, Tags.REASONING, Tags.QA],
        description=DESCRIPTION,
        paper_url='https://arxiv.org/abs/2406.18521',
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        eval_split='validation',
    )
)
class CharXivAdapter(VisionLanguageAdapter):
    """Data adapter for princeton-nlp/CharXiv.

    Expands each chart record into 5 samples:
    - 4 descriptive questions (from template IDs)
    - 1 reasoning question (free-form)

    Scoring uses LLM judge for both question types.
    """
    scoring_policy = ScoringPolicy.JUDGE_ONLY

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reformat_subset = True
        self.save_metadata = True

    def record_to_sample(self, record: Dict[str, Any]) -> Union[Sample, List[Sample]]:
        """Convert a CharXiv record into multiple Samples (4 descriptive + 1 reasoning)."""
        from .utils import get_descriptive_question_text, get_reasoning_question_text

        # Convert PIL image to base64
        image = record.get('image')
        if image is None:
            logger.warning('Record missing image field, skipping.')
            return []

        image_b64 = self._image_bytes_to_base64(image['bytes'], default_format='jpeg')

        subplot_loc = record.get('subplot_loc', None)
        category = record.get('category', '')

        samples: List[Sample] = []

        # 4 descriptive questions
        for i in range(1, 5):
            q_id = record.get(f'descriptive_q{i}')
            answer = record.get(f'descriptive_a{i}', '')

            if q_id is None:
                continue

            question_text = get_descriptive_question_text(q_id, subplot_loc)
            if not question_text:
                continue

            content_list: List[Content] = [
                ContentImage(image=image_b64),
                ContentText(text=question_text),
            ]

            samples.append(
                Sample(
                    input=[ChatMessageUser(content=content_list)],
                    target=str(answer),
                    subset_key='descriptive',
                    metadata={
                        'question_type': 'descriptive',
                        'question_id': q_id,
                        'category': category,
                        'original_id': record.get('original_id', ''),
                    },
                )
            )

        # 1 reasoning question
        reasoning_q = record.get('reasoning_q', '')
        reasoning_a = record.get('reasoning_a', '')
        reasoning_a_type = record.get('reasoning_a_type', 1)

        if reasoning_q:
            question_text = get_reasoning_question_text(reasoning_q, reasoning_a_type, reasoning_a)
            content_list: List[Content] = [
                ContentImage(image=image_b64),
                ContentText(text=question_text),
            ]

            samples.append(
                Sample(
                    input=[ChatMessageUser(content=content_list)],
                    target=str(reasoning_a),
                    subset_key='reasoning',
                    metadata={
                        'question_type': 'reasoning',
                        'reasoning_a_type': reasoning_a_type,
                        'category': category,
                        'original_id': record.get('original_id', ''),
                    },
                )
            )

        return samples

    def judge_definition(self, context: JudgeContext) -> JudgeDefinition:
        descriptive = (context.task_state.metadata or {}).get('question_type', 'reasoning') == 'descriptive'
        contract = DESCRIPTIVE_CONTRACT if descriptive else REASONING_CONTRACT

        def request(case, placement, completed_cases, judge_context) -> JudgeRequest:
            from .utils import build_descriptive_judge_prompt, build_reasoning_judge_prompt
            metadata = judge_context.task_state.metadata or {}
            if metadata.get('question_type', 'reasoning') == 'descriptive':
                prompt = build_descriptive_judge_prompt(
                    q_id=metadata.get('question_id', 1),
                    response=judge_context.original_prediction,
                    ground_truth=judge_context.reference,
                )
            else:
                prompt = build_reasoning_judge_prompt(
                    reasoning_a_type=metadata.get('reasoning_a_type', 1),
                    question=judge_context.task_state.input_text or '',
                    ground_truth=judge_context.reference,
                    response=judge_context.original_prediction,
                )
            return JudgeRequest(messages=[ChatMessageUser(content=prompt)])

        def reduce(case_verdicts, judge_context) -> ReducedVerdict:
            return ReducedVerdict(value={'acc': float(case_verdicts[0].value.score)})

        return JudgeDefinition.workflow(
            cases=[JudgeCase(case_id='grade', output_contract=contract)],
            request=request,
            reduce=reduce,
            main_score_name='acc'
        )
