import re
from pydantic import BaseModel
from typing import Any, Dict, List, Literal

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.judge import (
    CaseVerdict,
    JudgeCase,
    JudgeContext,
    JudgeDefinition,
    JudgeRequest,
    OutputContract,
    ReducedVerdict,
)
from evalscope.api.messages import ChatMessage, ChatMessageSystem, ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import ScoringPolicy, Tags
from evalscope.utils.logger import get_logger

# flake8: noqa

logger = get_logger()


# The judge prompt requires "GRADE: C" or "GRADE: I" as the final line.
class Grade(BaseModel):
    reasoning: str = ''
    verdict: Literal['C', 'I']


GRADE_CONTRACT = OutputContract(schema_model=Grade)

SUBSET_LIST = [
    'Biology/Medicine',
    'Chemistry',
    'Computer Science/AI',
    'Engineering',
    'Humanities/Social Science',
    'Math',
    'Physics',
    'Other',
]

ANSWER_TYPE_EXACT_MATCH = 'exactMatch'
ANSWER_TYPE_MULTIPLE_CHOICE = 'multipleChoice'

# System prompt constants
SYSTEM_EXACT_ANSWER = 'Your response should be in the following format:\nExplanation: {your explanation for your final answer}\nExact Answer: {your succinct, final answer}\nConfidence: {your confidence score between 0% and 100% for your answer}'

SYSTEM_MC = 'Your response should be in the following format:\nExplanation: {your explanation for your answer choice}\nAnswer: {your chosen answer}\nConfidence: {your confidence score between 0% and 100% for your answer}'

JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Your judgment must focus only on if there are meaningful differences between [correct_answer] and the [response]. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match. Explain why the [response] is correct or incorrect based on [correct_answer] in one or two sentences. Finally, write your answer in the format 'GRADE: C' for correct answer or 'GRADE: I' for incorrect answer.
"""


@register_benchmark(
    BenchmarkMeta(
        name='hle',
        pretty_name="Humanity's-Last-Exam",
        tags=[Tags.KNOWLEDGE, Tags.QA],
        description="""
## Overview

Humanity's Last Exam (HLE) is a comprehensive language model benchmark consisting of 2,500 questions across a broad range of subjects. Created jointly by the Center for AI Safety and Scale AI, it represents one of the most challenging academic benchmarks available.

## Task Description

- **Task Type**: Expert-Level Question Answering
- **Input**: Question with optional image (14% multimodal)
- **Output**: Answer with explanation and confidence score
- **Domains**: Mathematics (41%), Physics (9%), Biology/Medicine (11%), Computer Science/AI (10%), Humanities (9%), Engineering (4%), Chemistry (7%), Other (9%)

## Key Features

- 2,500 expert-level questions across multiple disciplines
- 14% of questions require multimodal understanding
- 24% multiple-choice, 76% short-answer exact-match
- Questions from various academic and professional domains
- Includes confidence scoring in response format

## Evaluation Notes

- Default evaluation uses the **test** split
- Primary metric: **Accuracy** with LLM judge
- Response format includes: Explanation, Answer, and Confidence (0-100%)
- **Note**: Set `extra_params["include_multi_modal"]` to `False` for text-only models
- Uses GRADE: C/I format for LLM judge scoring
""",  # noqa: E501
        dataset_id='cais/hle',
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        eval_split='test',
        prompt_template='{question}',
        extra_params={
            'include_multi_modal': {
                'type': 'bool',
                'description': 'Include multi-modal (image) questions during evaluation.',
                'value': True,
            }
        },
    )
)
class HLEAdapter(DefaultDataAdapter):
    scoring_policy = ScoringPolicy.JUDGE_ONLY

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.reformat_subset = True
        self.include_multi_modal = self.extra_params.get('include_multi_modal', True)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        answer_type = record['answer_type']
        system_prompt = SYSTEM_EXACT_ANSWER if answer_type == ANSWER_TYPE_EXACT_MATCH else SYSTEM_MC
        text_content = ContentText(text=record['question'])

        content: List[Content] = [text_content]
        if record['image']:
            image_content = ContentImage(image=record['image'])
            content.append(image_content)

        messages: List[ChatMessage] = [
            ChatMessageSystem(content=system_prompt),
            ChatMessageUser(content=content),
        ]
        return Sample(
            input=messages,
            subset_key=record['category'],
            metadata={
                'uid': record['id'],
                'author_name': record['author_name'],
                'rationale': record['rationale'],
                'raw_subject': record['raw_subject'],
                'category': record['category'],
                'has_image': bool(record['image']),
            },
            target=record['answer'],
        )

    def sample_filter(self, sample):
        if not self.include_multi_modal:
            if sample.metadata is not None and sample.metadata['has_image']:
                return False
        return True

    def judge_definition(self, context: JudgeContext) -> JudgeDefinition:

        def request(case, placement, completed_cases, judge_context) -> JudgeRequest:
            prompt = (
                JUDGE_PROMPT.format(
                    question=judge_context.task_state.input_text,
                    response=judge_context.filtered_prediction,
                    correct_answer=judge_context.reference,
                )
                + case.output_contract.instruction()
            )
            return JudgeRequest(messages=[ChatMessageUser(content=prompt)])

        def reduce(case_verdicts, judge_context) -> ReducedVerdict:
            return ReducedVerdict(value={'acc': 1.0 if case_verdicts[0].value.verdict == 'C' else 0.0})

        def finalize(score, review, judge_context) -> Score:
            score.metadata['confidence'] = self._stated_confidence(judge_context.task_state)
            return score

        return JudgeDefinition.workflow(
            cases=[JudgeCase(case_id='grade', output_contract=GRADE_CONTRACT)],
            request=request,
            reduce=reduce,
            main_score_name='acc',
            finalize=finalize,
        )

    @staticmethod
    def _stated_confidence(task_state: TaskState) -> int:
        """Read the confidence the evaluated model reported; unrelated to the judge verdict."""
        completion = task_state.output.completion if task_state.output else ''
        match = re.search(r'confidence:\s*(\d+)', completion or '', re.IGNORECASE)
        return int(match.group(1)) if match else 100
