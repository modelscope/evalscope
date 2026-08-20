# flake8: noqa: E501
import json
from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Tuple

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.judge import JudgeCase, JudgeContext, JudgeRequest, OutputContract, ReducedVerdict
from evalscope.api.messages import ChatMessageSystem, ChatMessageUser
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import ScoringPolicy, Tags
from evalscope.utils.logger import get_logger
from .utils import (
    CONSULTATION_MAX_QUESTIONS,
    JUDGE_SYSTEM_PROMPTS,
    JUDGE_TYPE_CASE_ANALYSIS,
    JUDGE_TYPE_LEGAL_QA,
    JUDGE_USER_PROMPTS,
    TASK_INSTRUCTIONS,
    USER_PROMPTS,
    build_conversation,
    parse_rubric_sections,
    score_case_analysis,
    score_total_points,
)

logger = get_logger()


class SectionScore(BaseModel):
    total_points: float


class CaseAnalysisGrade(BaseModel):
    """The ``case_analysis`` judge reply; section denominators come from the dataset, not the judge."""
    score_details: Dict[str, SectionScore]


class TotalPointsGrade(BaseModel):
    """The single-total judge reply for ``legal_qa`` / ``document_generation``."""
    total_points: float
    max_points: Optional[float] = None


SUBSET_LIST = [
    'case_analysis',
    'legal_consultation',
    'plaintiff_statement',
    'defendant_statement',
]

_EXTRA_PARAMS: Dict[str, Any] = {}

_DESCRIPTION = """
## Overview

PLawBench is a rubric-based benchmark that evaluates large language models on real-world Chinese legal practice.
It mirrors the workflow of a practising lawyer across three hierarchical levels: eliciting facts during a public
legal consultation, analysing a case with structured legal reasoning, and drafting professional legal documents.
Every item ships with a rubric annotated by legal experts, and grading is performed by an LLM judge against that
rubric rather than against a single reference answer.

## Task Description

- **Task Type**: Open-ended Chinese legal generation graded with expert rubrics
- **Input**: A client statement, or a case description plus a legal question
- **Output**: A question list, a structured case analysis, or a full legal document
- **Domain**: Chinese legal practice (personal affairs, marriage and family, corporate governance, intellectual
  property, criminal and civil litigation, cross-border matters, labour, environmental safety, and more)

## Key Features

- 280 samples split into four subsets, one per PLawBench task:
  - `case_analysis` (250): case analysis scored on four dimensions — conclusion, case facts, reasoning, and cited
    statutes. Answers must follow the 【结论】/【案件事实】/【推理过程】/【法条依据】 structure.
  - `legal_consultation` (18): the model plays a lawyer and must produce 10-25 verifiable follow-up questions that
    surface the facts the client omitted or distorted.
  - `plaintiff_statement` (6): drafting a statement of complaint from the client's account.
  - `defendant_statement` (6): drafting a statement of defense from the client's account and the opposing complaint.
- Client statements are deliberately vague, emotional, or misleading, so models must detect traps instead of
  restating the client's claims.
- Task prompts and judge prompts are ported verbatim from the official release, and the `case_analysis` rubric
  retains its per-dimension point allocation.

## Evaluation Notes

- Requires an LLM judge: set `judge.strategy='llm'` (or `'auto'`, which enables the judge for this benchmark)
  and provide `judge.models`. `judge.strategy='rule'` is not supported.
- Metrics are point ratios in `[0, 1]`. `acc` is reported for every subset; `case_analysis` additionally reports
  `conclusion_acc`, `fact_acc`, `reasoning_acc`, and `law_acc`. These map one-to-one onto the official leaderboard
  columns: `legal_consultation` is Task1, `case_analysis` is Task2-Avg with its four dimensions, and the two
  drafting subsets are Task3-Plaintiff and Task3-Defendant.
- Compare per-subset scores, not the `OVERALL` row. `OVERALL` is a per-sample mean, so `case_analysis` dominates it
  (250 of 280 samples). The paper's `Overall` column is an equal-weighted mean of the three task scores, which
  matches its published table far more closely (mean absolute error 0.72 versus 2.87 for a sample-weighted mean,
  fitted across the 24 models in the official ranking).
- Rubric point totals come from the dataset, not from the judge output, and awarded points are clamped into
  `[0, max_points]`, so a judge that mis-reports the denominator cannot distort the score.
- The judge output template for `case_analysis` is repaired relative to the official script, which ships malformed
  JSON and pins the conclusion section to zero points; every section is graded on its rubric allocation here.
- The judge model's transport retry policy is configured through its `generation_config`. A reply that still
  fails the output contract is unavailable and excluded rather than silently scored as zero.
- Case-analysis judging returns a long per-item breakdown. Give the judge a generous `max_tokens`
  (for example 8192) in `judge.models[].generation_config`.
- The drafting subsets ask for a 2,500-3,000 character legal document, so the evaluated model also needs a generous
  `generation_config.max_tokens`. A truncated filing is graded as an incomplete document and scores near zero, which
  depresses Task3 for reasons unrelated to legal ability.

Resources: [GitHub](https://github.com/skylenage/PLawbench) |
[Dataset](https://modelscope.cn/datasets/evalscope/PLawBench)
"""


@register_benchmark(
    BenchmarkMeta(
        name='plawbench',
        pretty_name='PLawBench',
        dataset_id='evalscope/PLawBench',
        tags=[Tags.KNOWLEDGE, Tags.QA, Tags.REASONING, Tags.CHINESE],
        description=_DESCRIPTION,
        paper_url='https://github.com/skylenage/PLawbench',
        subset_list=SUBSET_LIST,
        default_subset='case_analysis',
        eval_split='test',
        metric_list=['acc', 'conclusion_acc', 'fact_acc', 'reasoning_acc', 'law_acc'],
        primary_metric='accuracy',
        aggregation='mean',
        prompt_template='{question}',
        extra_params=_EXTRA_PARAMS,
    )
)
class PLawBenchAdapter(DefaultDataAdapter):
    """Rubric-based Chinese legal practice benchmark graded by an LLM judge."""

    scoring_policy = ScoringPolicy.JUDGE_ONLY
    judge_revision = '2'

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        judge_type = record['judge_type']
        user_prompt = USER_PROMPTS[judge_type].format(context=record['context'], question=record['question'])
        prompt = TASK_INSTRUCTIONS[judge_type] + user_prompt

        return Sample(
            input=prompt,
            target='',
            subset_key=record['task'],
            metadata={
                'id': record['id'],
                'task': record['task'],
                'judge_type': judge_type,
                'category': record['category'],
                'rubrics': record['rubrics'],
                'max_points': int(record['max_points']),
                'prompt': prompt,
            },
        )

    def build_judge_cases(self, context: JudgeContext) -> List[JudgeCase]:
        metadata = context.task_state.metadata or {}
        judge_type = metadata['judge_type']
        schema = CaseAnalysisGrade if judge_type == JUDGE_TYPE_CASE_ANALYSIS else TotalPointsGrade
        return [JudgeCase(
            case_id='rubric',
            output_contract=OutputContract(schema_model=schema),
        )]

    def build_judge_request(self, case, placement, completed_cases, context) -> JudgeRequest:
        metadata = context.task_state.metadata or {}
        judge_type = metadata['judge_type']
        rubric_sections = parse_rubric_sections(metadata['rubrics']) \
            if judge_type == JUDGE_TYPE_CASE_ANALYSIS else None
        system_prompt, user_prompt = self._build_judge_prompts(
            judge_type=judge_type,
            metadata=metadata,
            response=context.original_prediction,
            rubric_sections=rubric_sections,
        )
        return JudgeRequest(messages=[ChatMessageSystem(content=system_prompt), ChatMessageUser(content=user_prompt)])

    def reduce_judge_verdicts(self, case_verdicts, context) -> ReducedVerdict:
        metadata = context.task_state.metadata or {}
        judge_type = metadata['judge_type']
        judge_json = case_verdicts[0].value.model_dump()
        if judge_type == JUDGE_TYPE_CASE_ANALYSIS:
            values, details = score_case_analysis(judge_json, parse_rubric_sections(metadata['rubrics']))
        else:
            acc, details = score_total_points(judge_json, float(metadata['max_points']))
            values = {'acc': acc}
        return ReducedVerdict(value=values, metadata={'judge_type': judge_type, **details})

    def finalize_judge_score(self, review, context) -> Score:
        score = super().finalize_judge_score(review, context)
        score.main_score_name = 'acc'
        if review.metadata:
            # Surface the rubric breakdown in the review target column, which is otherwise empty.
            context.task_state.target = self._format_target(review.metadata['judge_type'], review.metadata)
        return score

    def _build_judge_prompts(
        self,
        judge_type: str,
        metadata: Dict[str, Any],
        response: str,
        rubric_sections: Optional[List[Dict[str, Any]]],
    ) -> Tuple[str, str]:
        """Render the judge system prompt and user prompt for one sample."""
        system_prompt = JUDGE_SYSTEM_PROMPTS[judge_type]
        if judge_type == JUDGE_TYPE_LEGAL_QA:
            system_prompt = system_prompt.replace('<<num>>', CONSULTATION_MAX_QUESTIONS)

        if rubric_sections is not None:
            rubric_text = json.dumps(rubric_sections, ensure_ascii=False, indent=2)
        else:
            rubric_text = metadata['rubrics']

        conversation = build_conversation(prompt=metadata['prompt'], response=response)
        user_prompt = JUDGE_USER_PROMPTS[judge_type] \
            .replace('<<conversation>>', conversation) \
            .replace('<<rubric_item>>', rubric_text) \
            .replace('<<score>>', str(metadata['max_points']))
        return system_prompt, user_prompt

    @staticmethod
    def _format_target(judge_type: str, details: Dict[str, Any]) -> str:
        lines = [f'**Rubric score**: {details["awarded_points"]:g} / {details["max_points"]:g}']
        for section, section_details in (details.get('sections') or {}).items():
            lines.append(f'- {section}: {section_details["awarded_points"]:g} / {section_details["max_points"]:g}')
        return '\n'.join(lines)
