# align with official CL-bench eval.py(https://github.com/Tencent-Hunyuan/CL-bench/blob/main/eval.py)
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field, field_validator

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.judge import JudgeCase, JudgeContext, JudgeDefinition, JudgeRequest, OutputContract, ReducedVerdict
from evalscope.api.messages import dict_to_chat_message
from evalscope.api.messages.chat_message import ChatMessageUser
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import ScoringPolicy, Tags
from evalscope.utils.logger import get_logger

logger = get_logger()


class CLGrade(BaseModel):
    grading_rationale: str = Field(alias='Grading Rationale', default='')
    requirement_status: List[str] = Field(alias='List of Requirement Satisfaction Status', default_factory=list)
    overall_score: Literal[0, 1] = Field(alias='Overall Score')

    @field_validator('overall_score', mode='before')
    @classmethod
    def _coerce_score(cls, value: Any) -> Any:
        # The contract instruction renders the allowed labels quoted (`"0"` / `"1"`), so a
        # compliant judge replies with a string; accept it instead of wasting a parse retry.
        if isinstance(value, str) and value.strip() in ('0', '1'):
            return int(value.strip())
        return value


GRADE_CONTRACT = OutputContract(schema_model=CLGrade)

GRADING_TEMPLATE = (
    'Starting now, you are a rigorous instruction-following grading teacher. Your task is to accurately grade and score student answers based on the 【Rubrics】.\n\n'
    'Grading Criteria\n'
    'This is a strict, all-or-nothing grading system. The final score is binary.\n'
    "To receive a score of 1, the student's answer must perfectly satisfy every single requirement listed in the 【Rubrics】.\n"
    'If even one requirement is not fully met, the final score will be 0.\n'
    'Grading Process\n'
    'Please strictly follow the steps below for analysis—no steps may be skipped:\n'
    'Step 1: Analyze the Standard Answer\n'
    'List all explicit requirements in the 【Rubrics】 item by item (including format, content, quantity, order, etc.).\n'
    'Identify implicit requirements in the 【Rubrics】 (e.g., language style, logical structure).\n'
    'Define specific evaluation criteria for each requirement (e.g., "must include X," "must not exceed Y").\n'
    "Step 2: Check Each Requirement Against the Student's Answer\n"
    "For every requirement in the 【Rubrics】, verify one by one whether the student's answer fully satisfies it.\n"
    'Step 3: Self-Reflection\n'
    'Before giving the final score, you must conduct the following checks:\n'
    '  Completeness Check: Whether all requirements in the standard answer have been reviewed with no omissions.\n'
    '  Strictness Check: Whether the evaluation strictly adheres to the "fully satisfied" standard without relaxing requirements due to subjective judgment.\n'
    '  Consistency Check: Whether the grading rationale aligns logically with the final score.\n'
    '  Objectivity Check: Whether judgments are based on objective facts rather than subjective speculation.\n\n'
    'Content to Be Graded\n'
    '【Rubrics】:\n{rubrics}\n'
    '【Student Response】:\n{response}\n'
)

CL_BENCH_DESCRIPTION = """
## Overview

CL-bench represents a step towards building LMs with this fundamental capability (Context Learning), making them more intelligent and advancing their deployment in real-world scenarios. This benchmark is specifically designed to evaluate a model's ability to learn specific, often novel or long-tail knowledge directly from the context and apply it to solve problems, simulating real-world learning processes.

**Resources:**
[Homepage](https://github.com/Tencent-Hunyuan/CL-bench) | [Dataset](https://huggingface.co/datasets/tencent/CL-bench)

## Task Description

- **Task Type**: In-Context Learning & Reasoning (Context-dependent QA)
- **Input**: A context containing new rules, fictional information, or specific logic, followed by a query
- **Output**: A solution derived strictly from the provided context (not pre-trained knowledge)
- **Difficulty**: Varied, requiring understanding of novel concepts defined in the prompt

## Key Features

- **Contamination-Free**: Uses synthetic or highly specific data (e.g., fictional laws, new programming syntax) to ensure models cannot rely on memorized training data
- **Real-world Simulation**: Mimics scenarios where humans learn new tasks by reading instructions or documentation
- **Diverse Domains**: Covers logic reasoning, rule following, language understanding, and puzzle solving within a given context
- **Evaluation of Adaptability**: Measures the "learning" capability rather than just "retrieval" or "memory"

## Evaluation Notes

- Focuses on the model's ability to follow strict instructions provided in the context
- Evaluation typically checks if the reasoning process utilizes the unique information given in the prompt
- Answers are often evaluated against specific ground-truth rules defined in the context
- Crucial for assessing how well models can adapt to private data or dynamic environments without fine-tuning
"""


@register_benchmark(
    BenchmarkMeta(
        name='cl_bench',
        pretty_name='CL-bench',
        tags=[Tags.INSTRUCTION_FOLLOWING, Tags.REASONING],
        description=CL_BENCH_DESCRIPTION,
        dataset_id='tencent-community/CL-bench',
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='train',
        prompt_template='',
    )
)
class CLBenchAdapter(DefaultDataAdapter):
    scoring_policy = ScoringPolicy.JUDGE_ONLY

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load(self):
        last_error = None
        original_split = self.eval_split
        candidate_splits = [original_split]  # only eval on the original split, but or subsequent adaptation, if needed
        seen = set()
        for split in candidate_splits:
            if not split or split in seen:
                continue
            seen.add(split)
            try:
                self.eval_split = split
                return super().load()
            except Exception as exc:
                last_error = exc
                logger.warning(f'Failed to load CL-bench split "{split}": {exc}')
        if original_split:
            self.eval_split = original_split
        if last_error:
            raise last_error
        return super().load()

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        messages = [dict_to_chat_message(msg) for msg in record.get('messages', [])]
        rubrics = record.get('rubrics', [])
        metadata = record.get('metadata') or record.get('meta_data') or {}
        return Sample(input=messages, target=rubrics, metadata=metadata)

    def judge_definition(self, context: JudgeContext) -> JudgeDefinition:
        if not context.filtered_prediction.strip():
            return JudgeDefinition.skip(
                Score(
                    extracted_prediction=context.filtered_prediction,
                    prediction=context.original_prediction,
                    value={'acc': 0.0},
                    main_score_name='acc',
                    explanation='Empty model output; scored as 0.',
                ),
                reason='empty_model_output',
            )

        def request(case, placement, completed_cases, judge_context) -> JudgeRequest:
            from .utils import build_rubrics_text

            target = judge_context.task_state.target
            prompt = (
                GRADING_TEMPLATE.format(
                    rubrics=build_rubrics_text(target if isinstance(target, list) else [target]),
                    response=judge_context.filtered_prediction,
                )
                + case.output_contract.instruction()
            )
            return JudgeRequest(messages=[ChatMessageUser(content=prompt)])

        def reduce(case_verdicts, judge_context) -> ReducedVerdict:
            grade = case_verdicts[0].value
            return ReducedVerdict(
                value={'acc': float(grade.overall_score)},
                metadata={
                    'requirement_status': grade.requirement_status,
                    'grading_rationale': grade.grading_rationale,
                },
            )

        def finalize(score, review, judge_context) -> Score:
            score.explanation = review.metadata.get('grading_rationale', '')
            return score

        return JudgeDefinition.workflow(
            cases=[JudgeCase(case_id='grade', output_contract=GRADE_CONTRACT)],
            request=request,
            reduce=reduce,
            main_score_name='acc',
            finalize=finalize,
        )
