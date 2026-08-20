from typing import Any, Dict, List

from evalscope.api.benchmark import AgentLoopAdapter, BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.judge import JudgeCase, JudgeContext, JudgeRequest, ReducedVerdict
from evalscope.api.messages import ChatMessageUser
from evalscope.api.metric import AggScore, SampleScore, Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import ScoringPolicy, Tags
from . import utils as deepsearchqa_utils
from .utils import (
    GRADE_CONTRACT,
    aggregate_official_scores,
    build_grader_prompt,
    metrics_from_grade,
    rule_fallback_score,
)

DEEPSEARCHQA_DATASET_ID = 'google/deepsearchqa'

DESCRIPTION = """
## Overview

DeepSearchQA is a Google DeepMind benchmark for evaluating deep research agents on difficult multi-step information-seeking tasks across the open web. It contains 900 prompts spanning 17 domains and is designed to measure exhaustive answer-set generation rather than single-answer retrieval alone.

## Task Description

- **Task Type**: Search-agent factual question answering
- **Input**: A natural-language research question
- **Output**: A single answer or complete answer set, depending on the question
- **Grading**: LLM-as-judge semantic matching against the gold answer and answer type

## Key Features

- Tests systematic collation of fragmented information from multiple sources
- Requires entity resolution and de-duplication for set-answer tasks
- Penalizes both under-retrieval and excessive/hallucinated answers
- Uses `problem_category` for analysis metadata; `answer_type` is withheld from the model during inference
- Compatible with EvalScope agent configurations for native or external web-capable agents

## Agent Tool Configuration

DeepSearchQA does not hard-code a search provider. By default it runs through EvalScope native AgentLoop without external search tools. To evaluate a web-capable agent, set `TaskConfig.agent_config` and attach the search/fetch tools that should be available to the model. If `NativeAgentConfig.max_steps` is omitted, DeepSearchQA uses its benchmark-level AgentLoop default of 30 steps.

See the [DeepSearchQA usage guide](https://evalscope.readthedocs.io/en/latest/third_party/deepsearchqa.html) for
runtime examples, MCP search/fetch configuration, and evaluation notes.

## Evaluation Notes

- EvalScope loads the ModelScope dataset `google/deepsearchqa` from the `eval` split.
- LLM judge is enabled by default. Official starter code uses Gemini 2.5 Flash with the DeepSearchQA judge prompt, but EvalScope can use any configured judge model for local runs.
- The primary metric is `f1`; `precision`, `recall`, and empty/invalid response rates are also reported.
- `JudgeStrategy.RULE` provides a conservative exact/substring fallback for smoke tests and is not equivalent to official LLM judging.
""".strip()  # noqa: E501


@register_benchmark(
    BenchmarkMeta(
        name='deepsearchqa',
        pretty_name='DeepSearchQA',
        tags=[Tags.AGENT, Tags.KNOWLEDGE, Tags.QA, Tags.RETRIEVAL],
        description=DESCRIPTION,
        dataset_id=DEEPSEARCHQA_DATASET_ID,
        metric_list=['f1', 'precision', 'recall'],
        primary_metric='f1',
        few_shot_num=0,
        train_split=None,
        eval_split='eval',
        prompt_template='{question}',
        paper_url='https://storage.googleapis.com/deepmind-media/DeepSearchQA/DeepSearchQA_benchmark_paper.pdf',
    )
)
class DeepSearchQAAdapter(AgentLoopAdapter):
    """Adapter for the DeepSearchQA deep-research benchmark."""
    scoring_policy = ScoringPolicy.JUDGE_DEFAULT
    judge_revision = '1'
    judge_cache_dependencies = (deepsearchqa_utils.__file__, )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._suppress_doc_sample_example = True

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        problem = record['problem']
        return Sample(
            input=problem,
            target=record['answer'] or '',
            metadata={
                'problem_category': record['problem_category'],
                'answer_type': record['answer_type'],
            },
        )

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        return prediction.strip()

    def build_max_steps_finalization_message(self, sample: Sample) -> str:
        return (
            'You have reached the maximum number of tool-use steps. Based only on the information already gathered '
            'in this conversation, provide your best final answer now. Return only the answer, with no explanation.'
        )

    def match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        answer_type = task_state.metadata['answer_type']
        value, metadata = rule_fallback_score(filtered_prediction, reference, answer_type)
        return Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
            value=value,
            metadata=metadata,
            main_score_name='f1',
        )

    def pre_judge_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        if not filtered_prediction:
            # No answer to grade: score nothing without a judge call, as in the official evaluator.
            return Score(
                extracted_prediction=filtered_prediction,
                prediction=original_prediction,
                value={},
                main_score_name='f1',
                metadata={
                    'empty_model_response': True,
                    'error_message': 'AI response was empty.'
                },
            )
        return None

    def build_judge_cases(self, context: JudgeContext) -> List[JudgeCase]:
        return [JudgeCase(case_id='grade', output_contract=GRADE_CONTRACT)]

    def build_judge_request(self, case, placement, completed_cases, context) -> JudgeRequest:
        prompt = build_grader_prompt(
            question=context.task_state.input_text,
            reference=context.reference,
            answer_type=context.task_state.metadata['answer_type'],
            response=context.filtered_prediction,
        )
        return JudgeRequest(messages=[ChatMessageUser(content=prompt)])

    def reduce_judge_verdicts(self, case_verdicts, context) -> ReducedVerdict:
        value, metadata = metrics_from_grade(case_verdicts[0].value)
        return ReducedVerdict(value=value, metadata=metadata)

    def finalize_judge_score(self, review, context) -> Score:
        score = super().finalize_judge_score(review, context)
        score.main_score_name = 'f1'
        return score

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        return aggregate_official_scores(sample_scores)
