import threading
from typing import Any, Dict, List

from evalscope.api.benchmark import AgentLoopAdapter, BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import AggScore, SampleScore, Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from .utils import aggregate_official_scores, build_grader_prompt, parse_judge_response, rule_fallback_score

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
- The primary metric is `f1_score`; `precision`, `recall`, and empty/invalid response rates are also reported.
- `JudgeStrategy.RULE` provides a conservative exact/substring fallback for smoke tests and is not equivalent to official LLM judging.
""".strip()  # noqa: E501


@register_benchmark(
    BenchmarkMeta(
        name='deepsearchqa',
        pretty_name='DeepSearchQA',
        tags=[Tags.AGENT, Tags.KNOWLEDGE, Tags.QA, Tags.RETRIEVAL],
        description=DESCRIPTION,
        dataset_id=DEEPSEARCHQA_DATASET_ID,
        metric_list=['f1_score', 'precision', 'recall'],
        few_shot_num=0,
        train_split=None,
        eval_split='eval',
        prompt_template='{question}',
        paper_url='https://storage.googleapis.com/deepmind-media/DeepSearchQA/DeepSearchQA_benchmark_paper.pdf',
    )
)
class DeepSearchQAAdapter(AgentLoopAdapter):
    """Adapter for the DeepSearchQA deep-research benchmark."""
    llm_judge_default = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._suppress_doc_sample_example = True
        self._judge_lock = threading.Lock()

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        problem = record['problem']
        return Sample(
            input=problem,
            target=record.get('answer') or '',
            metadata={
                'problem_category': record.get('problem_category') or '',
                'answer_type': record.get('answer_type') or 'Set Answer',
                'question': problem,
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
        answer_type = (task_state.metadata or {}).get('answer_type') or 'Set Answer'
        value, metadata = rule_fallback_score(filtered_prediction, reference, answer_type)
        return Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
            value=value,
            metadata=metadata,
            main_score_name='f1_score',
        )

    def llm_match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        metadata = task_state.metadata or {}
        answer_type = metadata.get('answer_type') or 'Set Answer'
        question = metadata.get('question') or task_state.input_text
        prompt = build_grader_prompt(
            question=question,
            reference=reference,
            answer_type=answer_type,
            response=filtered_prediction,
        )

        if not filtered_prediction:
            judge_response = ''
            value, judge_metadata = {}, {
                'empty_model_response': True,
                'error_message': 'AI response was empty.',
            }
        else:
            with self._judge_lock:
                judge_response = self.llm_judge.judge(prompt)
            value, judge_metadata = parse_judge_response(judge_response)

        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
            value=value,
            explanation=f'LLM judge: {judge_response}',
            main_score_name='f1_score',
        )
        score.metadata = {
            'source': 'llm_judge',
            'judge_strategy': self.judge_strategy,
            'model': self.llm_judge.model_id,
            'rating_prompt': prompt,
            **judge_metadata,
        }
        return score

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        return aggregate_official_scores(sample_scores)
