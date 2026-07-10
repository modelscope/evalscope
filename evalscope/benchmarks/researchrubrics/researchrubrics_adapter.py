from __future__ import annotations

import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

from evalscope.agent.tools.bash import BASH_TOOL_INFO, run_bash
from evalscope.api.agent import AgentEnvironment, AgentStrategy, EventType
from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.benchmark.adapters import AgentLoopAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import InferenceResult, TaskState
from evalscope.api.messages import ChatMessageUser
from evalscope.api.metric import AggScore, SampleScore, Score
from evalscope.api.registry import get_strategy, register_benchmark
from evalscope.constants import JudgeStrategy, Tags
from evalscope.utils.logger import get_logger
from .utils import (
    BINARY_SYSTEM_PROMPT,
    BINARY_USER_PROMPT,
    CHUNK_SYSTEM_PROMPT,
    CHUNK_USER_PROMPT,
    SYNTHESIS_USER_PROMPT,
    TemporaryLocalAgentEnvironment,
    chunk_document,
    parse_json_object,
    validate_binary_result,
    validate_chunk_result,
)

logger = get_logger()

_EXTRA_PARAMS: Dict[str, Any] = {
    'strategy': {
        'type': 'str',
        'description': 'Agent strategy used by the built-in AgentLoop.',
        'value': 'function_calling',
        'choices': ['function_calling', 'react'],
    },
    'max_steps': {
        'type': 'int',
        'description': 'Maximum number of agent steps per sample.',
        'value': 50,
    },
    'judge_context_limit': {
        'type': 'int',
        'description': 'Estimated token limit before rubric judging switches to chunking.',
        'value': 150000,
    },
    'judge_chunk_size': {
        'type': 'int',
        'description': 'Maximum estimated tokens in each document chunk sent to the judge.',
        'value': 100000,
    },
    'judge_retries': {
        'type': 'int',
        'description': 'Maximum attempts for each rubric judge request and JSON parse.',
        'value': 3,
    },
}

_DESCRIPTION = """
## Overview

ResearchRubrics evaluates Deep Research agents on realistic, open-ended research tasks. Each task pairs a user prompt
with expert-written, fine-grained rubrics covering explicit and implicit requirements, information synthesis,
references, communication quality, and instruction following.

## Task Description

- **Task Type**: Multi-turn research agent / long-form report generation
- **Input**: One open-ended research prompt
- **Output**: A Markdown research report produced after iterative tool use
- **Dataset**: 101 tasks and 2,593 weighted rubric criteria
- **Metric**: Binary rubric compliance score

## Agent Runtime

- Uses EvalScope's built-in AgentLoop with the ``function_calling`` strategy and a ``bash`` tool by default.
- The default bash tool runs in a per-sample temporary directory through ``LocalAgentEnvironment`` and uses the host
  network. This environment is not a security sandbox: absolute paths can still access the host filesystem. Do not use
  the default runtime with untrusted models on shared or sensitive machines.
- ``dataset_args.extra_params.strategy`` can be set to ``react``. Both built-in strategies require native function
  calling; ReAct additionally injects a Think -> Act -> Observation system prompt.
- Optional MCP servers from ``NativeAgentConfig.mcp_servers`` are merged with bash. ``ExternalAgentConfig`` routes the
  prompt through EvalScope's external agent bridge instead.
- For this benchmark-owned AgentLoop, strategy and max steps are configured through ``dataset_args.extra_params``;
  corresponding fields on ``NativeAgentConfig`` are not used.
- If the native loop exhausts ``max_steps`` while still calling tools, the benchmark makes one final tool-free model
  call so the gathered research is preserved as a reviewable Markdown report.

## Evaluation Notes

- ResearchRubrics requires ``judge_model_args`` and ``judge_strategy='auto'`` or ``'llm'``. Gemini 2.5 Pro is the
  recommended judge for comparison with the paper, but no provider or model is hard-coded.
- Every rubric is graded independently as Satisfied (1) or Not Satisfied (0), matching the public binary grader. The
  paper's ternary scores are not directly comparable.
- Negative-weight criteria subtract from the numerator when the undesirable behavior is present. Scores are not
  clipped.
- Long reports are evaluated with the official chunk-evidence-synthesis approach when they exceed the configured judge
  context threshold.
- A full run performs 2,593 rubric evaluations and can be expensive. Current-events tasks are also sensitive to the
  date and web sources available at evaluation time.

## Configuration

- ``strategy``: ``function_calling`` (default) or ``react``
- ``max_steps``: 50 by default
- ``judge_context_limit``: 150,000 estimated tokens
- ``judge_chunk_size``: 100,000 estimated tokens
- ``judge_retries``: 3 attempts per judge request

The judge must be configured explicitly. For example:

```python
from evalscope import TaskConfig, run_task

run_task(TaskConfig(
    model='YOUR_AGENT_MODEL',
    datasets=['researchrubrics'],
    judge_strategy='llm',
    judge_model_args={
        'model_id': 'YOUR_JUDGE_MODEL',
        'api_url': 'OPENAI_COMPATIBLE_JUDGE_URL',
        'api_key': 'YOUR_JUDGE_API_KEY',
        'generation_config': {'temperature': 0.0},
    },
    limit=1,
))
```

Resources: [Paper](https://arxiv.org/abs/2511.07685) |
[GitHub](https://github.com/scaleapi/researchrubrics) |
[Dataset](https://modelscope.cn/datasets/evalscope/researchrubrics)
"""


@register_benchmark(
    BenchmarkMeta(
        name='researchrubrics',
        pretty_name='ResearchRubrics',
        tags=[Tags.AGENT, Tags.MULTI_TURN, Tags.RETRIEVAL, Tags.REASONING],
        description=_DESCRIPTION,
        dataset_id='evalscope/researchrubrics',
        paper_url='https://arxiv.org/abs/2511.07685',
        subset_list=['default'],
        default_subset='default',
        eval_split='train',
        metric_list=['compliance_score'],
        prompt_template='{question}',
        extra_params=_EXTRA_PARAMS,
    )
)
class ResearchRubricsAdapter(AgentLoopAdapter):
    """Deep Research agent benchmark with binary rubric-based LLM judging."""

    strategy_name = 'function_calling'
    max_steps_default = 50

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.strategy_name = str(self.extra_params.get('strategy', self.strategy_name))
        self.max_steps = int(self.extra_params.get('max_steps', self.max_steps_default))
        self.judge_context_limit = int(self.extra_params.get('judge_context_limit', 150000))
        self.judge_chunk_size = int(self.extra_params.get('judge_chunk_size', 100000))
        self.judge_retries = int(self.extra_params.get('judge_retries', 3))
        if self.max_steps <= 0:
            raise ValueError('ResearchRubrics max_steps must be greater than 0.')
        if self.judge_context_limit <= 0 or self.judge_chunk_size <= 0:
            raise ValueError('ResearchRubrics judge context and chunk limits must be greater than 0.')
        if self.judge_retries <= 0:
            raise ValueError('ResearchRubrics judge_retries must be greater than 0.')
        self._use_llm_judge = True
        self.use_batch_scoring = True

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        rubrics = record.get('rubrics')
        if not isinstance(rubrics, list):
            raise ValueError('ResearchRubrics record must contain a rubrics list.')
        return Sample(
            input=str(record['prompt']),
            target=json.dumps(rubrics, ensure_ascii=False),
            tools=[BASH_TOOL_INFO],
            metadata={
                'sample_id': record.get('sample_id'),
                'domain': record.get('domain'),
                'conceptual_breadth': record.get('conceptual_breadth'),
                'logical_nesting': record.get('logical_nesting'),
                'exploration': record.get('exploration'),
            },
        )

    def build_strategy(self, sample: Sample) -> AgentStrategy:
        strategy_cls = get_strategy(self.strategy_name)
        return strategy_cls()

    def build_tools(self, sample: Sample) -> Dict[str, Any]:
        return {'bash': run_bash}

    def build_environment(self, sample: Sample) -> Optional[AgentEnvironment]:
        sample_id = sample.metadata.get('sample_id') or sample.id or 'unknown'
        return TemporaryLocalAgentEnvironment(sample_id=sample_id)

    def _on_inference(self, model: Any, sample: Sample) -> InferenceResult:
        result = super()._on_inference(model, sample)
        if result.output.completion.strip() or not self._reached_max_steps(result):
            return result

        finalization_message = ChatMessageUser(
            content=(
                'The tool-use budget is exhausted. Using the research already gathered, write and return the complete '
                'final Markdown report now. Do not call any tools.'
            )
        )
        finalization_input = list(result.messages or []) + [finalization_message]
        final_output = model.generate(input=finalization_input, tools=None)
        messages = finalization_input + [final_output.message]

        if result.trace is not None:
            step = result.trace.max_steps
            result.trace.add_event(
                step=step,
                type=EventType.NUDGE,
                message_id=finalization_message.id,
                payload={'reason': 'max_steps_finalization'},
            )
            usage = None
            if final_output.usage is not None:
                usage = {
                    'input': final_output.usage.input_tokens,
                    'output': final_output.usage.output_tokens,
                    'total': final_output.usage.total_tokens,
                }
            result.trace.add_event(
                step=step,
                type=EventType.MODEL_GENERATE,
                message_id=final_output.message.id,
                token_usage=usage,
                payload={
                    'stop_reason': final_output.stop_reason,
                    'phase': 'max_steps_finalization'
                },
            )
            if final_output.completion.strip():
                result.trace.add_event(
                    step=step,
                    type=EventType.SUBMIT,
                    message_id=final_output.message.id,
                    payload={
                        'final_answer': final_output.completion,
                        'phase': 'max_steps_finalization'
                    },
                )
                if result.trace.total_usage is not None and final_output.usage is not None:
                    result.trace.total_usage += final_output.usage

        return InferenceResult(output=final_output, messages=messages, trace=result.trace)

    @staticmethod
    def _reached_max_steps(result: InferenceResult) -> bool:
        if result.trace is None:
            return False
        return any(
            event.type == EventType.ERROR and event.payload.get('message') == 'max_steps_exceeded'
            for event in result.trace.events
        )

    def calculate_metrics(self, task_state: TaskState) -> SampleScore:
        """Return a placeholder; expensive rubric judging runs after predictions are persisted."""
        return SampleScore(
            score=Score(
                extracted_prediction=task_state.output.completion,
                prediction=task_state.output.completion,
                value={},
            ),
            sample_id=task_state.sample_id,
            group_id=task_state.group_id,
            sample_metadata=task_state.metadata,
        )

    def batch_calculate_metrics(
        self,
        task_states: List[TaskState],
        sample_scores: List[SampleScore],
    ) -> List[SampleScore]:
        if not task_states:
            return sample_scores
        self._validate_judge_config()
        self.llm_judge  # Initialize once before worker threads access it.
        workers = min(max(int(self._task_config.eval_batch_size), 1), len(task_states))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            reviewed_scores = list(executor.map(self._score_task_state, task_states))
        for sample_score, reviewed_score in zip(sample_scores, reviewed_scores):
            sample_score.score = reviewed_score
        return sample_scores

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        if not sample_scores:
            return []

        aggregate_groups: Dict[str, List[SampleScore]] = {'compliance_score': sample_scores}
        for field in ['domain', 'conceptual_breadth', 'logical_nesting', 'exploration']:
            grouped: Dict[str, List[SampleScore]] = defaultdict(list)
            for sample_score in sample_scores:
                value = (sample_score.sample_metadata or {}).get(field)
                if value:
                    grouped[str(value)].append(sample_score)
            for value, scores in grouped.items():
                aggregate_groups[f'{field}/{value}'] = scores

        axis_names = []
        for sample_score in sample_scores:
            for key in sample_score.score.value:
                if key.startswith('axis/') and key not in axis_names:
                    axis_names.append(key)

        results = [self._mean_agg_score('compliance_score', sample_scores, 'compliance_score')]
        for axis_name in axis_names:
            scores = [sample_score for sample_score in sample_scores if axis_name in sample_score.score.value]
            results.append(self._mean_agg_score(axis_name, scores, axis_name))
        for metric_name, scores in aggregate_groups.items():
            if metric_name == 'compliance_score':
                continue
            results.append(self._mean_agg_score(metric_name, scores, 'compliance_score'))
        return results

    def _score_task_state(self, task_state: TaskState) -> Score:
        report = task_state.output.completion or ''
        if not report.strip():
            raise ValueError(f'ResearchRubrics sample {task_state.metadata.get("sample_id")} produced an empty report.')
        try:
            rubrics = json.loads(task_state.target)
        except json.JSONDecodeError as exc:
            raise ValueError('ResearchRubrics target must be a JSON-encoded rubric list.') from exc
        if not isinstance(rubrics, list) or not rubrics:
            raise ValueError('ResearchRubrics target must contain at least one rubric.')

        rubric_results: List[Dict[str, Any]] = []
        axis_entries: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        source_sample_id = task_state.metadata.get('sample_id') or task_state.sample_id
        for index, rubric in enumerate(rubrics):
            if not isinstance(rubric, dict):
                raise ValueError(f'ResearchRubrics rubric {index} must be an object.')
            result = self._judge_rubric(
                report=report,
                rubric=rubric,
                index=index,
                sample_id=source_sample_id,
            )
            rubric_results.append(result)
            axis_entries[result['axis']].append(result)

        compliance_score = self._weighted_compliance(rubric_results)
        values: Dict[str, float] = {'compliance_score': compliance_score}
        for axis, entries in axis_entries.items():
            if any(entry['weight'] > 0 for entry in entries):
                values[f'axis/{axis}'] = self._weighted_compliance(entries)

        trace = task_state.agent_trace
        tool_names = {'bash'}
        if trace:
            for event in trace.events:
                tool_name = event.payload.get('name') or event.payload.get('tool_name') or event.payload.get('function')
                if tool_name:
                    tool_names.add(str(tool_name))

        return Score(
            extracted_prediction=report,
            prediction=report,
            value=values,
            explanation=f'Binary rubric compliance across {len(rubric_results)} criteria.',
            metadata={
                'source': 'researchrubrics_binary_judge',
                'grading_mode': 'binary',
                'judge_model': self.llm_judge.model_id,
                'rubrics': rubric_results,
                'rubric_count': len(rubric_results),
                'used_chunking': any(result['used_chunking'] for result in rubric_results),
                'agent': {
                    'framework': trace.framework if trace else None,
                    'strategy': trace.strategy if trace else self.strategy_name,
                    'environment': trace.environment if trace else 'local',
                    'max_steps': trace.max_steps if trace else self.max_steps,
                    'tools': sorted(tool_names),
                },
            },
            main_score_name='compliance_score',
        )

    def _judge_rubric(self, report: str, rubric: Dict[str, Any], index: int, sample_id: Any) -> Dict[str, Any]:
        criterion = str(rubric.get('criterion', '')).strip()
        axis = str(rubric.get('axis', '')).strip()
        try:
            weight = float(rubric.get('weight'))
        except (TypeError, ValueError) as exc:
            raise ValueError(f'ResearchRubrics rubric {index} has an invalid weight.') from exc
        if not criterion or not axis:
            raise ValueError(f'ResearchRubrics rubric {index} requires criterion and axis.')

        used_chunking = len(report) // 4 > self.judge_context_limit
        if used_chunking:
            result = self._judge_chunked(
                report=report,
                criterion=criterion,
                axis=axis,
                context_prefix=f'sample {sample_id} rubric {index}',
            )
        else:
            prompt = BINARY_USER_PROMPT.format(
                document_content=report,
                rubric_title=criterion,
                rubric_category=axis,
                rubric_weight=weight,
            )
            result = self._request_json(
                prompt=prompt,
                system_prompt=BINARY_SYSTEM_PROMPT,
                validator=validate_binary_result,
                context=f'sample {sample_id} rubric {index}',
            )

        return {
            'index': index,
            'criterion': criterion,
            'axis': axis,
            'weight': weight,
            **result,
            'used_chunking': used_chunking,
        }

    def _judge_chunked(self, report: str, criterion: str, axis: str, context_prefix: str) -> Dict[str, Any]:
        chunks = chunk_document(report, max_tokens=self.judge_chunk_size)
        all_evidence: List[str] = []
        for index, chunk in enumerate(chunks, 1):
            prompt = CHUNK_USER_PROMPT.format(
                chunk_num=index,
                total_chunks=len(chunks),
                context_summary='Previous chunks evaluated' if index > 1 else 'First chunk',
                chunk_content=chunk,
                rubric_title=criterion,
                rubric_category=axis,
            )
            result = self._request_json(
                prompt=prompt,
                system_prompt=CHUNK_SYSTEM_PROMPT,
                validator=validate_chunk_result,
                context=f'{context_prefix} chunk {index}/{len(chunks)}',
            )
            all_evidence.extend(result['relevant_evidence'])

        synthesis_prompt = SYNTHESIS_USER_PROMPT.format(
            all_evidence=json.dumps(all_evidence, ensure_ascii=False, indent=2),
            rubric_title=criterion,
            rubric_category=axis,
        )
        return self._request_json(
            prompt=synthesis_prompt,
            system_prompt=BINARY_SYSTEM_PROMPT,
            validator=validate_binary_result,
            context=f'{context_prefix} chunk synthesis',
        )

    def _request_json(
        self,
        prompt: str,
        system_prompt: str,
        validator: Callable[[Dict[str, Any]], Dict[str, Any]],
        context: str,
    ) -> Dict[str, Any]:
        last_error: Optional[Exception] = None
        last_response = ''
        for attempt in range(self.judge_retries):
            try:
                last_response = self.llm_judge.judge(prompt=prompt, system_prompt=system_prompt)
                return validator(parse_json_object(last_response))
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.judge_retries:
                    time.sleep(2**attempt)
        raise RuntimeError(
            f'ResearchRubrics judge failed for {context} after {self.judge_retries} attempts. '
            f'Last response: {last_response!r}'
        ) from last_error

    def _validate_judge_config(self) -> None:
        if self.judge_strategy not in {JudgeStrategy.AUTO, JudgeStrategy.LLM}:
            raise ValueError(
                'ResearchRubrics requires judge_strategy="auto" or "llm"; rule and llm_recall are not supported.'
            )
        if not self._task_config.judge_model_args:
            raise ValueError('ResearchRubrics requires explicit judge_model_args for binary rubric grading.')

    @staticmethod
    def _weighted_compliance(entries: List[Dict[str, Any]]) -> float:
        denominator = sum(float(entry['weight']) for entry in entries if float(entry['weight']) > 0)
        if denominator <= 0:
            raise ValueError('ResearchRubrics requires at least one positive-weight rubric.')
        numerator = sum(float(entry['score']) * float(entry['weight']) for entry in entries)
        return numerator / denominator

    @staticmethod
    def _mean_agg_score(metric_name: str, sample_scores: List[SampleScore], value_key: str) -> AggScore:
        values = [float(sample_score.score.value[value_key]) for sample_score in sample_scores]
        return AggScore(
            metric_name=metric_name,
            score=sum(values) / len(values),
            aggregation_name='mean',
            num=len(values),
            ids=[sample_score.sample_id for sample_score in sample_scores],
        )
