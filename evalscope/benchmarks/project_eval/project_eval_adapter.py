"""Adapter that evaluates an externally generated business-agent eval set."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit

import requests
from pydantic import BaseModel

from evalscope.api.agent import AgentTrace, EventType
from evalscope.api.benchmark import AgentAdapter, BenchmarkMeta
from evalscope.api.dataset import DatasetDict, Sample
from evalscope.api.evaluator import InferenceResult
from evalscope.api.judge import (
    CaseVerdict,
    JudgeCase,
    JudgeContext,
    JudgeDefinition,
    JudgeRequest,
    JudgeReview,
    OutputContract,
    Placement,
    ReducedVerdict,
)
from evalscope.api.messages import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser, dict_to_chat_message
from evalscope.api.metric import Score
from evalscope.api.metric.semantics import MetricSelector
from evalscope.api.model import ModelOutput
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.project_eval.agent_clients import (
    AgentClient,
    AgentProtocolError,
    HttpTaskClient,
    build_client,
    detect_capability,
)
from evalscope.benchmarks.project_eval.trace import build_agent_trace, model_usage
from evalscope.constants import ScoreStatus, ScoringPolicy, Tags
from evalscope.project_eval.checks import (
    DEFAULT_HTTP_PREFIXES,
    run_checks,
    saved_checks_from_metadata,
)
from evalscope.project_eval.schema import (
    AGENT_CAPABILITY_KEY,
    AGENT_STATE_KEY,
    PROJECT_ROOT_KEY,
    RUNTIME_CHECKS_KEY,
    EvalTask,
    ProjectEvalRunConfig,
    RubricCriterion,
)


class RubricVerdict(BaseModel):
    """Judge verdict for one rubric criterion."""

    explanation: str = ''
    criteria_met: bool


RUBRIC_CONTRACT = OutputContract(schema_model=RubricVerdict, allow_json_fences=True)


def _trace_context(trace: AgentTrace | None) -> tuple[str, str]:
    if trace is None:
        return '', ''
    trace_summary = '\n'.join(f'step {event.step} {event.type.value}: {event.payload}' for event in trace.events)
    conversation_lines = []
    for event in trace.events:
        if event.type != EventType.SUBMIT or not isinstance(event.payload.get('conversation'), list):
            continue
        for message in event.payload['conversation']:
            role = str(message.get('role', 'user'))
            content = str(message.get('content', '')).strip()
            if content:
                conversation_lines.append(f'{role}: {content}')
    return trace_summary, '\n'.join(conversation_lines)


def _actual_question(task: EvalTask) -> str:
    """Return the user turns that the Agent actually received."""
    if task.messages:
        return '\n'.join(message.content for message in task.messages)
    return task.question


def _render_judge_prompt(
    task: EvalTask,
    criterion: RubricCriterion,
    trace: AgentTrace | None,
    prediction: str,
    check_records: list[dict[str, Any]],
    agent_state: dict[str, Any] | None,
    instruction: str,
) -> str:
    trace_summary, conversation_block = _trace_context(trace)
    state_block = f'\n# Agent state\n{agent_state}\n' if agent_state else ''
    return (
        f'You are evaluating a business agent against one rubric criterion.\n\n'
        f'# Question (the user request being evaluated)\n{_actual_question(task)}\n\n'
        f'# Criterion (judge only this)\n{criterion.criterion}\n\n'
        f'# Reference answer (context only; do not require unrelated facts)\n'
        f'{task.reference_answer or "(not provided)"}\n\n'
        f'# Expected facts (context only; use only when the criterion names them)\n'
        f'{task.expected_facts or "(none)"}\n\n'
        f'# Forbidden claims\n{task.forbidden_facts or "(none)"}\n\n'
        f'# Conversation (full multi-turn transcript; judge criteria against it)\n'
        f'{conversation_block or "(not available)"}\n\n'
        f'# Agent answer (final turn)\n{prediction}\n\n'
        f'# Tool trace\n{trace_summary or "(not available)"}\n\n'
        f'{state_block}'
        f'# End-state checks\n{check_records}\n\n'
        f'Judge ONLY the criterion above. Do not fail it because unrelated reference '
        f'facts are absent. For end-state criteria, the checks are authoritative. '
        f'When a criterion refers to an earlier turn, verify it against the conversation.'
        f'{instruction}'
    )


@register_benchmark(
    BenchmarkMeta(
        name='project_eval',
        pretty_name='Project Eval',
        description=(
            '## Overview\n'
            'Evaluate externally generated business-agent eval sets with rubric judging '
            'and read-only end-state checks.\n\n'
            '## Task Description\n'
            '- **Task Type**: Business-agent conversation with optional multi-turn user scripts.\n'
            '- **Input**: A JSONL task with a required question plus optional user messages, '
            'reference answer, rubric and read-only checks.\n'
            '- **Output**: Agent reply, trace, rubric verdicts and end-state check evidence.\n'
            '- **Domain**: Enterprise business agents with local or remote business state.\n\n'
            '## Key Features\n'
            '- Consumes externally authored eval sets; EvalScope does not read project source.\n'
            '- Supports SQLite, HTTP/JSON, JSON file and file checks.\n'
            '- Supports L0 and L2 inline trace; process criteria are not evaluable without trace.\n\n'
            '## Evaluation Notes\n'
            '- Metrics: task_pass, rubric_score and evidence_coverage; task_pass is primary.\n'
            '- Checks are read-only; any failed check fails a task that carries a must-pass outcome criterion.\n'
            '- `project_root` is required when checks are present; remote state APIs need explicit '
            '`check_url_prefixes`.\n'
            '- `trace.mode=inline` requires inline steps; all-process rubrics without trace are excluded '
            'from metrics and marked `needs_review`.\n'
        ),
        dataset_id='project_eval',
        subset_list=['default'],
        tags=[Tags.AGENT],
        metric_list=['task_pass', 'rubric_score', 'evidence_coverage'],
        primary_metric=MetricSelector(name='task_pass', aggregation='mean'),
        evaluation_version='v1.0',
    )
)
class ProjectEvalAdapter(AgentAdapter):
    """Consume eval_set.v1.jsonl and call the business agent through HTTP."""

    scoring_policy = ScoringPolicy.JUDGE_ONLY
    _runtime_keys = {PROJECT_ROOT_KEY, RUNTIME_CHECKS_KEY, AGENT_CAPABILITY_KEY, AGENT_STATE_KEY}

    def load_from_disk(self, use_local_loader: bool = False) -> tuple[DatasetDict, DatasetDict | None]:
        """Load the local JSONL eval set and reject duplicate task ids."""
        # project_eval eval sets are always local JSONL files.
        dataset_dict, fewshot_dataset = super().load_from_disk(use_local_loader=True)
        self._validate_task_ids(dataset_dict)
        return dataset_dict, fewshot_dataset

    @staticmethod
    def _validate_task_ids(dataset_dict: DatasetDict) -> None:
        seen: set[str] = set()
        for _, dataset in dataset_dict.items():
            for sample in dataset:
                task_id = str(sample.metadata.get('id') or '').strip()
                if not task_id:
                    raise ValueError('task id must be non-empty')
                if task_id in seen:
                    raise ValueError(f'duplicate task id: {task_id}')
                seen.add(task_id)

    @classmethod
    def _task_from_metadata(cls, metadata: dict[str, Any]) -> EvalTask:
        return EvalTask.model_validate({key: value for key, value in metadata.items() if key not in cls._runtime_keys})

    def record_to_sample(self, record: dict[str, Any]) -> Sample:
        task = EvalTask.model_validate(record)
        if task.messages:
            input_messages = [dict_to_chat_message(message.model_dump()) for message in task.messages]
        else:
            input_messages = []
            if task.system:
                input_messages.append(ChatMessageSystem(content=task.system))
            input_messages.append(ChatMessageUser(content=task.question))
        return Sample(
            # The standard loader assigns sequential ids; 0 is a placeholder for direct adapter use.
            id=0,
            input=input_messages,
            target=task.reference_answer,
            metadata=record,
        )

    def _run_config(self) -> ProjectEvalRunConfig:
        if self._task_config is None:
            return ProjectEvalRunConfig()
        raw = self._task_config.dataset_args.get('project_eval') or {}
        return ProjectEvalRunConfig.model_validate(raw)

    def _client_for(self, sample: Sample) -> AgentClient | HttpTaskClient:
        task = self._task_from_metadata(sample.metadata)
        spec = task.exec
        env_name = spec.base_url_env if spec.type == 'http' else spec.agent_url_env
        base_url = self._run_config().agent_url
        base_url = (base_url or os.environ.get(env_name, '')).strip()
        if not base_url:
            raise AgentProtocolError(f'missing agent URL: set {env_name} or dataset_args.agent_url')
        model = os.environ.get('EVAL_AGENT_MODEL', 'business-agent')
        api_key = os.environ.get('EVAL_AGENT_API_KEY', 'EMPTY')
        return build_client(spec, base_url=base_url, model=model, api_key=api_key)

    @staticmethod
    def _reset_if_configured(task: EvalTask) -> None:
        url = task.exec.reset_url.strip()
        if not url:
            return
        parsed = urlsplit(url)
        if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost', '::1'):
            raise AgentProtocolError('reset_url must be a loopback http URL')
        try:
            response = requests.post(url, timeout=10)
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001 - surfaced as a task execution failure
            raise AgentProtocolError(f'reset failed: {exc}') from exc

    def _on_inference(self, model: Any, sample: Sample) -> InferenceResult:
        task = self._task_from_metadata(sample.metadata)
        root = self._root_for(task, sample.metadata)
        self._reset_if_configured(task)
        client = self._client_for(sample)
        if task.exec.type == 'http':
            if not isinstance(client, HttpTaskClient):
                raise AgentProtocolError('http task requires HttpTaskClient')
            turn = client.call(task.exec.request)
        else:
            turn = cast(AgentClient, client).chat(task, session_id=str(sample.metadata['id']))
        if task.exec.trace.mode == 'none':
            turn.steps = []
            turn.trace_id = None
            turn.trace_available = False
        elif task.exec.trace.mode == 'inline' and not turn.steps:
            raise AgentProtocolError('trace.mode=inline requires inline steps')
        sample.metadata[AGENT_CAPABILITY_KEY] = detect_capability(turn)
        if turn.state:
            sample.metadata[AGENT_STATE_KEY] = turn.state
        prefixes = tuple(DEFAULT_HTTP_PREFIXES) + tuple(self._run_config().check_url_prefixes)
        sample.metadata[RUNTIME_CHECKS_KEY] = [result.model_dump() for result in run_checks(task, root, prefixes)]

        output = ModelOutput.from_content(model=getattr(model, 'name', None) or 'project-eval', content=turn.reply)
        output.usage = model_usage(turn.usage)
        base_messages = sample.input if isinstance(sample.input, list) else [ChatMessageUser(content=sample.input)]
        messages = [*base_messages, ChatMessageAssistant(content=turn.reply)]
        return InferenceResult(
            output=output,
            messages=messages,
            trace=build_agent_trace(turn, task),
        )

    def _root_for(self, task: EvalTask, metadata: dict[str, Any] | None) -> Path:
        metadata = metadata or {}
        root = metadata.get(PROJECT_ROOT_KEY) or self._run_config().project_root
        if not root:
            if task.checks:
                raise AgentProtocolError(
                    'missing project_root: configure dataset_args.project_eval.project_root for read-only checks'
                )
            root = os.getcwd()
        return Path(root)

    @staticmethod
    def _trace_available(trace: AgentTrace | None) -> bool:
        if trace is None:
            return False
        for event in trace.events:
            if event.type in (EventType.TOOL_CALL, EventType.TOOL_RESULT, EventType.MODEL_GENERATE):
                return True
            if event.type == EventType.RUN_START and event.payload.get('trace_available') is False:
                return False
        return False

    def judge_definition(self, context: JudgeContext) -> JudgeDefinition:
        task = self._task_from_metadata(context.task_state.metadata)
        saved = saved_checks_from_metadata(context.task_state.metadata)
        if task.checks:
            if saved is None or [result.check_id for result in saved] != [check.id for check in task.checks]:
                raise AgentProtocolError(
                    'missing saved end-state checks: cached evidence does not match the task checks'
                )
        else:
            saved = []
        checks = saved
        check_records = [result.model_dump() for result in checks]
        capability = context.task_state.metadata.get(AGENT_CAPABILITY_KEY)
        raw_agent_state = context.task_state.metadata.get(AGENT_STATE_KEY)
        agent_state = raw_agent_state if isinstance(raw_agent_state, dict) else None

        trace_available = self._trace_available(context.task_state.agent_trace)
        evaluable = []
        not_evaluable = []
        for criterion in task.rubric:
            if criterion.evidence_requirement == 'process' and not trace_available:
                not_evaluable.append(criterion.id)
            else:
                evaluable.append(criterion)
        coverage = len(evaluable) / len(task.rubric)
        needs_review = any(criterion.must_pass and criterion.id in not_evaluable for criterion in task.rubric)

        has_must_pass_outcome = any(
            criterion.must_pass and criterion.evidence_requirement == 'outcome' for criterion in task.rubric
        )
        failed_checks = [result for result in checks if not result.ok]
        if failed_checks and has_must_pass_outcome:
            score = Score(
                value={'task_pass': 0.0, 'rubric_score': 0.0, 'evidence_coverage': coverage},
                explanation=f'must_pass check failed: {failed_checks[0].check_id}',
                metadata={
                    'checks': check_records,
                    'not_evaluable': not_evaluable,
                    'needs_review': needs_review,
                    'agent_capability': capability,
                    'agent_state': agent_state,
                },
            )
            return JudgeDefinition.skip(score, reason='must_pass check failed')

        if not evaluable:
            score = Score(
                value={},
                status=ScoreStatus.EXCLUDED,
                explanation='no evaluable rubric criteria without trace; sample needs review',
                metadata={
                    'checks': check_records,
                    'not_evaluable': not_evaluable,
                    'needs_review': True,
                    'agent_capability': capability,
                    'agent_state': agent_state,
                },
            )
            return JudgeDefinition.skip(score, reason='no evaluable criteria')

        cases = [
            JudgeCase(
                case_id=criterion.id,
                output_contract=RUBRIC_CONTRACT,
                metadata={'criterion': criterion.model_dump()},
            )
            for criterion in evaluable
        ]

        def request(
            case: JudgeCase,
            placement: Placement,
            completed_cases: Sequence[CaseVerdict],
            judge_context: JudgeContext,
        ) -> JudgeRequest:
            criterion = RubricCriterion.model_validate(case.metadata['criterion'])
            prompt = _render_judge_prompt(
                task,
                criterion,
                judge_context.task_state.agent_trace,
                judge_context.original_prediction,
                check_records,
                agent_state,
                case.output_contract.instruction(),
            )
            return JudgeRequest(messages=[ChatMessageUser(content=prompt)])

        def reduce(case_verdicts: Sequence[CaseVerdict], judge_context: JudgeContext) -> ReducedVerdict:
            by_case = {verdict.case_id: verdict for verdict in case_verdicts}
            verdicts = []
            weighted = 0.0
            total_weight = 0.0
            all_must_pass_met = True
            for criterion in task.rubric:
                if criterion.id in not_evaluable:
                    verdicts.append(
                        {
                            'criterion_id': criterion.id,
                            'verdict': False,
                            'reason': 'not evaluable: trace unavailable',
                            'evidence_refs': [],
                            'not_evaluable': True,
                        }
                    )
                    continue
                verdict = by_case.get(criterion.id)
                met = bool(verdict.value.criteria_met) if verdict is not None and verdict.value is not None else False
                refs = []
                if criterion.evidence_requirement == 'outcome':
                    refs = [{'type': 'check', 'id': result.check_id} for result in checks]
                elif criterion.evidence_requirement == 'process':
                    trace_events = (
                        judge_context.task_state.agent_trace.events if judge_context.task_state.agent_trace else []
                    )
                    refs = [
                        {'type': 'trace', 'step': event.step, 'event': event.type.value}
                        for event in trace_events
                        if event.type in (EventType.TOOL_CALL, EventType.TOOL_RESULT)
                    ]
                verdicts.append(
                    {
                        'criterion_id': criterion.id,
                        'verdict': met,
                        'reason': verdict.value.explanation if verdict is not None and verdict.value else '',
                        'evidence_refs': refs,
                    }
                )
                weighted += criterion.weight if met else 0.0
                total_weight += criterion.weight
                if criterion.must_pass and not met:
                    all_must_pass_met = False
            rubric_score = weighted / total_weight if total_weight else 0.0
            task_pass = all_must_pass_met and rubric_score >= task.pass_threshold
            return ReducedVerdict(
                value={
                    'task_pass': float(task_pass),
                    'rubric_score': rubric_score,
                    'evidence_coverage': coverage,
                },
                metadata={
                    'rubric_verdicts': verdicts,
                    'not_evaluable': not_evaluable,
                    'needs_review': needs_review,
                    'checks': check_records,
                    'agent_capability': capability,
                    'agent_state': agent_state,
                },
            )

        def finalize(score: Score, review: JudgeReview, judge_context: JudgeContext) -> Score:
            score.metadata = {**(score.metadata or {}), **review.metadata}
            return score

        return JudgeDefinition.workflow(
            cases=cases,
            request=request,
            reduce=reduce,
            main_score_name='task_pass',
            finalize=finalize,
        )


__all__ = ['ProjectEvalAdapter']
