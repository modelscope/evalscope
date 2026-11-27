"""Adapter that evaluates an externally generated business-agent eval set."""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

import requests
from pydantic import BaseModel

from evalscope.api.agent import AgentTrace, AgentTraceEvent, EventType
from evalscope.api.benchmark import AgentAdapter, BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import InferenceResult
from evalscope.api.judge import (
    JudgeCase,
    JudgeContext,
    JudgeDefinition,
    JudgeRequest,
    OutputContract,
    ReducedVerdict,
)
from evalscope.api.messages import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser, dict_to_chat_message
from evalscope.api.metric import Score
from evalscope.api.metric.semantics import MetricSelector
from evalscope.api.model import ModelOutput, ModelUsage
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.project_eval.agent_clients import (
    AgentProtocolError,
    AgentTurn,
    HttpTaskClient,
    build_client,
    detect_capability,
)
from evalscope.constants import ScoringPolicy
from evalscope.project_eval.checks import CheckResult, run_checks
from evalscope.project_eval.schema import EvalTask


class RubricVerdict(BaseModel):
    explanation: str = ''
    criteria_met: bool


RUBRIC_CONTRACT = OutputContract(schema_model=RubricVerdict, allow_json_fences=True)


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
            '- **Input**: A JSONL task with a question or messages, reference answer, rubric and read-only checks.\n'
            '- **Output**: Agent reply, trace, rubric verdicts and end-state check evidence.\n'
            '- **Domain**: Enterprise business agents with local or remote business state.\n\n'
            '## Key Features\n'
            '- Consumes externally authored eval sets; EvalScope does not read project source.\n'
            '- Supports SQLite, HTTP/JSON, JSON file and file checks.\n'
            '- Supports L0 and L2 inline trace; process criteria are not evaluable without trace.\n\n'
            '## Evaluation Notes\n'
            '- Metrics: task_pass, rubric_score and evidence_coverage; task_pass is primary.\n'
            '- Checks are read-only; a failed must-pass outcome check short-circuits the task.\n'
            '- The feature is gated behind the project_eval service flags.\n'
        ),
        dataset_id='project_eval',
        subset_list=['default'],
        metric_list=['task_pass', 'rubric_score', 'evidence_coverage'],
        primary_metric=MetricSelector(name='task_pass', aggregation='mean'),
        evaluation_version='v1.0',
    )
)
class ProjectEvalAdapter(AgentAdapter):
    """Consume eval_set.v1.jsonl and call the business agent through HTTP."""

    scoring_policy = ScoringPolicy.JUDGE_ONLY
    _runtime_keys = {'project_root', 'runtime_checks', 'agent_capability'}

    def load_from_disk(self, **kwargs: Any) -> Any:
        return super().load_from_disk(use_local_loader=True)

    @classmethod
    def _task_from_metadata(cls, metadata: dict[str, Any]) -> EvalTask:
        return EvalTask.model_validate({key: value for key, value in metadata.items() if key not in cls._runtime_keys})

    def record_to_sample(self, record: dict[str, Any]) -> Sample:
        task = EvalTask.model_validate(record)
        if task.messages:
            input_messages = [dict_to_chat_message(message) for message in task.messages]
        else:
            input_messages = []
            if task.system:
                input_messages.append(ChatMessageSystem(content=task.system))
            input_messages.append(ChatMessageUser(content=task.question))
        return Sample(
            id=abs(hash(task.id)) % (2**31),
            input=input_messages,
            target=task.reference_answer,
            metadata=record,
        )

    def _client_for(self, sample: Sample) -> Any:
        task = self._task_from_metadata(sample.metadata)
        spec = task.exec
        env_name = spec.base_url_env if spec.type == 'http' else spec.agent_url_env
        base_url = ''
        if self._task_config is not None:
            base_url = (self._task_config.dataset_args.get('project_eval') or {}).get('agent_url', '')
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
        if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost'):
            raise AgentProtocolError('reset_url must be a loopback http URL')
        try:
            response = requests.post(url, timeout=10)
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001 - surfaced as a task execution failure
            raise AgentProtocolError(f'reset failed: {exc}') from exc

    def _on_inference(self, model: Any, sample: Sample) -> InferenceResult:
        task = self._task_from_metadata(sample.metadata)
        self._reset_if_configured(task)
        if task.exec.trace.mode == 'fetch':
            raise AgentProtocolError('trace.mode=fetch is not implemented; use auto/inline/none')
        client = self._client_for(sample)
        if task.exec.type == 'http':
            if not isinstance(client, HttpTaskClient):
                raise AgentProtocolError('http task requires HttpTaskClient')
            turn = client.call(task.exec.request)
        else:
            turn = client.chat(task, session_id=str(sample.metadata['id']))
        if task.exec.trace.mode == 'none':
            turn.steps = []
            turn.trace_available = False
        sample.metadata['agent_capability'] = detect_capability(turn)
        sample.metadata['runtime_checks'] = [
            asdict(result) for result in run_checks(task, self._root_for(sample.metadata))
        ]

        output = ModelOutput.from_content(model='project-eval', content=turn.reply)
        if turn.usage:
            output.usage = ModelUsage(
                input_tokens=int(turn.usage.get('input_tokens', 0)),
                output_tokens=int(turn.usage.get('output_tokens', 0)),
                total_tokens=int(turn.usage.get('total_tokens', 0)),
            )
        messages = list(sample.input) + [ChatMessageAssistant(content=turn.reply)]
        return InferenceResult(
            output=output,
            messages=messages,
            trace=build_agent_trace(turn, task),
        )

    def _root_for(self, metadata: dict[str, Any] | None) -> Path:
        metadata = metadata or {}
        root = metadata.get('project_root')
        if not root and self._task_config is not None:
            root = (self._task_config.dataset_args.get('project_eval') or {}).get('project_root')
        if not root:
            root = os.getcwd()
        return Path(root)

    def _task_root(self, context: JudgeContext) -> Path:
        return self._root_for(context.task_state.metadata)

    @staticmethod
    def _saved_checks(metadata: dict[str, Any] | None) -> list[CheckResult] | None:
        records = metadata.get('runtime_checks') if isinstance(metadata, dict) else None
        if not isinstance(records, list):
            return None
        try:
            return [CheckResult(**record) for record in records if isinstance(record, dict)]
        except TypeError:
            return None

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
        saved = self._saved_checks(context.task_state.metadata)
        if saved is not None:
            checks = saved
            check_records = [asdict(result) for result in checks]
        else:
            checks = run_checks(task, self._task_root(context))
            check_records = [asdict(result) for result in checks]
            context.task_state.metadata['runtime_checks'] = check_records
        context.task_state.metadata['runtime_checks'] = check_records

        trace_available = self._trace_available(context.task_state.agent_trace)
        evaluable = []
        not_evaluable = []
        for criterion in task.rubric:
            if criterion.evidence_requirement == 'process' and not trace_available:
                not_evaluable.append(criterion.id)
            else:
                evaluable.append(criterion)
        coverage = len(evaluable) / len(task.rubric) if task.rubric else 0.0

        must_pass_outcome = [
            criterion
            for criterion in task.rubric
            if criterion.must_pass and criterion.evidence_requirement == 'outcome'
        ]
        failed_checks = [result for result in checks if not result.ok]
        if failed_checks and must_pass_outcome:
            score = Score(
                value={'task_pass': 0.0, 'rubric_score': 0.0, 'evidence_coverage': coverage},
                explanation=f'must_pass check failed: {failed_checks[0].check_id}',
                metadata={
                    'checks': check_records,
                    'not_evaluable': not_evaluable,
                },
            )
            return JudgeDefinition.skip(score, reason='must_pass check failed')

        if not evaluable:
            score = Score(
                value={'task_pass': 0.0, 'rubric_score': 0.0, 'evidence_coverage': 0.0},
                explanation='no evaluable rubric criteria without trace',
                metadata={
                    'checks': check_records,
                    'not_evaluable': not_evaluable,
                    'needs_review': any(
                        criterion.must_pass and criterion.id in not_evaluable for criterion in task.rubric
                    ),
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
            placement: str,
            completed_cases: list[JudgeCase],
            judge_context: JudgeContext,
        ) -> JudgeRequest:
            criterion = case.metadata['criterion']
            trace_summary = ''
            conversation_lines = []
            trace = judge_context.task_state.agent_trace
            if trace is not None:
                trace_summary = '\n'.join(
                    f'step {event.step} {event.type.value}: {event.payload}' for event in trace.events
                )
                for event in trace.events:
                    if event.type == EventType.SUBMIT and isinstance(event.payload.get('conversation'), list):
                        for message in event.payload['conversation']:
                            role = str(message.get('role', 'user'))
                            content = str(message.get('content', '')).strip()
                            if content:
                                conversation_lines.append(f'{role}: {content}')
            conversation_block = '\n'.join(conversation_lines)
            prompt = (
                f'You are evaluating a business agent against one rubric criterion.\n\n'
                f'# Criterion (judge only this)\n{criterion["criterion"]}\n\n'
                f'# Reference answer (context only; do not require unrelated facts)\n'
                f'{task.reference_answer or "(not provided)"}\n\n'
                f'# Expected facts (context only; use only when the criterion names them)\n'
                f'{task.expected_facts or "(none)"}\n\n'
                f'# Forbidden claims\n{task.forbidden_facts or "(none)"}\n\n'
                f'# Conversation (full multi-turn transcript; judge criteria against it)\n'
                f'{conversation_block or "(not available)"}\n\n'
                f'# Agent answer (final turn)\n{judge_context.original_prediction}\n\n'
                f'# Tool trace\n{trace_summary or "(not available)"}\n\n'
                f'# End-state checks\n{check_records}\n\n'
                f'Judge ONLY the criterion above. Do not fail it because unrelated reference '
                f'facts are absent. For end-state criteria, the checks are authoritative. '
                f'When a criterion refers to an earlier turn, verify it against the conversation.'
                f'{case.output_contract.instruction()}'
            )
            return JudgeRequest(messages=[ChatMessageUser(content=prompt)])

        def reduce(case_verdicts: list[Any], judge_context: JudgeContext) -> ReducedVerdict:
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
            coverage = len(evaluable) / len(task.rubric) if task.rubric else 0.0
            needs_review = any(criterion.must_pass and criterion.id in not_evaluable for criterion in task.rubric)
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
                },
            )

        def finalize(score: Score, review: Any, judge_context: JudgeContext) -> Score:
            score.metadata.update(review.metadata)
            return score

        return JudgeDefinition.workflow(
            cases=cases,
            request=request,
            reduce=reduce,
            main_score_name='task_pass',
            finalize=finalize,
        )


def build_agent_trace(turn: AgentTurn, task: EvalTask) -> AgentTrace:
    trace = AgentTrace(
        framework='business_agent',
        strategy=task.exec.protocol,
        environment=None,
        max_steps=len(turn.steps),
        trial_id=turn.trace_id,
        total_usage=ModelUsage(
            input_tokens=int(turn.usage.get('input_tokens', 0)),
            output_tokens=int(turn.usage.get('output_tokens', 0)),
            total_tokens=int(turn.usage.get('total_tokens', 0)),
        )
        if turn.usage
        else None,
        events=[],
    )

    step = 0
    trace.add_event(
        step=step,
        type=EventType.RUN_START,
        payload={
            'protocol': task.exec.protocol,
            'trace_available': turn.trace_available,
        },
    )
    for item in turn.steps:
        step += 1
        trace.add_event(
            step=step,
            type=EventType.TOOL_CALL,
            latency_ms=item.elapsed_ms,
            payload={
                'tool': item.name,
                'arguments': item.arguments,
                'confirmed': item.raw.get('confirmed'),
                'phase': item.raw.get('phase'),
                'preview': item.raw.get('preview'),
                'outcome': item.raw.get('outcome'),
            },
        )
        trace.add_event(
            step=step,
            type=EventType.TOOL_RESULT,
            latency_ms=item.elapsed_ms,
            payload={
                'tool': item.name,
                'result': item.result,
                'confirmed': item.raw.get('confirmed'),
                'phase': item.raw.get('phase'),
                'outcome': item.raw.get('outcome'),
            },
        )
    step += 1
    conversation = turn.conversation or [{'role': 'assistant', 'content': turn.reply}]
    trace.add_event(
        step=step,
        type=EventType.SUBMIT,
        payload={'reply': turn.reply, 'conversation': conversation},
    )
    step += 1
    trace.add_event(step=step, type=EventType.RUN_END, payload={'trace_available': turn.trace_available})
    return trace


__all__ = ['ProjectEvalAdapter', 'build_agent_trace']
