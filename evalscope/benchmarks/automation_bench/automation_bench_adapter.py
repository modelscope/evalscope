import json
from typing import Any, Dict, List, Tuple

from evalscope.api.agent import NativeAgentConfig
from evalscope.api.benchmark import AgentAdapter, BenchmarkMeta
from evalscope.api.dataset import DatasetDict, Sample, build_dataset_dict_from_record_map
from evalscope.api.evaluator import InferenceResult
from evalscope.api.metric import Score
from evalscope.api.model import Model, ModelOutput, ModelUsage
from evalscope.api.registry import register_benchmark
from evalscope.constants import EvalType, Tags
from evalscope.utils.argument_utils import get_secret_value
from .utils import (
    DEFAULT_AUTOMATION_BENCH_PACKAGE,
    DEFAULT_AUTOMATION_BENCH_VERIFIERS,
    PUBLIC_DOMAINS,
    convert_automation_bench_messages,
    ensure_automation_bench_runtime,
    load_automation_bench_records,
    run_automation_bench_task,
)

_EXTRA_PARAMS = {
    'toolset': {
        'type': 'str',
        'description': 'Official tool style exposed to the agent.',
        'value': 'api',
        'choices': ['api', 'zapier', 'limited_zapier'],
    },
}

_DESCRIPTION = f"""
## Overview

AutomationBench evaluates agents on realistic business workflows across sales, marketing, operations, support,
finance, and HR. EvalScope runs the public tasks, simulated SaaS services, and assertion-based scoring provided by
Zapier's official Python package.

## Task Description

- **Task Type**: Stateful business workflows across 47 simulated SaaS tools.
- **Public Dataset**: 600 tasks, with 100 tasks in each of `sales`, `marketing`, `operations`, `support`, `finance`,
  and `hr`.
- **Simple Baseline**: The optional `simple` subset contains 200 foundational single- and two-step tasks. Its metrics
  use the `simple_` prefix and are reported separately from the public benchmark score.
- **Scoring**: `partial_credit` is the fraction of scored assertions satisfied. `pass_rate` is the mean official
  `task_completed_correctly` signal, which is 1 only when every scored assertion passes.

## Evaluation Notes

- Prepare a Python 3.13+ environment and install the pinned dependencies before evaluation:
  `python -m pip install "{DEFAULT_AUTOMATION_BENCH_VERIFIERS}" "{DEFAULT_AUTOMATION_BENCH_PACKAGE}"`.
- By default, EvalScope evaluates all six public domains (600 tasks). Use `subset_list` to select domains or `limit`
  for a smaller run.
- Use an API-backed EvalScope model with `openai_api`, `openai_responses_api`, or `anthropic_api` evaluation type.
- No Docker runtime, external dataset download, or real SaaS credentials are required. Model API credentials are still
  required.
- The released tasks are public. Zapier's official leaderboard uses a separate held-out private set, so local public
  scores are directional rather than leaderboard-identical.
"""


@register_benchmark(
    BenchmarkMeta(
        name='automation_bench',
        pretty_name='AutomationBench',
        dataset_id='https://github.com/zapier/AutomationBench',
        tags=[Tags.AGENT, Tags.FUNCTION_CALLING, Tags.MULTI_TURN],
        description=_DESCRIPTION,
        paper_url='https://arxiv.org/abs/2604.18934',
        metric_list=['pass_rate', 'partial_credit', 'error_rate'],
        aggregation='mean',
        eval_split='test',
        subset_list=PUBLIC_DOMAINS,
        default_subset='sales',
        extra_params=_EXTRA_PARAMS,
    )
)
class AutomationBenchAdapter(AgentAdapter):
    """EvalScope wrapper around the official AutomationBench Python environment."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._task_records: Dict[str, Dict[str, Any]] = {}
        ensure_automation_bench_runtime()

    def load(self) -> Tuple[DatasetDict, None]:
        domains = list(self.subset_list)
        official_records = load_automation_bench_records(domains)
        records_by_domain: Dict[str, List[Dict[str, Any]]] = {}
        self._task_records.clear()

        for domain in domains:
            records_by_domain[domain] = []
            for index, record in enumerate(official_records[domain]):
                record_key = f'{domain}:{index}'
                self._task_records[record_key] = dict(record)
                records_by_domain[domain].append({
                    'record_key': record_key,
                    'domain': domain,
                    'example_id': record.get('example_id'),
                    'task': record.get('task', record_key),
                    'prompt': record.get('prompt') or [],
                    'answer': record.get('answer') or '',
                })

        return build_dataset_dict_from_record_map(
            record_map=records_by_domain,
            sample_fields=self.record_to_sample,
            location=self.dataset_id,
            limit=self.limit,
            repeats=self.repeats,
            shuffle=self.shuffle,
            seed=self.seed,
        ), None

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        return Sample(
            input=convert_automation_bench_messages(record['prompt']),
            target=record.get('answer') or '',
            subset_key=record['domain'],
            metadata={
                'record_key': record['record_key'],
                'domain': record['domain'],
                'task': record['task'],
                'official_example_id': record.get('example_id'),
            },
        )

    def _on_inference(self, model: Model, sample: Sample) -> InferenceResult:
        record_key = str(sample.metadata.get('record_key') or '')
        task_record = self._task_records.get(record_key)
        if task_record is None:
            raise ValueError(f'AutomationBench task record is unavailable: {record_key!r}.')

        result = run_automation_bench_task(
            task_record=task_record,
            model_name=model.name,
            model_api=model.api,
            api=self._resolve_api_protocol(),
            toolset=self.extra_params['toolset'],
            max_turns=self._resolve_max_turns(),
            sampling_args=self._build_sampling_args(model),
            extra_headers=dict(get_secret_value(model.config.extra_headers) or {}),
        )
        metrics = result.get('metrics') or {}
        content = json.dumps(
            {
                'task': result.get('task'),
                'partial_credit': metrics.get('partial_credit', 0.0),
                'task_completed_correctly': metrics.get('task_completed_correctly', 0.0),
                'error': result.get('error'),
            },
            ensure_ascii=False,
        )
        output = ModelOutput.from_content(
            model=model.name,
            content=content,
            error=str(result['error']) if result.get('error') else None,
        )
        usage = result.get('usage') or {}
        output.usage = ModelUsage(
            input_tokens=int(usage.get('input_tokens', 0) or 0),
            output_tokens=int(usage.get('output_tokens', 0) or 0),
            total_tokens=int(usage.get('input_tokens', 0) or 0) + int(usage.get('output_tokens', 0) or 0),
        )
        messages = convert_automation_bench_messages(result.pop('messages', []))
        output.metadata = result
        return InferenceResult(output=output, messages=messages or None)

    def match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: Any,
    ) -> Score:
        result = getattr(task_state.output, 'metadata', None) or {}
        metrics = result.get('metrics') or {}
        error = result.get('error')
        prefix = 'simple_' if task_state.metadata.get('domain') == 'simple' else ''
        pass_rate = f'{prefix}pass_rate'
        return Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
            value={
                pass_rate: float(metrics.get('task_completed_correctly', 0.0) or 0.0),
                f'{prefix}partial_credit': float(metrics.get('partial_credit', 0.0) or 0.0),
                f'{prefix}error_rate': float(bool(error)),
            },
            main_score_name=pass_rate,
            metadata={
                'task': result.get('task'),
                'error': error,
                'usage': result.get('usage') or {},
                'debug': result.get('debug') or {},
                'assertion_results': result.get('assertion_results') or [],
                'end_state': result.get('end_state'),
                'perf': result.get('perf') or {},
            },
        )

    def _resolve_api_protocol(self) -> str:
        eval_type = self._task_config.eval_type if self._task_config is not None else None
        if eval_type == EvalType.ANTHROPIC_API:
            return 'anthropic'
        if eval_type == EvalType.OPENAI_RESPONSES_API:
            return 'responses'
        return 'chat_completions'

    def _resolve_max_turns(self) -> int:
        agent_config = self._task_config.agent_config if self._task_config is not None else None
        if isinstance(agent_config, NativeAgentConfig) and 'max_steps' in agent_config.model_fields_set:
            return agent_config.max_steps
        return 50

    @staticmethod
    def _build_sampling_args(model: Model) -> Dict[str, Any]:
        fields = ('temperature', 'top_p', 'max_tokens', 'reasoning_effort', 'extra_body')
        return {
            field: getattr(model.config, field)
            for field in fields
            if getattr(model.config, field, None) is not None
        }
