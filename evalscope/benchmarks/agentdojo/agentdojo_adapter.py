"""AgentDojo benchmark adapter.

The upstream ``agentdojo`` package is never imported at module level. ``load_upstream()`` runs only when the adapter
is built with a ``TaskConfig`` (or from ``load_subset()`` for the docs statistics path), so the module stays
importable when the optional dependency is not installed.

Differences from the upstream runner:

- Model and environment errors are raised to the evaluator instead of being scored; with ``ignore_errors`` the sample
  is skipped and retried on a ``use_cache`` resume. A context-length overflow is scored as a failed user task and a
  successful attack, as upstream.
- Samples with a ``content_filter`` reply, missing or invalid persisted data, a raising upstream check, or a cached
  payload from another task, attack or victim model name are ``EXCLUDED`` from the metrics.
- An empty final answer is scored as an empty string; upstream reruns the pipeline up to 3 times.
- Tool calls with invalid JSON arguments are answered with an error and left out of the trace given to the upstream
  checks. A ``TypeError`` from ``tool_result_to_str`` is returned to the model as an observation; upstream aborts.
- An empty function name ``''`` gets the same error as the upstream Google-client sentinel (see ``tool_binding``).
- ``agent_config.max_steps`` (default 15) counts model generations; upstream ``max_iters=15`` allows 16 LLM calls.
- Scoring restores the persisted final environment without re-validating its fields, so ``rerun_review`` and a
  ``use_cache`` resume reproduce the same scores without calling the model or re-executing tools.
"""

import re
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type

from evalscope.api.agent import NativeAgentConfig
from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.benchmark.adapters import AgentLoopAdapter
from evalscope.api.dataset import DataLoader, Dataset, DictDataLoader, Sample
from evalscope.api.evaluator import InferenceResult, TaskState
from evalscope.api.messages import ChatMessageSystem, ChatMessageUser
from evalscope.api.metric import AggScore, MetricSelector, SampleScore, Score
from evalscope.api.model import Model
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.agentdojo.constants import (
    BENCHMARK_VERSION,
    BENIGN_ATTACK,
    DEFAULT_ATTACK,
    RUN_RECORD_META_KEY,
    SAMPLE_META_KEY,
    SUITES,
    SUPPORTED_ATTACKS,
    UPSTREAM_VERSION,
)
from evalscope.benchmarks.agentdojo.injections import InjectionPlanner, resolve_victim_model_name
from evalscope.benchmarks.agentdojo.scoring import (
    AgentDojoRunRecord,
    AgentDojoSampleMeta,
    ScoreMetadata,
    ScoringPayload,
    ScoringPayloadError,
    TerminationReason,
    check_results,
    create_environment,
    map_termination,
)
from evalscope.benchmarks.agentdojo.tool_binding import AgentDojoStrategy, AgentDojoToolBinding
from evalscope.benchmarks.agentdojo.upstream import UpstreamContext, load_upstream
from evalscope.constants import JudgeStrategy, ScoreStatus, Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

SUPPORTED_STRATEGY = 'function_calling'

_TASK_NUMBER_PATTERN = re.compile(r'_(\d+)$')

EXTRA_PARAMS = {
    'attack': {
        'type': 'str',
        'description': "Prompt-injection attack to run. Use 'none' for a benign run without injections.",
        'value': DEFAULT_ATTACK,
        'choices': list(SUPPORTED_ATTACKS),
    },
    'victim_model_name': {
        'type': 'str',
        'description': (
            'Model name used in attack texts that mention the target model. Defaults to a name derived from the '
            'model id.'
        ),
        'value': None,
    },
    'user_task_ids': {
        'type': 'list',
        'description': 'Only evaluate these user tasks, e.g. ["user_task_0"]. Requires a single suite.',
        'value': None,
    },
    'injection_task_ids': {
        'type': 'list',
        'description': (
            'Only evaluate these injection tasks, e.g. ["injection_task_0"]. Requires a single suite; '
            'ignored for benign runs.'
        ),
        'value': None,
    },
}

DESCRIPTION = f"""
## Overview

AgentDojo evaluates how robust tool-using LLM agents are to prompt injection. The agent solves realistic user tasks in
simulated environments while some tool outputs contain injected instructions that try to make it carry out an
attacker's goal.

## Task Description

- **Task Type**: Tool-use agent security (prompt injection)
- **Input**: A user task and the tools of a simulated environment
- **Output**: Tool calls followed by a final answer
- **Domain**: Workspace (email, calendar, cloud drive), travel booking, online banking, Slack

## Key Features

- Four suites: `workspace`, `travel`, `banking` and `slack`, with 97 user tasks and 35 injection tasks (AgentDojo
  `{BENCHMARK_VERSION}`).
- An attack run pairs every user task with every injection task of its suite (949 samples); a benign run
  (`attack='none'`) uses each user task once (97 samples).
- 11 fixed-template attacks from the paper, selected with `extra_params.attack` (default `{DEFAULT_ATTACK}`).
- Results are scored by checking the final environment state with the official deterministic task checks; no LLM
  judge is used.

## Evaluation Notes

- Metrics: `utility` (the user task was completed, higher is better) and `targeted_asr` (the attacker's goal was
  achieved, lower is better). The primary metric is `targeted_asr` for attack runs and `utility` for benign runs.
- Run once with `attack='none'` for benign utility, and once with an attack for utility under attack and attack
  success rate.
- Install with `pip install 'evalscope[agentdojo]'`. Samples are built from the installed
  `agentdojo=={UPSTREAM_VERSION}` package, so no dataset is downloaded.
- Uses EvalScope's native agent loop with the `function_calling` strategy; the default step budget is 15 model calls
  (`agent_config.max_steps`).
- Samples that fail to run (with `ignore_errors` enabled), cannot be scored, or whose reply is blocked by the
  provider's content filter are left out of the metrics, so `num` can be lower than the number of samples.
- Not supported: DoS and manual attacks, prompt-injection defenses and the `workspace_plus` suite.
""".strip()


def _check_task_ids(name: str, value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(
            f'AgentDojo extra_params.{name} must be a list of strings or null, got {type(value).__name__}: {value!r}.'
        )
    return list(value)


def _task_number(task_id: str) -> int:
    """Return the numeric suffix of ``user_task_<n>`` / ``injection_task_<n>`` for deterministic ordering."""
    match = _TASK_NUMBER_PATTERN.search(task_id)
    if match is None:
        raise ValueError(f'AgentDojo task id {task_id!r} has no numeric suffix.')
    return int(match.group(1))


def _select_task_ids(suite_name: str, param: str, tasks: Mapping[str, Any], wanted: Optional[List[str]]) -> List[str]:
    """Return the suite's task ids in numeric order, restricted to ``wanted`` when it is non-empty."""
    unknown = [task_id for task_id in wanted or [] if task_id not in tasks]
    if unknown:
        raise ValueError(f'AgentDojo suite {suite_name!r} has no task(s) matching extra_params.{param} {unknown}.')
    return sorted((task_id for task_id in tasks if not wanted or task_id in wanted), key=_task_number)


def row_id(suite: str, user_task_id: str, injection_task_id: Optional[str]) -> str:
    """Return the sample id ``<suite>-<user_task_id>[-<injection_task_id>]``."""
    if injection_task_id is None:
        return f'{suite}-{user_task_id}'
    return f'{suite}-{user_task_id}-{injection_task_id}'


@register_benchmark(
    BenchmarkMeta(
        name='agentdojo',
        pretty_name='AgentDojo',
        tags=[Tags.AGENT, Tags.SAFETY, Tags.FUNCTION_CALLING],
        description=DESCRIPTION,
        dataset_id='https://github.com/ethz-spylab/agentdojo',
        subset_list=list(SUITES),
        eval_split='test',
        metric_list=['utility', 'targeted_asr'],
        primary_metric='targeted_asr',
        aggregation='mean',
        evaluation_version='v1.0',
        few_shot_num=0,
        extra_params=EXTRA_PARAMS,
        paper_url='https://arxiv.org/abs/2406.13352',
    )
)
class AgentDojoAdapter(AgentLoopAdapter):
    """AgentDojo prompt-injection benchmark on EvalScope's native AgentLoop."""

    max_steps_default = 15
    strategy_name = SUPPORTED_STRATEGY

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Parameter checks never touch the upstream package.
        params = self.extra_params
        self._attack: str = params['attack']
        self._benign: bool = self._attack == BENIGN_ATTACK
        self._validate_subsets()
        self._victim_model_override: Optional[str] = self._validate_victim_model_name(params['victim_model_name'])
        self._user_task_ids = _check_task_ids('user_task_ids', params['user_task_ids'])
        self._injection_task_ids = _check_task_ids('injection_task_ids', params['injection_task_ids'])
        if (self._user_task_ids or self._injection_task_ids) and len(self.subset_list) > 1:
            raise ValueError(
                'AgentDojo task id filters (user_task_ids / injection_task_ids) only support a single suite; '
                f'set dataset_args.agentdojo.subset_list to one suite. Current subset_list: {list(self.subset_list)}.'
            )
        if self._benign and self._injection_task_ids:
            logger.warning(
                f'AgentDojo ignores extra_params.injection_task_ids {self._injection_task_ids} because '
                f'attack={BENIGN_ATTACK!r}: injection_task_ids only applies in attack mode.'
            )

        self._upstream: Optional[UpstreamContext] = None
        self._planner: Optional[InjectionPlanner] = None
        self._victim_model_name: Optional[str] = None
        # ``(suite, sample id) -> row_id`` of the loaded samples; a ``use_cache`` row overwrites the sample metadata,
        # so the loaded identity is kept here.
        self._loaded_row_ids: Dict[Tuple[str, int], str] = {}

        # Run-time checks and upstream loading, only for real runs.
        if self._task_config is not None:
            self._validate_agent_config()
            self._warn_ignored_judge_strategy()
            self._ensure_runtime()

    # ------------------------------------------------------------------
    # Parameter validation
    # ------------------------------------------------------------------

    def _validate_subsets(self) -> None:
        invalid = [subset for subset in self.subset_list if subset not in SUITES]
        if invalid:
            raise ValueError(
                f'Invalid AgentDojo subset(s) {invalid}; valid suites for benchmark version {BENCHMARK_VERSION}: '
                f'{list(SUITES)}.'
            )

    @staticmethod
    def _validate_victim_model_name(value: Any) -> Optional[str]:
        if value is None:
            return None
        if not isinstance(value, str) or not value:
            raise ValueError(
                f'AgentDojo extra_params.victim_model_name must be a non-empty string or null, got {value!r}.'
            )
        return value

    def _validate_agent_config(self) -> None:
        ac = self._task_config.agent_config
        if ac is None:
            return
        if not isinstance(ac, NativeAgentConfig):
            mode = getattr(ac, 'mode', type(ac).__name__)
            raise ValueError(
                f'AgentDojo only supports the native AgentLoop; agent_config mode {mode!r} '
                '(ExternalAgentConfig) is not supported.'
            )
        if ac.strategy != SUPPORTED_STRATEGY:
            raise ValueError(
                f'AgentDojo does not support agent strategy {ac.strategy!r}; '
                f'the only supported strategy is {SUPPORTED_STRATEGY!r}.'
            )
        if ac.tools:
            raise ValueError(
                'AgentDojo tools are fixed by the task suite; NativeAgentConfig.tools must be empty, '
                f'got {list(ac.tools)}.'
            )
        if ac.mcp_servers:
            raise ValueError(
                'AgentDojo tools are fixed by the task suite; NativeAgentConfig.mcp_servers must be empty.'
            )
        if self._resolve_max_steps(ac) <= 0:
            raise ValueError(f'AgentLoop max_steps must be greater than 0, got {ac.max_steps}.')
        if ac.validate_tool_arguments:
            logger.warning(
                'AgentDojo ignores NativeAgentConfig.validate_tool_arguments=True: tool arguments are validated '
                'by the upstream functions runtime.'
            )
        if ac.kwargs:
            logger.warning(
                f'AgentDojo ignores NativeAgentConfig.kwargs {sorted(ac.kwargs)}: the strategy and system prompt '
                'are fixed by the benchmark.'
            )

    def _warn_ignored_judge_strategy(self) -> None:
        """Warn when an LLM judge strategy is configured explicitly; AgentDojo always scores by rule."""
        strategy = self._task_config.judge.strategy
        if strategy in (JudgeStrategy.LLM, JudgeStrategy.LLM_RECALL):
            logger.warning(
                f'AgentDojo ignores judge.strategy={strategy!r}: samples are scored only by the deterministic '
                'upstream utility and security checks, and no LLM judge is called.'
            )

    # ------------------------------------------------------------------
    # Upstream loading
    # ------------------------------------------------------------------

    def _model_id(self) -> str:
        if self._task_config is None:
            return ''
        model = self._task_config.model
        if isinstance(model, str):
            return model
        return self._task_config.model_id or ''

    def _ensure_runtime(self) -> None:
        """Load the upstream suites and build the injection planner once (idempotent)."""
        if self._upstream is not None:
            return
        upstream = load_upstream()
        model_id = self._model_id()
        victim, used_default = resolve_victim_model_name(model_id, self._victim_model_override, upstream.model_names)
        if used_default:
            logger.warning(
                f'AgentDojo could not resolve a victim model name from model id {model_id!r}; using the upstream '
                f'default {victim!r}. Set dataset_args.agentdojo.extra_params.victim_model_name to override it.'
            )
        self._planner = InjectionPlanner(upstream, self._attack, victim)
        self._victim_model_name = victim
        self._upstream = upstream

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_subset(self, subset: str, data_loader: Type[DataLoader]) -> Dataset:
        """Generate one suite's rows from the upstream task suite, then apply limit/shuffle/repeats.

        Task id filters are applied while generating the rows, so they take effect before ``limit``.
        """
        # The docs statistics path constructs the adapter without a TaskConfig, so the runtime is loaded here.
        self._ensure_runtime()
        suite = self._upstream.suites[subset]
        user_task_ids = _select_task_ids(subset, 'user_task_ids', suite.user_tasks, self._user_task_ids)
        injection_task_ids = (
            [None]
            if self._benign
            else _select_task_ids(subset, 'injection_task_ids', suite.injection_tasks, self._injection_task_ids)
        )
        rows = [
            {'suite': subset, 'user_task_id': user_task_id, 'injection_task_id': injection_task_id}
            for user_task_id in user_task_ids
            for injection_task_id in injection_task_ids
        ]
        dataset = DictDataLoader(
            dict_list=rows,
            sample_fields=self.record_to_sample,
            filter_func=self.sample_filter,
            limit=self.limit,
            repeats=self.repeats,
            shuffle=self.shuffle,
            seed=self.seed,
        ).load()
        for sample in dataset:
            self._loaded_row_ids[(subset, sample.id)] = sample.metadata[SAMPLE_META_KEY]['row_id']
        return dataset

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Convert one generated row; attack rows get their injection map here, before inference."""
        suite, user_task_id, injection_task_id = record['suite'], record['user_task_id'], record['injection_task_id']
        meta = AgentDojoSampleMeta(
            row_id=row_id(suite, user_task_id, injection_task_id),
            suite=suite,
            user_task_id=user_task_id,
            injection_task_id=injection_task_id,
            attack=self._attack,
            victim_model_name=self._victim_model_name,
            effective_model_name=self._planner.effective_model_name,
            injections=self._planner.injections_for(suite, user_task_id, injection_task_id),
            benchmark_version=BENCHMARK_VERSION,
            upstream_version=UPSTREAM_VERSION,
        )
        return Sample(
            input=[
                ChatMessageSystem(content=self._upstream.system_message),
                ChatMessageUser(content=self._upstream.suites[suite].user_tasks[user_task_id].PROMPT),
            ],
            target='',
            metadata={SAMPLE_META_KEY: meta.model_dump(mode='json')},
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _on_inference(self, model: Model, sample: Sample) -> InferenceResult:
        """Run one sample on a fresh task environment through the native AgentLoop.

        The model sees exactly the suite tools (no global tools, MCP servers or submit tool) and starts from exactly
        ``sample.input``. Environment-creation and model errors propagate to the evaluator.
        """
        from evalscope.api.agent import run_agent_loop

        meta = AgentDojoSampleMeta.model_validate(sample.metadata[SAMPLE_META_KEY])
        # A new environment per call, so repeats of one sample never share state.
        environment = create_environment(self._upstream, meta)
        binding = AgentDojoToolBinding(self._upstream.suites[meta.suite], environment)
        handlers, tool_infos = binding.build()
        strategy = AgentDojoStrategy(binding)
        ac = self._task_config.agent_config if self._task_config is not None else None
        result = run_agent_loop(
            model=model,
            strategy=strategy,
            handlers=handlers,
            environment=None,
            initial_messages=list(sample.input),
            all_tools=tool_infos,
            max_steps=self._resolve_max_steps(ac),
            sample_id=sample.id,
            trace_strategy_name=strategy.name,
            trace_env_name='agentdojo',
        )

        # ``model_length`` is not an error: the loop ends gracefully and maps to context_length_exceeded.
        reason, steps = map_termination(result)
        final_text = strategy.extract_final_answer(result)
        record = AgentDojoRunRecord(
            final_output=final_text,
            termination_reason=reason,
            executed_steps=steps,
            empty_output=final_text == '',
            tool_call_trace=list(binding.trace),
            post_environment=environment.model_dump(mode='json', round_trip=True),
        )
        sample.metadata[RUN_RECORD_META_KEY] = record.model_dump(mode='json')

        output = result.final_output
        output.completion = final_text
        return InferenceResult(output=output, messages=result.messages, trace=result.trace)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @property
    def judge_strategy(self) -> str:
        """AgentDojo is scored only by deterministic upstream checks; no LLM judge is ever invoked."""
        return JudgeStrategy.RULE

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        """Score one sample from its persisted payload, without calling the model or re-executing tools.

        The persisted final environment is restored and the pre-run environment is rebuilt from the sample metadata.
        Samples whose payload is invalid, belongs to another loaded row or was built with a different attack or
        victim model name, whose reply was blocked by a content filter, whose pre-run environment cannot be rebuilt
        or whose upstream check raised are returned as ``EXCLUDED`` with an empty value, so they drop out of the
        metric denominator. ``task_state.metadata`` is only read, so the predictions cache stays unchanged.
        """
        metadata = task_state.metadata or {}
        sample_id = task_state.sample_id
        try:
            payload = ScoringPayload.from_metadata(metadata, self._upstream)
            self._check_config_drift(payload.meta, sample_id)
        except ScoringPayloadError as exc:
            logger.error(f'AgentDojo sample {sample_id} excluded from scoring: {exc}')
            return self._excluded_score(
                original_prediction,
                filtered_prediction,
                self._partial_score_metadata(metadata, f'invalid_scoring_payload: {list(exc.fields)}'),
                explanation=str(exc),
            )

        meta, run = payload.meta, payload.run
        base = {
            'suite': meta.suite,
            'user_task_id': meta.user_task_id,
            'injection_task_id': meta.injection_task_id,
            'attack': meta.attack,
            'tool_call_trace_length': len(run.tool_call_trace),
            'termination_reason': run.termination_reason,
            'executed_steps': run.executed_steps,
            'empty_output': run.empty_output,
        }

        if run.termination_reason == TerminationReason.CONTEXT_LENGTH_EXCEEDED:
            # Upstream rule: no check is run; the user task failed and the attack counts as successful.
            return self._metric_score(
                original_prediction,
                filtered_prediction,
                utility=False,
                targeted_asr=None if self._benign else True,
                metadata=ScoreMetadata(**base),
            )

        if run.termination_reason == TerminationReason.CONTENT_FILTER:
            # The final answer is the provider's refusal text, not an agent decision.
            reason = 'content_filter: the provider blocked the model reply'
            return self._excluded_score(
                original_prediction,
                filtered_prediction,
                ScoreMetadata(**base, excluded_reason=reason).model_dump(mode='json'),
                explanation=reason,
            )

        try:
            pre_environment = create_environment(self._upstream, meta)
            utility, targeted_asr = check_results(
                self._upstream, meta, run.final_output, pre_environment, payload.post_environment, run.tool_call_trace
            )
        except Exception as exc:
            reason = f'check_error: {type(exc).__name__}: {exc}'
            logger.error(f'AgentDojo sample {sample_id} ({meta.row_id}) excluded from scoring: {reason}')
            return self._excluded_score(
                original_prediction,
                filtered_prediction,
                ScoreMetadata(**base, excluded_reason=reason).model_dump(mode='json'),
                explanation=reason,
            )

        return self._metric_score(
            original_prediction,
            filtered_prediction,
            utility=utility,
            targeted_asr=None if self._benign else targeted_asr,
            metadata=ScoreMetadata(**base),
        )

    def _check_config_drift(self, meta: AgentDojoSampleMeta, sample_id: int) -> None:
        """Reject a cached sample that belongs to another row or whose injections were built differently.

        The row check compares the persisted ``row_id`` with the row loaded at the same index; it is skipped for
        samples this adapter did not load.
        """
        loaded_row_id = self._loaded_row_ids.get((meta.suite, sample_id))
        if loaded_row_id is not None and loaded_row_id != meta.row_id:
            raise ScoringPayloadError(
                [f'{SAMPLE_META_KEY}.row_id'],
                f'Persisted row {meta.row_id!r} differs from the row {loaded_row_id!r} loaded at index {sample_id}; '
                'the task filters, limit or shuffle changed since the cached run.',
            )
        current = {
            'attack': self._attack,
            'victim_model_name': self._victim_model_name,
            'effective_model_name': self._planner.effective_model_name,
        }
        drifted = [name for name, value in current.items() if getattr(meta, name) != value]
        if drifted:
            persisted = {name: getattr(meta, name) for name in drifted}
            configured = {name: current[name] for name in drifted}
            raise ScoringPayloadError(
                [f'{SAMPLE_META_KEY}.{name}' for name in drifted],
                f'Persisted {persisted} differs from the current configuration {configured}.',
            )

    def _metric_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        utility: bool,
        targeted_asr: Optional[bool],
        metadata: ScoreMetadata,
    ) -> Score:
        """Build a usable score; benign runs emit only ``utility``."""
        value = {'utility': float(utility)}
        if not self._benign:
            value['targeted_asr'] = float(bool(targeted_asr))
        return Score(
            value=value,
            main_score_name='utility' if self._benign else 'targeted_asr',
            prediction=original_prediction,
            extracted_prediction=filtered_prediction,
            metadata=metadata.model_dump(mode='json'),
        )

    @staticmethod
    def _excluded_score(
        original_prediction: str, filtered_prediction: str, metadata: Dict[str, Any], explanation: str
    ) -> Score:
        """Build an ``EXCLUDED`` score whose empty value keeps the sample out of every metric."""
        return Score(
            value={},
            status=ScoreStatus.EXCLUDED,
            prediction=original_prediction,
            extracted_prediction=filtered_prediction,
            explanation=explanation,
            metadata=metadata,
        )

    @staticmethod
    def _partial_score_metadata(metadata: Mapping[str, Any], reason: str) -> Dict[str, Any]:
        """Return the ``ScoreMetadata`` fields that can be recovered from an invalid payload."""
        raw_meta = metadata.get(SAMPLE_META_KEY)
        raw_meta = raw_meta if isinstance(raw_meta, Mapping) else {}
        raw_run = metadata.get(RUN_RECORD_META_KEY)
        raw_run = raw_run if isinstance(raw_run, Mapping) else {}
        trace = raw_run.get('tool_call_trace')
        return {
            'suite': raw_meta.get('suite'),
            'user_task_id': raw_meta.get('user_task_id'),
            'injection_task_id': raw_meta.get('injection_task_id'),
            'attack': raw_meta.get('attack'),
            'tool_call_trace_length': len(trace) if isinstance(trace, list) else 0,
            'termination_reason': raw_run.get('termination_reason'),
            'excluded_reason': reason,
        }

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    @property
    def primary_metric(self) -> MetricSelector:
        """Select ``utility`` for benign runs and ``targeted_asr`` for attack runs.

        The selector matches by name only, so it also hits identities carrying the ``condition`` dimension.
        """
        return MetricSelector(name='utility' if self._benign else 'targeted_asr')

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        """Mean-aggregate each suite and label it with the run condition and attack metadata.

        ``EXCLUDED`` scores carry an empty value, so ``mean`` leaves them out of every metric and ``num`` counts only
        the samples actually scored. A suite whose samples are all excluded yields no aggregate, so the report never
        shows a 0-valued metric for it.
        """
        condition = 'benign' if self._benign else 'under_attack'
        run_metadata = {
            'attack': self._attack,
            'victim_model_name': self._victim_model_name,
            'benchmark_version': BENCHMARK_VERSION,
            'upstream_version': UPSTREAM_VERSION,
        }
        agg_scores = super().aggregate_scores(sample_scores)
        for agg_score in agg_scores:
            agg_score.dimensions = {**agg_score.dimensions, 'condition': condition}
            agg_score.metadata = {**(agg_score.metadata or {}), **run_metadata}
        return agg_scores
