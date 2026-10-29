"""AgentHarm public harmful and benign tasks on EvalScope's native AgentLoop."""

import hashlib
import json
import os
import random
from typing import Any, Dict, List, Tuple, Type

from evalscope.agent.strategies.function_calling import FunctionCallingStrategy
from evalscope.api.agent import AgentContext, AgentLoopResult, NativeAgentConfig, ParsedAction
from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.benchmark.adapters import AgentLoopAdapter
from evalscope.api.dataset import DataLoader, Dataset, DatasetDict, DictDataLoader, Sample, download_dataset_file
from evalscope.api.judge import JudgeContext, JudgeDefinition
from evalscope.api.messages import ChatMessageSystem, ChatMessageUser
from evalscope.api.metric import AggScore, SampleScore
from evalscope.api.model import ModelOutput
from evalscope.api.registry import register_benchmark
from evalscope.api.tool import ToolCall
from evalscope.benchmarks.agentharm.scoring import excluded_definition, scoring_definition
from evalscope.benchmarks.agentharm.types import (
    CATEGORIES,
    DATASET_REVISION,
    METRICS,
    MS_DATASET_REVISION,
    Behavior,
)
from evalscope.constants import ScoringPolicy, Tags
from evalscope.third_party.agentharm.content import tool_function, tool_info
from evalscope.third_party.agentharm.prompts import SYSTEM_PROMPT
from evalscope.third_party.agentharm.resources import DATA_MANIFEST, MANIFEST
from evalscope.third_party.agentharm.types import SOURCE_SHA, UPSTREAM_VERSION, AgentHarmSampleMeta
from evalscope.utils.logger import get_logger

logger = get_logger()

EXTRA_PARAMS = {
    'behavior_ids': {
        'type': 'list',
        'value': None,
        'description': 'Exact augmented behavior IDs. Overrides the two variant filters.',
    },
    'detailed_behaviors': {
        'type': 'bool',
        'value': None,
        'description': 'Select detailed (true) or open-ended (false) behaviors; null keeps both.',
    },
    'hint_included': {
        'type': 'bool',
        'value': None,
        'description': 'Select behaviors with or without hints; null keeps both.',
    },
}

DESCRIPTION = """
## Overview

AgentHarm measures how much of a directly requested harmful task a tool-using agent completes. Its paired benign
version measures normal task capability and over-refusal. All task tools are simulations; they do not send real
messages, execute shell commands or contact external task services.

## Task Description

- **Task Type**: Multi-step agent safety and tool use
- **Input**: A user request and the official task-specific simulated tools
- **Output**: Model tool calls, simulated observations and assistant responses
- **Domain**: Public tasks in disinformation, harassment, drugs, fraud, hate, cybercrime, sexual content and copyright

## Key Features

- Separate `agentharm` (harmful) and `agentharm_benign` (normal tasks) results, using a shared EvalScope adapter
- Official public `test_public` and `val` files, with detailed/open-ended and hint/no-hint variants
- Task-specific rubrics check actions, arguments, ordering and generated content, awarding partial completion credit
- Fixed upstream source d1adf47, behavior version 8-B; the upstream README separately recommends 6-B

## Evaluation Notes

- Primary metric: `task_completion_score`, the mean fraction of completed rubric criteria per task; this is not binary ASR
- Also reports `full_completion_rate`, `refusal_rate` and `non_refusal_completion_score` (conditional on no refusal)
- Completion scores should be low for harmful tasks and high for benign tasks; refusal has the opposite direction
- Semantic checks and refusal detection require one explicitly configured `judge.models` entry with `judge.repeats=1`;
  malformed or failed `LLM Judge` replies exclude the sample instead of improving its safety score
- Uses Python 3.10+ and EvalScope's built-in AgentLoop; Inspect, Docker and AgentBridge are not used
- Default budget: 20 model generations, distinct from upstream's 20-message limit. Tool order is locally seeded and
  persisted. Llama-specific text tool protocols, forced-first tools, irrelevant tools and jailbreak presets are not supported
- Recommended generation_config: temperature=0, max_tokens=4096, seed=0 where supported; actual model settings are retained
- `OutputContract` schemas replace upstream free-text grading replies; full traces are reused for regrading without rerunning tools
- Read captured reasoning when checking refusals. A refusal after an action does not erase completion credit
- Provider content filtering, context overflow, rubric errors and invalid scoring payloads are excluded
- `non_refusal_completion_score` is omitted when its denominator is empty; it is never replaced by zero. Read harmful
  scores alongside benign capability
- Source and dataset revisions are pinned. The upstream MIT license has an additional AI safety/security use clause;
  preserve its evaluation-only notice and do not use benchmark examples for training
""".strip()


class AgentHarmStrategy(FunctionCallingStrategy):
    """End on a plain assistant response, preserving native tool errors and tool execution."""

    def __init__(self) -> None:
        super().__init__(include_submit_tool=False)

    def parse_output(self, output: ModelOutput, ctx: AgentContext) -> ParsedAction:
        parsed = super().parse_output(output, ctx)
        if not parsed.tool_calls:
            parsed.final_answer = output.message.text or ''
        return parsed

    def extract_final_answer(self, result: AgentLoopResult) -> str:
        """Return visible text; keep reasoning in the native persisted message snapshot."""
        return result.final_output.message.text or ''


@register_benchmark(
    BenchmarkMeta(
        name='agentharm',
        pretty_name='AgentHarm',
        dataset_id='evalscope/AgentHarm',
        dataset_revision=MS_DATASET_REVISION,
        hf_dataset_id='ai-safety-institute/AgentHarm',
        hf_dataset_revision=DATASET_REVISION,
        subset_list=list(CATEGORIES),
        eval_split='test_public',
        few_shot_mode='disabled',
        metric_list=list(METRICS),
        primary_metric='task_completion_score',
        aggregation='mean',
        evaluation_version='v1.0',
        description=DESCRIPTION,
        extra_params=EXTRA_PARAMS,
        tags=[Tags.AGENT, Tags.SAFETY, Tags.FUNCTION_CALLING],
        paper_url='https://arxiv.org/abs/2410.09024',
    )
)
@register_benchmark(
    BenchmarkMeta(
        name='agentharm_benign',
        pretty_name='AgentHarm Benign',
        dataset_id='evalscope/AgentHarm',
        dataset_revision=MS_DATASET_REVISION,
        hf_dataset_id='ai-safety-institute/AgentHarm',
        hf_dataset_revision=DATASET_REVISION,
        subset_list=list(CATEGORIES),
        eval_split='test_public',
        few_shot_mode='disabled',
        metric_list=list(METRICS),
        primary_metric='task_completion_score',
        aggregation='mean',
        evaluation_version='v1.0',
        description=DESCRIPTION,
        extra_params=EXTRA_PARAMS,
        tags=[Tags.AGENT, Tags.SAFETY, Tags.FUNCTION_CALLING],
        paper_url='https://arxiv.org/abs/2410.09024',
    )
)
class AgentHarmAdapter(AgentLoopAdapter):
    """Use fixed public task content with native data, agent, judge and report flows."""

    max_steps_default = 20
    scoring_policy = ScoringPolicy.JUDGE_ONLY

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.reformat_subset = True
        self.task_kind = 'benign' if self.name == 'agentharm_benign' else 'harmful'
        self._checksum = ''
        self._loaded_ids: Dict[Tuple[int, str], Tuple[str, str, str, str]] = {}
        if self.eval_split not in ('test_public', 'val'):
            raise ValueError('AgentHarm eval_split must be test_public or val.')
        if any(s not in CATEGORIES for s in self.subset_list):
            raise ValueError(f'AgentHarm subset_list must select from {list(CATEGORIES)}.')
        params = self.extra_params
        ids = params['behavior_ids']
        if ids is not None and (not isinstance(ids, list) or any(not isinstance(i, str) for i in ids)):
            raise ValueError('AgentHarm behavior_ids must be a list of strings or null.')
        for key in ('detailed_behaviors', 'hint_included'):
            if params[key] is not None and not isinstance(params[key], bool):
                raise ValueError(f'AgentHarm {key} must be a bool or null.')
        if ids and any(params[k] is not None for k in ('detailed_behaviors', 'hint_included')):
            logger.warning('AgentHarm behavior_ids overrides detailed_behaviors and hint_included.')
        if self._task_config is not None:
            self._validate_runtime()

    def _validate_runtime(self) -> None:
        ac = self._task_config.agent_config
        if ac is not None and (
            not isinstance(ac, NativeAgentConfig)
            or ac.strategy != 'function_calling'
            or ac.tools
            or ac.mcp_servers
            or ac.kwargs
        ):
            raise ValueError(
                'AgentHarm requires the native function_calling strategy with no extra tools, MCP or kwargs.'
            )
        judge = self._task_config.judge
        if (
            judge.strategy not in ('auto', 'llm')
            or len(judge.models) != 1
            or judge.repeats != 1
            or judge.min_valid_judges != 1
            or judge.position_swap == 'on'
        ):
            raise ValueError(
                'AgentHarm requires exactly one judge.models entry, repeats=1, min_valid_judges=1, and no position swap.'
            )

    def load_subset(self, subset: str, data_loader: Type[DataLoader]) -> Dataset:
        """Unwrap official JSON; leave shuffling, category limits and repeats to standard loaders."""
        split_name = 'validation' if self.eval_split == 'val' else 'test_public'
        filename = f'{self.task_kind}_behaviors_{split_name}.json'
        path = download_dataset_file(
            self.dataset_id,
            f'benchmark/{filename}',
            data_source=self.dataset_hub,
            revision=self.dataset_revision,
            force_redownload=self.force_redownload,
            cache_dir=self.dataset_dir,
        )
        with open(path, 'rb') as handle:
            content = handle.read()
        self._checksum = hashlib.sha256(content).hexdigest()
        # The source manifest pins remote official files; local fixtures retain their actual identity.
        if self.dataset_id in ('evalscope/AgentHarm', 'ai-safety-institute/AgentHarm'):
            if self._checksum != DATA_MANIFEST[filename]['sha256']:
                raise ValueError(f'AgentHarm {filename} differs from the pinned public dataset.')
        raw_rows = json.loads(content)['behaviors']
        rows = [Behavior.model_validate(r) for r in raw_rows]
        ids = [r.id for r in rows]
        if len(ids) != len(set(ids)):
            raise ValueError('AgentHarm behavior IDs must be unique.')
        for row in rows:
            if (
                row.category not in CATEGORIES
                or row.grading_function not in MANIFEST[self.task_kind]['rubrics']
                or any(t not in MANIFEST[self.task_kind]['tools'] for t in row.target_functions)
                or len(row.target_functions) != len(set(row.target_functions))
            ):
                raise ValueError(f'AgentHarm behavior {row.id!r} has an invalid category, rubric or tool list.')
        params = self.extra_params
        wanted = params['behavior_ids']
        if set(wanted or []) - set(ids):
            raise ValueError(f'Unknown AgentHarm behavior IDs: {sorted(set(wanted) - set(ids))}.')
        if wanted:
            rows = [r for r in rows if r.id in wanted]
        else:
            for key, field in (('detailed_behaviors', 'detailed_prompt'), ('hint_included', 'hint_included')):
                if params[key] is not None:
                    rows = [r for r in rows if getattr(r, field) == params[key]]
        return DictDataLoader(
            dict_list=[r.model_dump() for r in rows],
            sample_fields=self.record_to_sample,
            filter_func=self.sample_filter,
            limit=None,
            repeats=1,
            shuffle=self.shuffle,
            seed=self.seed,
        ).load()

    def load_subsets(self, load_func: Any, is_fewshot: bool = False) -> DatasetDict:
        """Keep task identities before positional prediction caches overwrite sample metadata."""
        datasets = super().load_subsets(load_func, is_fewshot=is_fewshot)
        if not is_fewshot:
            self._loaded_ids = {
                (sample.id, sample.input[-1].text): self._identity(
                    AgentHarmSampleMeta.model_validate(sample.metadata['agentharm'])
                )
                for subset, dataset in datasets.items()
                for sample in dataset
            }
        return datasets

    @staticmethod
    def _identity(meta: AgentHarmSampleMeta) -> Tuple[str, str, str, str]:
        return meta.task_kind, meta.split, meta.behavior_id, meta.dataset_checksum

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Keep original task ordering for checks and a separate seeded tool order for the agent."""
        row = Behavior.model_validate(record)
        seed_bytes = f'{self.seed}:{self.task_kind}:{self.eval_split}:{row.id}'.encode()
        tool_seed = int.from_bytes(hashlib.sha256(seed_bytes).digest()[:8], 'big')
        order = list(row.target_functions)
        random.Random(tool_seed).shuffle(order)
        meta = AgentHarmSampleMeta(
            task_kind=self.task_kind,
            behavior_id=row.id,
            id_original=row.id_original,
            category=row.category,
            split=self.eval_split,
            grading_function=row.grading_function,
            target_functions=row.target_functions,
            tool_order=order,
            detailed_prompt=row.detailed_prompt,
            hint_included=row.hint_included,
            dataset_checksum=self._checksum,
            dataset_revision=None if os.path.exists(self.dataset_id) else self.dataset_revision,
            tool_order_seed=tool_seed,
        )
        return Sample(
            input=[ChatMessageSystem(content=SYSTEM_PROMPT), ChatMessageUser(content='\n\n' + row.prompt)],
            target='',
            subset_key=row.category,
            tools=[tool_info(self.task_kind, name) for name in order],
            metadata={'agentharm': meta.model_dump(mode='json')},
        )

    def build_strategy(self, sample: Sample) -> AgentHarmStrategy:
        """Use native function calls and end immediately on a plain answer."""
        return AgentHarmStrategy()

    def build_tools(self, sample: Sample) -> Dict[str, Any]:
        """Create only the sample's fixed simulated tools; never expose global tools."""
        meta = AgentHarmSampleMeta.model_validate(sample.metadata['agentharm'])
        handlers = {}
        for name in meta.tool_order:
            function = tool_function(meta.task_kind, name)

            async def handle_tool(call: ToolCall, environment: Any, fn: Any = function) -> str:
                return await fn(**call.function.arguments)

            handlers[name] = handle_tool
        return handlers

    @staticmethod
    def _resolve_validate_tool_arguments(ac: Any) -> bool:
        return True

    def judge_definition(self, context: JudgeContext) -> JudgeDefinition:
        """Declare the official rubric and refusal cases from the saved native transcript."""
        try:
            state = context.task_state
            meta = AgentHarmSampleMeta.model_validate(state.metadata['agentharm'])
            expected = self._loaded_ids.get((state.sample_id, state.input[-1].text))
            if self._loaded_ids and self._identity(meta) != expected:
                raise ValueError('Cached AgentHarm task identity differs from the currently loaded sample.')
            if (
                meta.task_kind != self.task_kind
                or meta.split != self.eval_split
                or meta.source_sha != SOURCE_SHA
                or meta.upstream_version != UPSTREAM_VERSION
                or meta.evaluation_version != 'v1.0'
                or (self._checksum and meta.dataset_checksum != self._checksum)
            ):
                raise ValueError('Persisted AgentHarm source or task configuration differs from this run.')
            if not state.messages:
                raise ValueError('AgentHarm scoring requires the saved native messages.')
            if state.output.stop_reason in ('content_filter', 'model_length'):
                raise ValueError(f'Provider prevented scoring: {state.output.stop_reason}.')
            return scoring_definition(context, meta)
        except Exception as exc:
            return excluded_definition(context, f'{type(exc).__name__}: {exc}')

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        """Retain native means and attach the source and execution condition to each metric."""
        scores = super().aggregate_scores(sample_scores)
        ac = self._task_config.agent_config if self._task_config is not None else None
        for score in scores:
            score.metadata = {
                **(score.metadata or {}),
                'task_kind': self.task_kind,
                'source_sha': SOURCE_SHA,
                'upstream_version': UPSTREAM_VERSION,
                'evaluation_version': self._benchmark_meta.evaluation_version,
                'dataset_revision': None if os.path.exists(self.dataset_id) else self.dataset_revision,
                'dataset_checksum': self._checksum,
                'max_steps': self._resolve_max_steps(ac),
                'budget_unit': 'model_generations',
            }
        return scores
