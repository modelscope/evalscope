from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from evalscope.agent.environments.local import TemporaryLocalAgentEnvironment
from evalscope.agent.tools.bash import BASH_TOOL_INFO, run_bash
from evalscope.api.agent import AgentEnvironment, AgentLoopResult
from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.benchmark.adapters import AgentLoopAdapter
from evalscope.api.dataset import DatasetDict, Sample, load_local_file_dataset, resolve_snapshot_or_local_path
from evalscope.api.evaluator import TaskState
from evalscope.api.judge import (
    CaseVerdict,
    JudgeCase,
    JudgeContext,
    JudgeDefinition,
    JudgeRequest,
    OutputContract,
    ReducedVerdict,
)
from evalscope.api.messages import ChatMessageSystem, ChatMessageUser
from evalscope.api.metric import AggScore, SampleScore, Score
from evalscope.api.metric.semantics import MetricSelector
from evalscope.api.registry import register_benchmark
from evalscope.api.sandbox import merge_sandbox_config_dicts
from evalscope.constants import ScoringPolicy, Tags
from evalscope.utils.import_utils import check_import

from .utils import (
    EVAL_COLUMN_PROMPT,
    METRIC_NAMES,
    PRIMARY_KEY_PREPROCESS_PROMPT,
    WideSearchSession,
    aggregate_official_scores,
)

DATASET_ID = 'bytedance-community/WideSearch'

SYSTEM_PROMPTS = {
    'en': """# Role
You are an expert in online search. You task is gathering relevant information using advanced online search tools based on the user's query, and providing accurate answers according to the search results.

# Task Description
Upon receiving the user's query, you must thoroughly analyze and understand the user's requirements. In order to effectively address the user's query, you should make the best use of the provided tools to acquire comprehensive and reliable information and data. Below are the principles you should adhere to while performing this task:

- Fully understand the user's needs: Analyze the user's query, if necessary, break it down into smaller components to ensure a clear understanding of the user's primary intent.
- Flexibly use tools: After fully comprehending the user's needs, employ the provided tools to retrieve the necessary information.If the information retrieved previously is deemed incomplete or inaccurate and insufficient to answer the user's query, reassess what additional information is required and invoke the tool again until all necessary data is obtained.""",  # noqa: E501
    'zh': """# 角色设定
你是一位联网信息搜索专家，你需要根据用户的问题，通过联网搜索来搜集相关信息，然后根据这些信息来回答用户的问题。

# 任务描述
当你接收到用户的问题后，你需要充分理解用户的需求，利用我提供给你的工具，获取相对应的信息、资料，以解答用户的问题。
以下是你在执行任务过程中需要遵循的原则：
- 充分理解用户需求：你需要全面分析和理解用户的问题，必要时对用户的问题进行拆解，以确保领会到用户问题的主要意图。
- 灵活使用工具：当你充分理解用户需求后，请你使用我提供的工具获取信息；当你认为上次工具获取到的信息不全或者有误，以至于不足以回答用户问题时，请思考还需要搜索什么信息，再次调用工具获取信息，直至信息完备。""",
}

DESCRIPTION = """
## Overview

WideSearch evaluates search agents on broad web information-seeking tasks. Each task asks the agent to collect many
atomic facts and return one structured Markdown table. EvalScope uses the ModelScope
`bytedance-community/WideSearch` dataset.

## Task Description

- **Task Type**: Multi-turn search agent
- **Input**: Natural-language collection request with an explicit table schema
- **Output**: Complete Markdown table
- **Dataset**: 200 tasks in the ``full`` split; 100 English and 100 Chinese

## Key Features

- Official single-agent protocol: language-specific system prompt, ``function_calling``, and 50 default steps.
- Bash is available by default in a per-sample temporary local directory; Docker sandbox and MCP servers are optional.
- A single full run derives ``all``, ``en``, and ``zh`` reports without repeated inference.

## Evaluation Notes

- Uses the official Markdown table alignment and hybrid rule/LLM scoring semantics.
- Requires ``judge.strategy='auto'`` or ``'llm'`` with at least one ``judge.models`` entry; rule-only scoring is unsupported.
- See the [WideSearch usage guide](https://evalscope.readthedocs.io/en/latest/third_party/wide_search.html) for runtime
  examples and paper-style repeat settings.
"""


@register_benchmark(
    BenchmarkMeta(
        name='wide_search',
        pretty_name='WideSearch',
        tags=[Tags.AGENT, Tags.MULTI_TURN, Tags.RETRIEVAL],
        description=DESCRIPTION,
        dataset_id=DATASET_ID,
        subset_list=['default'],
        default_subset='default',
        eval_split='full',
        prompt_template='{question}',
        metric_list=list(METRIC_NAMES),
        primary_metric=MetricSelector(name='success_rate', aggregation='pass_at_k', dimensions={'scope': 'all'}),
        paper_url='https://arxiv.org/abs/2508.07999',
    )
)
class WideSearchAdapter(AgentLoopAdapter):
    """Official single-agent WideSearch benchmark adapter."""
    scoring_policy = ScoringPolicy.JUDGE_ONLY

    strategy_name = 'function_calling'
    max_steps_default = 50
    command_timeout_default = 120.0
    docker_image_default = 'python:3.11-slim'

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        check_import('dateparser', extra='wide_search', raise_error=True, feature_name='WideSearch evaluation')
        self._dataset_root: Optional[Path] = None

    def load(self) -> Tuple[DatasetDict, None]:
        # NOTE: download the full snapshot rather than an ``allow_file_pattern`` list.
        # ModelScope drops root-level file matches (e.g. ``widesearch.jsonl``) whenever the
        # pattern list also contains a subdirectory glob (e.g. ``widesearch_gold/*.csv``),
        # which left the data file missing. The dataset is small (one jsonl + gold CSVs).
        dataset_root = Path(resolve_snapshot_or_local_path(self))
        self._dataset_root = dataset_root
        data_path = dataset_root / 'widesearch.jsonl'
        if not data_path.exists():
            raise FileNotFoundError(f'WideSearch data file not found: {data_path}')
        dataset = load_local_file_dataset(
            adapter=self,
            dataset_path=str(data_path),
            subset='default',
            split=self.eval_split,
            sample_fields=self.record_to_sample,
            limit=self.limit,
            repeats=self.repeats,
            shuffle=self.shuffle,
        )
        return DatasetDict({'default': dataset}), None

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        if self._dataset_root is None:
            raise RuntimeError('WideSearch dataset root is not initialized.')
        instance_id = str(record['instance_id'])
        gold_path = self._dataset_root / 'widesearch_gold' / f'{instance_id}.csv'
        if not gold_path.exists():
            raise FileNotFoundError(f'WideSearch gold file not found: {gold_path}')
        evaluation = record['evaluation']
        if isinstance(evaluation, str):
            evaluation = json.loads(evaluation)
        return Sample(
            input=str(record['query']),
            target=gold_path.read_text(encoding='utf-8-sig'),
            tools=[BASH_TOOL_INFO],
            metadata={
                'instance_id': instance_id,
                'language': str(record['language']),
                'evaluation': evaluation,
            },
        )

    def build_tools(self, sample: Sample) -> Dict[str, Any]:
        return {'bash': run_bash}

    def build_environment(self, sample: Sample) -> Optional[AgentEnvironment]:
        sample_id = sample.metadata.get('instance_id') or sample.id or 'unknown'
        sandbox = self._task_config.sandbox if self._task_config is not None else None
        if sandbox is None or not sandbox.enabled:
            return TemporaryLocalAgentEnvironment(sample_id=sample_id, prefix='evalscope-wide-search-')
        check_import('ms_enclave', extra='sandbox', raise_error=True, feature_name='WideSearch Docker environment')
        from evalscope.agent.environments.enclave import EnclaveAgentEnvironment
        sandbox_config = merge_sandbox_config_dicts(
            {
                'image': self.docker_image_default,
                'network_enabled': True,
            },
            self._task_sandbox_config(),
        )
        return EnclaveAgentEnvironment(
            engine='docker',
            sandbox_config=sandbox_config,
        )

    def build_initial_messages(self, sample: Sample) -> List[Any]:
        messages = super().build_initial_messages(sample)
        language = str(sample.metadata.get('language', 'en'))
        return [ChatMessageSystem(content=SYSTEM_PROMPTS.get(language, SYSTEM_PROMPTS['en']))] + messages

    def build_max_steps_finalization_message(self, sample: Sample) -> str:
        return (
            '[Max Step] The tool has been used too many times. Please stop invoking the tool immediately and answer the user\'s question.'
        )

    def should_finalize_after_max_steps(self, result: AgentLoopResult) -> bool:
        return True

    def judge_definition(self, context: JudgeContext) -> JudgeDefinition:
        return JudgeDefinition.workflow(
            cases=self._build_cases(context),
            request=self._build_request,
            reduce=self._reduce_verdicts,
            main_score_name='success_rate',
            expand=self._expand_cases
        )

    def _build_cases(self, context: JudgeContext) -> List[JudgeCase]:
        session = self._session(context)
        if not session.needs_column_alignment():
            return []
        return [JudgeCase(case_id='column_alignment', output_contract=OutputContract(schema_model=_MappingVerdict))]

    def _build_request(
        self,
        case: JudgeCase,
        placement: Any,
        completed_cases: Sequence[CaseVerdict],
        context: JudgeContext,
    ) -> JudgeRequest:
        session = self._session(context)
        if case.metadata.get('kind') == 'column_score':
            response = case.metadata['response']
            prompt = EVAL_COLUMN_PROMPT.format(criterion=case.metadata.get('criterion'), response=response)
        else:
            prompt = PRIMARY_KEY_PREPROCESS_PROMPT.format(
                response=case.metadata.get(
                    'response',
                    session.response_df.columns.tolist() if session.response_df is not None else []
                ),
                reference=case.metadata.get('reference', session.required_columns),
            )
        prompt += case.output_contract.instruction()
        return JudgeRequest(messages=[ChatMessageUser(content=prompt)])

    def _expand_cases(
        self,
        stage: int,
        completed_cases: Sequence[CaseVerdict],
        context: JudgeContext,
    ) -> List[JudgeCase]:
        session = self._session(context)
        column_map = self._mapping(completed_cases, 'column_alignment')
        if stage == 1:
            answer_df, response_df, _ = session.frames(column_map)
            if answer_df is None or response_df is None:
                return []
            cases = []
            for column in session.unique_columns:
                pipeline = session.evaluation['eval_pipeline'].get(column, {})
                if {'llm_judge', 'exact_match'} & set(pipeline.get('metric', [])):
                    cases.append(
                        JudgeCase(
                            case_id=f'primary_key:{column}',
                            output_contract=OutputContract(schema_model=_MappingVerdict),
                            metadata={
                                'response': response_df[column].tolist(),
                                'reference': answer_df[column].tolist(),
                            },
                        )
                    )
            return cases
        if stage != 2:
            return []
        primary_key_maps = {
            case.case_id.removeprefix('primary_key:'): self._mapping_value(case)
            for case in completed_cases
            if case.case_id.startswith('primary_key:')
        }
        inner_df, _ = session.inner_frame(column_map, primary_key_maps)
        if inner_df is None or inner_df.empty:
            return []
        cases = []
        for column in session.required_columns:
            if column in session.unique_columns:
                continue
            pipeline = session.evaluation['eval_pipeline'][column]
            if 'llm_judge' not in pipeline.get('metric', []):
                continue
            response = {
                f'idx_{index}': {
                    'response': response_value,
                    'target': target_value,
                }
                for index, (response_value, target_value
                            ) in enumerate(zip(inner_df[f'{column}_response'], inner_df[f'{column}_query']))
            }
            cases.append(
                JudgeCase(
                    case_id=f'column_score:{column}',
                    output_contract=OutputContract(schema_model=_column_score_model(len(inner_df))),
                    metadata={
                        'kind': 'column_score',
                        'criterion': pipeline.get('criterion'),
                        'response': response,
                    },
                )
            )
        return cases

    def _reduce_verdicts(
        self,
        case_verdicts: Sequence[CaseVerdict],
        context: JudgeContext,
    ) -> ReducedVerdict:
        session = self._session(context)
        column_map = self._mapping(case_verdicts, 'column_alignment')
        primary_key_maps = {
            case.case_id.removeprefix('primary_key:'): self._mapping_value(case)
            for case in case_verdicts
            if case.case_id.startswith('primary_key:')
        }
        column_scores = {
            f'{case.case_id.removeprefix("column_score:")}_llm_judge': self._score_values(case)
            for case in case_verdicts
            if case.case_id.startswith('column_score:')
        }
        values, diagnostics = session.score(column_scores, column_map, primary_key_maps)
        return ReducedVerdict(value=values, metadata=diagnostics)

    @staticmethod
    def _mapping(cases: Sequence[CaseVerdict], case_id: str) -> Dict[str, str]:
        for case in cases:
            if case.case_id == case_id:
                return WideSearchAdapter._mapping_value(case)
        return {}

    @staticmethod
    def _mapping_value(case: CaseVerdict) -> Dict[str, str]:
        return {str(key): str(value) for key, value in case.value.mapping.items()}

    @staticmethod
    def _score_values(case: CaseVerdict) -> List[float]:
        values = case.value.model_dump()
        return [float(values[f'idx_{index}']) for index in range(len(values))]

    @staticmethod
    def _session(context: JudgeContext) -> WideSearchSession:
        return WideSearchSession.create(
            prediction=context.filtered_prediction,
            gold_csv=context.reference,
            evaluation=context.task_state.metadata['evaluation'],
        )

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        return aggregate_official_scores(sample_scores)


class _MappingVerdict(BaseModel):
    """A semantic alignment returned through the shared JSON judge contract."""

    mapping: Dict[str, str]


def _column_score_model(size: int) -> type[BaseModel]:
    """Build the exact per-row 0/1 output shape for one WideSearch column case."""
    from pydantic import create_model

    return create_model(
        f'WideSearchColumnScore{size}',
        **{f'idx_{index}': (float, Field(ge=0.0, le=1.0))
           for index in range(size)},
    )
