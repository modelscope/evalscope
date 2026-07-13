from __future__ import annotations

import json
import threading
from modelscope import dataset_snapshot_download
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from evalscope.agent.environments.local import TemporaryLocalAgentEnvironment
from evalscope.agent.tools.bash import BASH_TOOL_INFO, run_bash
from evalscope.api.agent import AgentEnvironment, AgentLoopResult
from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.benchmark.adapters import AgentLoopAdapter
from evalscope.api.dataset import DatasetDict, LocalDataLoader, Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageSystem
from evalscope.api.metric import AggScore, SampleScore, Score
from evalscope.api.registry import register_benchmark
from evalscope.api.sandbox import merge_sandbox_config_dicts
from evalscope.constants import JudgeStrategy, Tags
from evalscope.utils.import_utils import check_import
from .utils import METRIC_NAMES, WideSearchScorer, aggregate_official_scores

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
- Requires ``judge_strategy='auto'`` or ``'llm'`` with explicit ``judge_model_args``; rule-only scoring is unsupported.
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
        paper_url='https://arxiv.org/abs/2508.07999',
    )
)
class WideSearchAdapter(AgentLoopAdapter):
    """Official single-agent WideSearch benchmark adapter."""

    strategy_name = 'function_calling'
    max_steps_default = 50
    command_timeout_default = 120.0
    docker_image_default = 'python:3.11-slim'

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        check_import('dateparser', extra='wide_search', raise_error=True, feature_name='WideSearch evaluation')
        self._use_llm_judge = True
        self._dataset_root: Optional[Path] = None
        self._judge_lock = threading.Lock()

    def load(self) -> Tuple[DatasetDict, None]:
        dataset_name_or_path = self.dataset_id
        if Path(dataset_name_or_path).exists():
            dataset_root = Path(dataset_name_or_path).expanduser().resolve()
        else:
            dataset_root = Path(
                dataset_snapshot_download(
                    dataset_name_or_path,
                    allow_file_pattern=['widesearch.jsonl', 'widesearch_gold/*.csv'],
                )
            )
        self._dataset_root = dataset_root
        data_path = dataset_root / 'widesearch.jsonl'
        if not data_path.exists():
            raise FileNotFoundError(f'WideSearch data file not found: {data_path}')
        dataset = LocalDataLoader(
            data_id_or_path=str(data_path),
            split=self.eval_split,
            subset='default',
            sample_fields=self.record_to_sample,
            limit=self.limit,
            repeats=self.repeats,
            shuffle=self.shuffle,
        ).load()
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
            timeout=self._resolve_command_timeout(self._task_config.agent_config if self._task_config else None),
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

    def llm_match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        self._validate_judge_config()
        with self._judge_lock:
            judge = self.llm_judge
        scorer = WideSearchScorer(judge=judge.judge)
        result = scorer.evaluate(
            prediction=filtered_prediction,
            gold_csv=reference,
            evaluation=task_state.metadata['evaluation'],
        )
        return Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
            value=result.values,
            explanation=result.diagnostics.get('error') or 'Official WideSearch table evaluation completed.',
            metadata=result.diagnostics,
            main_score_name='success_rate',
        )

    def match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        raise ValueError("WideSearch requires judge_strategy='auto' or 'llm' with judge_model_args.")

    def _validate_judge_config(self) -> None:
        if self.judge_strategy not in {JudgeStrategy.AUTO, JudgeStrategy.LLM}:
            raise ValueError("WideSearch requires judge_strategy='auto' or 'llm'.")
        if not self._task_config.judge_model_args:
            raise ValueError('WideSearch requires judge_model_args for official table alignment and scoring.')

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        return aggregate_official_scores(sample_scores)
