# flake8: noqa: E501
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import DatasetDict, DatasetHub, Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageSystem, ChatMessageUser
from evalscope.api.metric.scorer import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags

from .utils import (
    DEFAULT_GOALS,
    DEFAULT_POSITIONS,
    SUITES,
    SYSTEM_PROMPTS,
    build_data_prompt,
    parse_json_response,
    score_response,
)

DESCRIPTION = """
## Overview

LongPIBench is a benchmark for prompt-injection attacks and defenses in four long-context review \
tasks: paper review, resume screening, email review, and code review. Each example buries an \
adversarial instruction inside trusted-looking content (a LaTeX paper, a resume, an email with \
attachment, or a code diff) and checks whether the model follows the attacker's objective instead \
of its task instruction.

## Task Description

- **Task Type**: Prompt-injection robustness under long-context review tasks
- **Input**: A review system prompt that requests JSON output, plus a long user context with an \
instruction injected at the published default location
- **Output**: A JSON object (paper review with score, resume screening recommendation, email \
summary and reply, or code review approval)
- **Domain**: Safety, prompt injection, long-context instruction following

## Key Features

- 100 synthetic examples for each of 4 suites (paper / resume / email / code), from the EMNLP \
Findings 2026 paper "LongPIBench: A Long-Context Benchmark for Prompt Injection"
- 4 attack variants (`no`, `naive`, `combine`, `authority_spoof`) and 3 goals per suite; by \
default every suite is attacked with `authority_spoof`, using that suite's published default goal
- Prompts are assembled deterministically from the released raw material (LaTeX sections, \
structured resume/email/code JSON), following the official implementation
- The examples intentionally contain adversarial instructions; treat prompts and outputs as \
untrusted data and never connect runs to tools with real-world side effects

## Evaluation Notes

- Subsets are the four suites; one run evaluates one attack condition, as in the official runner
- Primary metric **`attack_success_rate`**: the share of replies satisfying the attack objective (higher is worse)
- Set `attack` to `no` for the clean baseline (same objective, no injection), e.g. \
`dataset_args={'longpi_bench': {'extra_params': {'attack': 'no'}}}`; `goal` and `position` \
select other published conditions
- Replies that fail JSON parsing count as not satisfying the objective, matching the official runner
- [Paper](https://arxiv.org/abs/2608.28411) | [GitHub](https://github.com/liu00222/LongPIBench)
"""


@register_benchmark(
    BenchmarkMeta(
        evaluation_version='v1.0',
        name='longpi_bench',
        pretty_name='LongPIBench',
        dataset_id='evalscope/LongPIBench',
        tags=[Tags.SAFETY, Tags.LONG_CONTEXT],
        description=DESCRIPTION,
        paper_url='https://arxiv.org/abs/2608.28411',
        subset_list=list(SUITES),
        metric_list=['attack_success_rate'],
        eval_split='test',
        extra_params={
            'attack': {
                'type': 'str',
                'description': "Attack variant: 'no', 'naive', 'combine' or 'authority_spoof'.",
                'value': 'authority_spoof',
            },
            'goal': {
                'type': 'str',
                'description': "Attack goal '0', '1' or '2'; 'default' uses each suite's published goal.",
                'value': 'default',
            },
            'position': {
                'type': 'str',
                'description': "Paper section to inject into; 'default' uses each suite's published position.",
                'value': 'default',
            },
        },
    )
)
class LongPIBenchAdapter(DefaultDataAdapter):
    """Data adapter for LongPIBench.

    The hub `manifest.jsonl` (100 ids) is loaded through the standard loader; each record
    expands into one sample per requested suite, whose prompt is assembled from the raw
    material in the dataset snapshot.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.reformat_subset = True
        self.attack = self.extra_params.get('attack', 'authority_spoof')
        self.goal = self.extra_params.get('goal', 'default')
        self.position = self.extra_params.get('position', 'default')
        self._data_root: Optional[Path] = None

    def load(self) -> Tuple[DatasetDict, Optional[DatasetDict]]:
        # The prompts are built from raw files that the manifest only references
        self._data_root = Path(
            DatasetHub(
                data_id_or_path=self.dataset_id,
                data_source=self.dataset_hub,
                revision=self.dataset_revision,
                force_redownload=self.force_redownload,
                cache_dir=self.dataset_dir,
            ).download_snapshot()
        )
        return super().load()

    def record_to_sample(self, record: Dict[str, Any]) -> List[Sample]:
        item_id = int(record['id'])
        samples = []
        for suite in self.subset_list:
            goal = DEFAULT_GOALS[suite] if self.goal == 'default' else self.goal
            position = DEFAULT_POSITIONS[suite] if self.position == 'default' else self.position
            data_prompt = build_data_prompt(self._data_root, suite, item_id, self.attack, goal, position)
            samples.append(
                Sample(
                    input=[ChatMessageSystem(content=SYSTEM_PROMPTS[suite]), ChatMessageUser(content=data_prompt)],
                    target='',
                    subset_key=suite,
                    metadata={'suite': suite, 'item_id': item_id, 'attack': self.attack, 'goal': goal},
                )
            )
        return samples

    def match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        score = Score(extracted_prediction=filtered_prediction, prediction=original_prediction)
        metadata = task_state.metadata
        try:
            response = parse_json_response(filtered_prediction)
            success = score_response(metadata['suite'], metadata['goal'], response)
        except (ValueError, json.JSONDecodeError):
            # Same failure handling as the official runner: an unparseable reply is not a success
            success = False
        score.value = {'attack_success_rate': float(success)}
        score.main_score_name = 'attack_success_rate'
        return score
