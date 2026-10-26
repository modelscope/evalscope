# flake8: noqa: E501
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import DatasetDict, DatasetHub, Sample, build_dataset_from_records
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageSystem, ChatMessageUser
from evalscope.api.metric.scorer import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

from .utils import (
    DEFAULT_GOALS,
    DEFAULT_POSITIONS,
    SUITES,
    SYSTEM_PROMPTS,
    build_data_prompt,
    load_item_ids,
    parse_json_response,
    score_response,
)

logger = get_logger()

SUBSET_LIST = [f'{suite}_{condition}' for suite in SUITES for condition in ('clean', 'attacked')]

# Subset name -> (suite, clean flag)
SUBSET_CONDITIONS: Dict[str, Tuple[str, bool]] = {
    f'{suite}_{condition}': (suite, condition == 'clean') for suite in SUITES for condition in ('clean', 'attacked')
}

DESCRIPTION = """
## Overview

LongPIBench is a benchmark for prompt-injection attacks and defenses in four long-context review \
tasks: paper review, resume screening, email review, and code review. Each example buries an \
adversarial instruction deep inside trusted-looking content (a LaTeX paper, a resume, an email with \
attachment, or a code diff) and checks whether the model follows the attacker's objective instead \
of its task instruction.

## Task Description

- **Task Type**: Long-context instruction-following with a fixed JSON output contract
- **Input**: A review system prompt (requesting JSON output) plus a long user context of up to \
several tens of thousands of tokens with an injected instruction in the published default location
- **Output**: A JSON object (paper review with score, resume screening recommendation, email \
summary + reply, or code review approval)
- **Grading**: Deterministic parsing of the JSON reply against the attack objective; no LLM judge

## Key Features

- 100 synthetic examples per suite × 4 suites (paper / resume / email / code), from the EMNLP \
Findings 2026 paper "LongPIBench: A Long-Context Benchmark for Prompt Injection"
- 4 attack variants (none / naive / combine / authority_spoof) × 3 goals per suite; the default \
attacked condition uses `authority_spoof` with the published default goal of each suite, and the \
clean condition runs the same objective without any injection
- The examples intentionally contain adversarial instructions; treat prompts and outputs as \
untrusted data and never connect runs to tools with real-world side effects

## Evaluation Notes

- Subsets: `<suite>_clean` (no injection; measures the model's baseline compliance with the \
objective) and `<suite>_attacked` (default: `authority_spoof` at the published default goal)
- Primary metric **`asr`** — the share of replies satisfying the attack objective (higher is \
worse); for `_clean` subsets it reads as the no-injection baseline for the same objective
- Replies that fail JSON parsing score 0 (objective not satisfied) rather than being re-parsed
- The attack/goal condition of the attacked subsets can be overridden via `dataset_args`, e.g. \
`dataset_args={'longpi_bench': {'attack': 'naive', 'goal': '1'}}`
- [Paper](https://arxiv.org/abs/2608.28411) | [GitHub](https://github.com/liu00222/LongPIBench)
"""


@register_benchmark(
    BenchmarkMeta(
        evaluation_version='v1.0',
        name='longpi_bench',
        pretty_name='LongPIBench',
        dataset_id='evalscope/LongPIBench',
        tags=[Tags.LONG_CONTEXT, Tags.REASONING, Tags.QA],
        description=DESCRIPTION,
        paper_url='https://arxiv.org/abs/2608.28411',
        subset_list=SUBSET_LIST,
        metric_list=['asr'],
        eval_split='test',
        extra_params={
            'attack': {
                'type': 'str',
                'description': 'Attack variant of the attacked subsets.',
                'value': 'authority_spoof',
            },
            'goal': {
                'type': 'str',
                'description': "Attack goal ('0'/'1'/'2' or 'default' per suite).",
                'value': 'default',
            },
            'position': {
                'type': 'str',
                'description': 'Injection position (paper sections or default).',
                'value': 'default',
            },
        },
    )
)
class LongPiBenchAdapter(DefaultDataAdapter):
    """Data adapter for evalscope/LongPIBench (prompt injection in long contexts).

    The hub repo stores raw material (LaTeX paper projects, person_info / emails /
    code_changes JSON); the model-ready prompt is assembled deterministically per
    the official implementation. Each subset pins one suite × condition.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.reformat_subset = True
        self.attack = str(self.extra_params.get('attack') or 'authority_spoof')
        self.goal = str(self.extra_params.get('goal') or 'default')
        self.position = str(self.extra_params.get('position') or 'default')

    def load(self) -> Tuple[DatasetDict, None]:
        """Download the dataset snapshot and build one subset per suite × condition."""
        hub = DatasetHub(
            data_id_or_path=self.dataset_id,
            data_source=self.dataset_hub,
            revision=self.dataset_revision,
            force_redownload=self.force_redownload,
            cache_dir=self.dataset_dir,
        )
        # Only the raw material of the released benchmark; ignore unrelated files
        root = Path(
            hub.download_snapshot(
                allow_file_pattern=[
                    'manifest.jsonl',
                    'papers/*/*',
                    'person_info/*.json',
                    'emails/*.json',
                    'code_changes/*.json',
                ],
            )
        )
        item_ids = load_item_ids(root)

        records: List[Dict[str, Any]] = []
        for subset, (suite, clean) in SUBSET_CONDITIONS.items():
            if subset not in self.subset_list:
                continue
            if clean:
                attack, goal = 'no', '0'
            else:
                attack = self.attack
                goal = DEFAULT_GOALS[suite] if self.goal == 'default' else self.goal
            position = DEFAULT_POSITIONS[suite] if self.position == 'default' else self.position
            for item_id in item_ids:
                records.append(
                    {
                        'subset': subset,
                        'suite': suite,
                        'item_id': item_id,
                        'attack': attack,
                        'goal': goal,
                        'position': position,
                    }
                )

        def optional_record_to_sample(record: Dict[str, Any]) -> List[Sample]:
            sample = self.record_to_sample(root, record)
            return [sample] if sample is not None else []

        mem_dataset = build_dataset_from_records(
            records=records,
            sample_fields=optional_record_to_sample,
            name=self.name,
            location=str(root),
            limit=None,
            repeats=1,
            shuffle=False,
            seed=None,
        )
        dataset_dict = DatasetDict.from_dataset(
            dataset=mem_dataset,
            subset_list=self.subset_list,
            limit=self.limit,
            repeats=self.repeats,
        )
        return dataset_dict, None

    def record_to_sample(self, root: Path, record: Dict[str, Any]) -> Optional[Sample]:
        """Assemble one model-ready example from the snapshot material."""
        suite = record['suite']
        try:
            data_prompt = build_data_prompt(
                root=root,
                suite=suite,
                item_id=record['item_id'],
                attack=record['attack'],
                goal=record['goal'],
                position=record['position'],
            )
        except (FileNotFoundError, KeyError, ValueError) as e:
            logger.warning(f'Failed to build example {suite}/{record["item_id"]}: {e}; skipping.')
            return None

        return Sample(
            input=[ChatMessageSystem(content=SYSTEM_PROMPTS[suite]), ChatMessageUser(content=data_prompt)],
            target='',
            subset_key=record['subset'],
            metadata={
                'suite': suite,
                'item_id': record['item_id'],
                'attack': record['attack'],
                'goal': record['goal'],
            },
        )

    def match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        """Score 1.0 when the JSON reply satisfies the (attack) objective, else 0.0."""
        score = Score(extracted_prediction=filtered_prediction, prediction=original_prediction)
        metadata = task_state.metadata or {}
        try:
            response = parse_json_response(filtered_prediction)
            satisfied = score_response(metadata.get('suite', ''), metadata.get('goal', '0'), response)
        except (json.JSONDecodeError, ValueError):
            # A reply that violates the JSON contract does not satisfy the objective
            satisfied = False
        score.value = {'asr': float(satisfied)}
        score.main_score_name = 'asr'
        return score
