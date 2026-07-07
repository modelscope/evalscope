# Copyright (c) Alibaba, Inc. and its affiliates.
import importlib
import json
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

DATASET_ID = 'evalscope/PerspectiveGap'
PERSPECTIVE_GAP_COMMIT = '9c6921b3337ff3e6a6a453f68d117a8c1663135e'
PERSPECTIVE_GAP_REQUIREMENT = (
    'perspective-gap @ '
    f'git+https://github.com/WhymustIhaveaname/PerspectiveGap.git@{PERSPECTIVE_GAP_COMMIT}'
)
INSTALL_HINT = (
    'PerspectiveGap scoring is required for this benchmark. Install it with '
    f"`pip install '{PERSPECTIVE_GAP_REQUIREMENT}'` or `uv pip install '{PERSPECTIVE_GAP_REQUIREMENT}'`."
)
ROLE_ASSIGNMENT = 'role_assignment'
PROMPT_WRITING = 'prompt_writing'
STRICT_PASS = 'strict_pass'
SUBMETRICS = ['net_match_score', 'required_coverage', 'boundary_precision', 'distractor_leakage']

DESCRIPTION = """
## Overview

PerspectiveGap evaluates whether a model can compose orchestration prompts for multi-agent systems while routing only the context each sub-agent needs.

## Tasks

- `perspective_gap_role_assignment`: select the visible fragment IDs for each role and return a JSON object.
- `perspective_gap_prompt_writing`: write one markdown prompt section per role while including only the needed fragments.

## Data

The benchmark uses the ModelScope dataset `evalscope/PerspectiveGap`, which contains the released `test` split. You can also pass `dataset_args.<task>.local_path` to a local JSONL mirror with the same fields.

## Scoring

Scores are computed by `perspective_gap.scoring` from the official PerspectiveGap repository. The scorer is imported lazily so EvalScope can list benchmarks without installing the optional dependency.
""".strip()  # noqa: E501


def _load_scoring_module():
    try:
        return importlib.import_module('perspective_gap.scoring')
    except ImportError as exc:
        raise ImportError(INSTALL_HINT) from exc


def _json_target(reference_need_sets: Dict[str, Any]) -> str:
    return json.dumps(reference_need_sets, ensure_ascii=False)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _score_from_result(result: Dict[str, Any], prediction: str, filtered_prediction: str) -> Score:
    metrics = result.get('metrics') or {}
    counts = result.get('counts') or {}
    strict_pass_value = metrics.get(STRICT_PASS)
    if strict_pass_value is None:
        strict_pass_value = result.get('pass')

    value = {STRICT_PASS: _safe_float(strict_pass_value)}
    for metric_name in SUBMETRICS:
        if metric_name in metrics:
            value[metric_name] = _safe_float(metrics.get(metric_name))

    score = Score(
        value=value,
        prediction=prediction,
        extracted_prediction=filtered_prediction,
        explanation=result.get('error'),
        metadata={
            'metrics': metrics,
            'counts': counts,
            'raw_result': result,
        },
        main_score_name=STRICT_PASS,
    )
    return score


class PerspectiveGapBaseAdapter(DefaultDataAdapter):
    task_name = ''
    prompt_field = ''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._use_llm_judge = False

    def load_from_disk(self, **kwargs):
        return super().load_from_disk(use_local_loader=True)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        reference_need_sets = record.get('reference_need_sets') or {}
        return Sample(
            input=record[self.prompt_field],
            target=_json_target(reference_need_sets),
            metadata={
                'task': self.task_name,
                'evaluation_id': record.get('evaluation_id'),
                'scenario_id': record.get('scenario_id'),
                'shuffle_seed': record.get('shuffle_seed'),
                'roles': record.get('roles') or [],
                'fragments': record.get('fragments') or [],
                'distractor_id': record.get('distractor_id'),
                'reference_need_sets': reference_need_sets,
            },
        )

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        scoring = _load_scoring_module()
        metadata = task_state.metadata
        try:
            result = self._score(scoring, filtered_prediction, metadata)
        except Exception as exc:
            logger.error('PerspectiveGap %s scoring failed: %s', self.task_name, exc)
            result = {
                'pass': False,
                'metrics': {
                    STRICT_PASS: 0.0,
                    'net_match_score': 0.0,
                    'required_coverage': 0.0,
                    'boundary_precision': 0.0,
                    'distractor_leakage': 0.0,
                },
                'counts': {
                    'tp': 0,
                    'fp': 0,
                    'fn': 0,
                    'distractor_leak': 0
                },
                'error': str(exc),
            }
        return _score_from_result(result, original_prediction, filtered_prediction)

    def _score(self, scoring, filtered_prediction: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


@register_benchmark(
    BenchmarkMeta(
        name='perspective_gap_role_assignment',
        pretty_name='PerspectiveGap Role Assignment',
        dataset_id=DATASET_ID,
        tags=[Tags.AGENT, Tags.INSTRUCTION_FOLLOWING],
        description=DESCRIPTION,
        paper_url='https://arxiv.org/abs/2606.08878',
        subset_list=['default'],
        metric_list=[STRICT_PASS, *SUBMETRICS],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template='{question}',
    )
)
class PerspectiveGapRoleAssignmentAdapter(PerspectiveGapBaseAdapter):
    task_name = ROLE_ASSIGNMENT
    prompt_field = 'role_assignment_prompt'

    def _score(self, scoring, filtered_prediction: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        metadata = metadata or {}
        return scoring.score_role_assignment(
            filtered_prediction,
            metadata.get('reference_need_sets') or {},
            metadata.get('distractor_id'),
        )


@register_benchmark(
    BenchmarkMeta(
        name='perspective_gap_prompt_writing',
        pretty_name='PerspectiveGap Prompt Writing',
        dataset_id=DATASET_ID,
        tags=[Tags.AGENT, Tags.INSTRUCTION_FOLLOWING],
        description=DESCRIPTION,
        paper_url='https://arxiv.org/abs/2606.08878',
        subset_list=['default'],
        metric_list=[STRICT_PASS, *SUBMETRICS],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template='{question}',
    )
)
class PerspectiveGapPromptWritingAdapter(PerspectiveGapBaseAdapter):
    task_name = PROMPT_WRITING
    prompt_field = 'prompt_writing_prompt'

    def _score(self, scoring, filtered_prediction: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        metadata = metadata or {}
        return scoring.score_prompt_writing(
            filtered_prediction,
            metadata.get('fragments') or [],
            metadata.get('reference_need_sets') or {},
            metadata.get('distractor_id'),
        )
