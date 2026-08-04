# Copyright (c) Alibaba, Inc. and its affiliates.
import json
from typing import Any, Dict, List, Optional

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Dataset, DatasetDict, Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageSystem, ChatMessageUser
from evalscope.api.metric import AggScore, SampleScore, Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.report import Category, Report, Subset, unweighted_average_from_subsets
from evalscope.utils.logger import get_logger
from .checker import check_agent_end_state, check_normal_answer, check_special_answer, milestone_accuracy
from .parser import CallFormatError, decode_calls, decode_tool_calls
from .prompts import build_single_turn_prompts
from .utils import (
    ACE_DATA_CATEGORY,
    ACEBENCH_CATEGORIES,
    ACEBENCH_LANGUAGES,
    ACEBENCH_SPLITS,
    build_tool_infos,
    decode_maybe_json,
    dialogue_id_of,
    extract_bracket_blocks,
    resolve_categories,
    split_of_category,
    test_category_of,
)

logger = get_logger()

# Official leaderboard weights, see ``generate_result_csv`` in ACEBench/model_eval/evaluation_helper.py
OFFICIAL_GROUP_WEIGHTS = {'normal': 0.578, 'special': 0.2676, 'agent': 0.1545}

_REPORT_GROUPS = {
    'ATOM': ACE_DATA_CATEGORY['atom'],
    'SINGLE_TURN': ['normal_single_turn_single_function', 'normal_single_turn_parallel_function'],
    'MULTI_TURN': ACE_DATA_CATEGORY['multi_turn'],
    'NORMAL': ACE_DATA_CATEGORY['normal'],
    'SPECIAL': ACE_DATA_CATEGORY['special'],
    'AGENT': ACE_DATA_CATEGORY['agent'],
}


@register_benchmark(
    BenchmarkMeta(
        name='acebench',
        pretty_name='ACEBench',
        tags=[Tags.FUNCTION_CALLING, Tags.AGENT, Tags.MULTI_TURN],
        description="""
## Overview

ACEBench evaluates whether large language models can use tools in realistic settings: picking the
right API, filling its arguments, pushing back on requests that cannot be satisfied, and driving
multi-step agent tasks against a simulated environment. Data is split into three families -
`normal` (ordinary tool use), `special` (incomplete, incorrect or out-of-scope requests) and
`agent` (multi-step and multi-turn interaction) - reported over 17 fine-grained categories.

## Task Description

- **Task Type**: Function calling and agentic tool use
- **Input**: Conversation history, API specifications, and optional time or character-profile context
- **Output**: A `[ApiName(key='value')]` call list, a diagnostic sentence, or a full agent trajectory
- **Domain**: 8 domains and 68 sub-domains including technology, finance, health and society

## Key Features

- 1023 samples per language across English and Chinese, selectable through `extra_params.language`.
- Uses the official ACEBench prompts and the official `[ApiName(...)]` output contract, so an
  output that cannot be decoded scores zero instead of being rescued by lenient parsing.
- `normal_multi_turn_*` categories are scored per dialogue: every step must be correct for the
  dialogue to count, matching the official turn-level aggregation.
- `agent` categories run a real rollout against ACEBench's simulated phone, food-delivery and
  travel APIs, and are graded on the resulting environment state.

## Evaluation Notes

- `acc` is the primary metric. For `normal` and `special` it is answer accuracy; for `agent` it is
  end-state accuracy. `process_acc` additionally reports milestone progress for `agent` samples and
  per-step progress for `normal_multi_turn_*` samples.
- The report adds the official groupings (ATOM, SINGLE_TURN, MULTI_TURN, NORMAL, SPECIAL, AGENT)
  and an OVERALL score weighted `normal` 0.578 / `special` 0.2676 / `agent` 0.1545. Weights are
  renormalized over the groups actually evaluated, so a partial run stays interpretable.
- `agent_multi_turn` additionally needs a user simulator; set `extra_params.user_model` to the model
  that should play the user (the official runner uses `gpt-4o`). Without it those samples are
  skipped rather than silently scored.
- Set `extra_params.is_fc_model=true` to evaluate through native tool calling. This deviates from
  the official prompt-only protocol, so such numbers are not directly comparable to the leaderboard.
""",
        dataset_id='evalscope/acebench',
        subset_list=list(ACEBENCH_CATEGORIES),
        default_subset='en',
        metric_list=['acc', 'process_acc'],
        eval_split='normal',
        extra_params={
            'language': {
                'type': 'str',
                'description': 'Dataset language to evaluate, either `en` or `zh`.',
                'value': 'en',
            },
            'is_fc_model': {
                'type': 'bool',
                'description': 'Evaluate through native tool calling instead of the official '
                'prompt-only protocol. Not comparable with the official leaderboard.',
                'value': False,
            },
            'user_model': {
                'type': 'str',
                'description': 'Model that plays the user in `agent_multi_turn` rollouts, '
                'e.g. `gpt-4o`. Samples are skipped when unset.',
                'value': '',
            },
            'max_dialog_turns': {
                'type': 'int',
                'description': 'Maximum number of agent rollout steps.',
                'value': 40,
            },
        },
    )
)
class AceBenchAdapter(DefaultDataAdapter):
    """ACEBench adapter following the official prompt, decoding and scoring protocol."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.split_as_subset = True
        self.reformat_subset = True
        self.add_aggregation_name = False
        self.add_overall_metric = False

        self.category_map = dict(ACEBENCH_CATEGORIES)

        self.language = str(self.extra_params.get('language') or self.default_subset or 'en')
        if self.language not in ACEBENCH_LANGUAGES:
            raise ValueError(f'ACEBench language must be one of {ACEBENCH_LANGUAGES}, got {self.language!r}.')
        # The hub dataset keeps one configuration per language and one split per data family.
        self.default_subset = self.language

        self.is_fc_model = bool(self.extra_params.get('is_fc_model', False))
        self.user_model = str(self.extra_params.get('user_model') or '')
        self.max_dialog_turns = int(self.extra_params.get('max_dialog_turns', 40))

        self.subset_list = resolve_categories(self.subset_list)

    # #########################
    # DATASET LOADING
    # #########################

    def load_subsets(self, load_func, is_fewshot: bool = False) -> DatasetDict:
        """Load the ACEBench splits and re-bucket their samples into fine-grained categories."""
        dataset_dicts = []
        for split in ACEBENCH_SPLITS:
            categories = [category for category in self.subset_list if split_of_category(category) == split]
            if not categories:
                continue
            with self._temporary_attribute('current_subset_name', split):
                dataset: Dataset = load_func(split)
            dataset_dicts.append(
                DatasetDict.from_dataset(
                    dataset=dataset,
                    subset_list=categories,
                    limit=self.few_shot_num if is_fewshot else self.limit,
                    repeats=1 if is_fewshot else self.repeats,
                )
            )
        return DatasetDict.from_dataset_dicts(dataset_dicts)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Convert an ACEBench record into a Sample carrying the official prompt pair."""
        functions = decode_maybe_json(record.get('function'), [])
        rubric = decode_maybe_json(record.get('rubric'), {})
        record_id = record.get('id') or ''
        test_category = test_category_of(record)

        record = {**record, 'function': functions}
        system_prompt, user_prompt = build_single_turn_prompts(record, test_category, self.language)
        ground_truth = rubric.get('ground_truth', {})
        milestones = rubric.get('mile_stone', [])

        return Sample(
            input=[
                ChatMessageSystem(content=system_prompt),
                ChatMessageUser(content=user_prompt),
            ],
            target=json.dumps({'ground_truth': ground_truth, 'mile_stone': milestones}, ensure_ascii=False),
            subset_key=test_category,
            # Native tool schemas are only attached in the (non-official) function-calling mode.
            tools=build_tool_infos(functions) if self.is_fc_model else [],
            metadata={
                'id': record_id,
                'test_category': test_category,
                'dialogue_id': dialogue_id_of(record_id, test_category),
                'language': self.language,
                'functions': functions,
                'ground_truth': ground_truth,
                'mile_stone': milestones,
                'initial_config': decode_maybe_json(record.get('initial_config'), {}),
                'involved_classes': decode_maybe_json(record.get('involved_classes'), []),
                'question': record.get('question', ''),
                'time': record.get('time', ''),
                'profile': record.get('profile', ''),
            },
        )

    # #########################
    # SCORING
    # #########################

    def match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        score = Score(extracted_prediction=filtered_prediction, prediction=original_prediction)
        metadata = task_state.metadata or {}
        test_category = metadata.get('test_category', '')

        if 'special' in test_category:
            result = self._score_special(filtered_prediction, metadata, test_category)
        elif 'agent' in test_category:
            result = self._score_agent(filtered_prediction, metadata)
        else:
            result = self._score_normal(filtered_prediction, metadata, test_category, task_state)

        score.value = {key: value for key, value in result.items() if key in {'acc', 'process_acc'}}
        score.main_score_name = 'acc'
        score.explanation = _first_error(result) or 'Evaluation completed'
        score.metadata = {
            'test_category': test_category,
            'valid': result.get('valid', False),
            'error': result.get('error', []),
            'error_type': result.get('error_type', ''),
            'predicted_calls': result.get('predicted_calls'),
        }
        return score

    def _score_normal(
        self,
        prediction: str,
        metadata: Dict[str, Any],
        test_category: str,
        task_state: TaskState,
    ) -> Dict[str, Any]:
        """Decode the answer and check it against the ground truth."""
        try:
            if self.is_fc_model:
                predicted_calls = decode_tool_calls(task_state.output)
            else:
                predicted_calls = decode_calls(prediction, test_category)
        except CallFormatError as error:
            return {'valid': False, 'acc': 0.0, 'error': [str(error)], 'error_type': 'wrong_output_format'}

        result = check_normal_answer(
            metadata.get('functions') or [],
            predicted_calls,
            metadata.get('ground_truth'),
            test_category,
        )
        return {**result, 'acc': 1.0 if result['valid'] else 0.0, 'predicted_calls': predicted_calls}

    def _score_special(self, prediction: str, metadata: Dict[str, Any], test_category: str) -> Dict[str, Any]:
        """Check a special sample against the diagnostic-string contract."""
        result = check_special_answer(prediction, metadata.get('ground_truth'), test_category)
        return {**result, 'acc': 1.0 if result['valid'] else 0.0}

    def _score_agent(self, prediction: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Score an agent sample on its end state, and report milestone progress alongside."""
        process_trace = metadata.get('process')
        if process_trace is None:
            # No rollout was performed: fall back to the call blocks present in the raw answer.
            process_trace = extract_bracket_blocks(prediction)
        process_acc = milestone_accuracy(process_trace, metadata.get('mile_stone'))

        end_state = metadata.get('end_state')
        if end_state is None:
            return {
                'valid': False,
                'acc': 0.0,
                'process_acc': process_acc,
                'error': ['No environment state was recorded for this agent sample.'],
                'error_type': 'missing_end_state',
            }

        result = check_agent_end_state(end_state, metadata.get('ground_truth'))
        acc = 1.0 if result['valid'] else 0.0
        # Upstream credits full process accuracy whenever the end state is correct.
        return {**result, 'acc': acc, 'process_acc': 1.0 if acc else process_acc}

    # #########################
    # AGGREGATION
    # #########################

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        """Aggregate per sample, or per dialogue for the ``normal_multi_turn_*`` categories."""
        test_category = _test_category_of_scores(sample_scores)
        if not (test_category and 'multi_turn' in test_category and 'agent' not in test_category):
            return super().aggregate_scores(sample_scores)

        dialogues: Dict[Any, List[SampleScore]] = {}
        for sample_score in sample_scores:
            dialogue_id = (sample_score.sample_metadata or {}).get('dialogue_id') or sample_score.sample_id
            dialogues.setdefault(dialogue_id, []).append(sample_score)

        end_scores, process_scores = [], []
        for steps in dialogues.values():
            correct = [step.score.value.get('acc', 0.0) == 1.0 for step in steps]
            end_scores.append(0.0 if False in correct else 1.0)
            process_scores.append(round(correct.count(True) / len(correct), 3))

        ids = [sample_score.sample_id for sample_score in sample_scores]
        return [
            AggScore(
                score=sum(end_scores) / len(end_scores),
                metric_name='acc',
                aggregation_name=self.aggregation,
                num=len(dialogues),
                ids=ids,
            ),
            AggScore(
                score=sum(process_scores) / len(process_scores),
                metric_name='process_acc',
                aggregation_name=self.aggregation,
                num=len(dialogues),
                ids=ids,
            ),
        ]

    def _on_generate_report_end(self, report: Report, output_dir: str, **kwargs) -> None:
        """Append the official ACEBench groupings and the weighted OVERALL score."""
        for metric in report.metrics:
            if metric.name != 'acc':
                continue

            subset_dict: Dict[str, Subset] = {
                subset.name: subset
                for category in metric.categories for subset in category.subsets
            }

            group_subsets = {}
            for group_name, categories in _REPORT_GROUPS.items():
                group_subsets[group_name] = unweighted_average_from_subsets(categories, subset_dict, group_name)

            overall = self._weighted_overall(group_subsets)
            reported = [subset for subset in group_subsets.values() if subset.num > 0]
            if overall is not None:
                reported.append(overall)
            if reported:
                metric.categories.append(Category(name='-', subsets=reported))

    @staticmethod
    def _weighted_overall(group_subsets: Dict[str, Subset]) -> Optional[Subset]:
        """Combine the three families with the official weights, renormalized over what ran."""
        total_weight = 0.0
        total_score = 0.0
        total_num = 0
        for group_name, weight in OFFICIAL_GROUP_WEIGHTS.items():
            subset = group_subsets.get(group_name.upper())
            if subset is None or subset.num == 0:
                continue
            total_weight += weight
            total_score += subset.score * weight
            total_num += subset.num
        if total_weight == 0:
            return None
        return Subset(name='OVERALL', score=total_score / total_weight, num=total_num)


def _first_error(result: Dict[str, Any]) -> str:
    """Render the first error entry of a check result as text."""
    errors = result.get('error') or []
    if isinstance(errors, str):
        return errors
    return str(errors[0]) if errors else ''


def _test_category_of_scores(sample_scores: List[SampleScore]) -> str:
    """Read the ACEBench category the given sample scores belong to."""
    for sample_score in sample_scores:
        category = (sample_score.sample_metadata or {}).get('test_category')
        if category:
            return category
    return ''
