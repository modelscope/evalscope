# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import os
from typing import Any, Dict, List, Optional

from evalscope.api.benchmark import AgentAdapter, BenchmarkMeta
from evalscope.api.dataset import Dataset, DatasetDict, Sample
from evalscope.api.evaluator import InferenceResult, TaskState
from evalscope.api.messages import ChatMessageSystem, ChatMessageUser
from evalscope.api.metric import AggScore, SampleScore, Score
from evalscope.api.model import ChatCompletionChoice, GenerateConfig, Model, ModelOutput, get_model
from evalscope.api.registry import register_benchmark
from evalscope.constants import EvalType, Tags
from evalscope.report import Category, Report, Subset, unweighted_average_from_subsets
from evalscope.utils.logger import get_logger

from .checker import (
    check_agent_end_state,
    check_normal_answer,
    check_special_answer,
    milestone_accuracy,
    multi_turn_accuracy,
)
from .parser import CallFormatError, decode_calls
from .prompts import build_single_turn_prompts
from .utils import (
    ACE_DATA_CATEGORY,
    ACEBENCH_CATEGORIES,
    ACEBENCH_LANGUAGES,
    ACEBENCH_SPLITS,
    category_of_record,
    decode_maybe_json,
    dialogue_id_of,
    resolve_categories,
    split_of_category,
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

- 1023 English and 1017 Chinese samples, selectable through `extra_params.language`.
- Uses the official ACEBench prompts and the official `[ApiName(...)]` output contract, so an
  output that cannot be decoded scores zero instead of being rescued by lenient parsing.
- `normal_multi_turn_*` categories are scored per dialogue: every step must be correct for the
  dialogue to count, matching the official turn-level aggregation.
- `agent` categories run a real rollout against ACEBench's simulated phone, food-delivery and
  travel APIs, and are graded on the resulting environment state.

## Evaluation Notes

- `accuracy` is the primary metric. For `normal` and `special` it is answer accuracy; for `agent` it is
  end-state accuracy. `process_acc` additionally reports milestone progress for `agent` samples and
  per-step progress for `normal_multi_turn_*` samples.
- The report adds the official groupings (ATOM, SINGLE_TURN, MULTI_TURN, NORMAL, SPECIAL, AGENT)
  and an OVERALL score weighted `normal` 0.578 / `special` 0.2676 / `agent` 0.1545. Weights are
  renormalized over the groups actually evaluated, so a partial run stays interpretable.
- `agent_multi_turn` additionally needs a user simulator; set `extra_params.user_model` to the model
  that should play the user (the official runner uses `gpt-4o`). Without it those rollouts fail and
  score zero, so configure it before reading an OVERALL number.
""",
        dataset_id='evalscope/acebench',
        subset_list=list(ACEBENCH_CATEGORIES),
        default_subset='en',
        metric_list=['acc', 'process_acc'],
        primary_metric='accuracy',
        eval_split='normal',
        extra_params={
            'language': {
                'type': 'str',
                'description': 'Dataset language to evaluate, either `en` or `zh`.',
                'value': 'en',
            },
            'user_model': {
                'type': 'str',
                'description': 'Model that plays the user in `agent_multi_turn` rollouts, '
                'e.g. `gpt-4o`. Those rollouts fail and score zero when unset.',
                'value': '',
            },
            'user_model_api_url': {
                'type': 'str',
                'description': 'Base URL for `user_model`. Defaults to `MODELSCOPE_API_BASE`.',
                'value': '',
            },
            'user_model_api_key': {
                'type': 'str',
                'description': 'API key for `user_model`. Defaults to `MODELSCOPE_SDK_TOKEN`.',
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
class AceBenchAdapter(AgentAdapter):
    """ACEBench adapter following the official prompt, decoding and scoring protocol."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.split_as_subset = True
        self.reformat_subset = True
        self.add_overall_metric = False

        self.category_map = dict(ACEBENCH_CATEGORIES)

        self.language = str(self.extra_params.get('language') or self.default_subset or 'en')
        if self.language not in ACEBENCH_LANGUAGES:
            raise ValueError(f'ACEBench language must be one of {ACEBENCH_LANGUAGES}, got {self.language!r}.')
        # The hub dataset keeps one configuration per language and one split per data family.
        self.default_subset = self.language

        self.user_model_id = str(self.extra_params.get('user_model') or '')
        self.max_dialog_turns = int(self.extra_params.get('max_dialog_turns', 40))
        self._user_model: Optional[Model] = None

        self.subset_list = resolve_categories(self.subset_list)
        if 'agent_multi_turn' in self.subset_list and not self.user_model_id:
            # Warn once here rather than per sample: those rollouts need a second model to play the
            # user, and without one they fail and score zero, which drags the AGENT group down.
            logger.warning(
                'The agent_multi_turn category needs a user simulator and will score zero without '
                "one. Set dataset_args={'acebench': {'extra_params': {'user_model': '<model-id>'}}} "
                'to evaluate it (the official runner uses gpt-4o).'
            )

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
        test_category = category_of_record(record)

        record = {**record, 'function': functions}
        system_prompt, user_prompt = build_single_turn_prompts(record, test_category, self.language)
        ground_truth = rubric.get('ground_truth', {})
        milestones = rubric.get('mile_stone', [])

        return Sample(
            input=[
                ChatMessageSystem(content=system_prompt),
                ChatMessageUser(content=user_prompt),
            ],
            target=json.dumps({
                'ground_truth': ground_truth,
                'mile_stone': milestones
            }, ensure_ascii=False),
            subset_key=test_category,
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
    # INFERENCE
    # #########################

    def _on_inference(self, model: Model, sample: Sample):
        """Run the agent rollout for agent samples, and a single generation otherwise."""
        test_category = (sample.metadata or {}).get('test_category', '')
        if 'agent' not in test_category:
            # ACEBench is a prompt-only protocol: the API specs live in the system prompt and the
            # answer must be a ``[ApiName(...)]`` list, so no native tool schemas are attached.
            return model.generate(input=sample.input)

        from .rollout import run_rollout

        try:
            result = run_rollout(
                model=model,
                metadata=sample.metadata,
                max_steps=self.max_dialog_turns,
                user_model=self._get_user_model() if 'multi_turn' in test_category else None,
            )
        except Exception as error:  # noqa: BLE001 - a failed rollout must not abort the run
            logger.error(f'ACEBench rollout failed for {sample.metadata.get("id")}: {error}')
            sample.metadata['rollout_error'] = str(error)
            return ModelOutput.from_content(model=model.name, content='')

        # match_score grades the recorded state, not the text, so both are carried on the metadata.
        sample.metadata['process'] = result.process
        sample.metadata['end_state'] = result.end_state
        output = ModelOutput(
            model=model.name,
            choices=[ChatCompletionChoice.from_content('\n'.join(result.process))],
            usage=result.usage,
        )
        return InferenceResult(output=output, messages=result.messages, trace=result.trace)

    def _get_user_model(self) -> Optional[Model]:
        """Build the model that plays the user in ``agent_multi_turn`` rollouts."""
        if not self.user_model_id:
            return None
        if self._user_model is None:
            self._user_model = get_model(
                model=self.user_model_id,
                eval_type=EvalType.OPENAI_API,
                base_url=self.extra_params.get('user_model_api_url') or os.environ.get('MODELSCOPE_API_BASE'),
                api_key=self.extra_params.get('user_model_api_key') or os.environ.get('MODELSCOPE_SDK_TOKEN'),
                config=GenerateConfig(temperature=0.001, top_p=1, max_tokens=1000),
            )
        return self._user_model

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
            result = self._score_agent(metadata)
        else:
            result = self._score_normal(filtered_prediction, metadata, test_category)

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

    def _score_normal(self, prediction: str, metadata: Dict[str, Any], test_category: str) -> Dict[str, Any]:
        """Decode the answer and check it against the ground truth."""
        try:
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

    def _score_agent(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Score an agent sample on its end state, and report milestone progress alongside."""
        # ``process`` is absent only when the rollout itself failed, in which case nothing was
        # executed and no milestone can have been reached.
        process_acc = milestone_accuracy(metadata.get('process') or [], metadata.get('mile_stone'))

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
            end_score, process_score = multi_turn_accuracy([step.score.value.get('acc', 0.0) == 1.0 for step in steps])
            end_scores.append(end_score)
            process_scores.append(process_score)

        ids = [sample_score.sample_id for sample_score in sample_scores]
        return [
            AggScore(
                score=sum(end_scores) / len(end_scores),
                metric_name='accuracy',
                aggregation=self.aggregation,
                num=len(dialogues),
                ids=ids,
            ),
            AggScore(
                score=sum(process_scores) / len(process_scores),
                metric_name='process_acc',
                aggregation=self.aggregation,
                num=len(dialogues),
                ids=ids,
            ),
        ]

    def _on_generate_report_end(self, report: Report, output_dir: str, **kwargs) -> None:
        """Append the official ACEBench groupings and the weighted OVERALL score."""
        for metric in report.metrics:
            if metric.identity.name != 'accuracy':
                continue

            subset_dict: Dict[str, Subset] = {
                subset.name: subset
                for category in metric.categories
                for subset in category.subsets
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
