import json
import re
import traceback
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.dataset.dataset import DatasetDict
from evalscope.api.dataset.loader import DictDataLoader
from evalscope.api.evaluator import TaskState
from evalscope.api.messages.chat_message import ChatMessageUser
from evalscope.api.metric import Score
from evalscope.api.model import Model, ModelOutput
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.report import Category, Report, Subset, unweighted_average_from_subsets, weighted_average_from_subsets
from evalscope.utils.import_utils import check_import
from evalscope.utils.logger import get_logger

logger = get_logger()

ALL_AVAILABLE_MEMORY_BACKENDS = [
    'kv',
    'vector',
    'rec_sum',
]

NON_LIVE_CATEGORY = [
    'simple_python',
    'simple_java',
    'simple_javascript',
    'multiple',
    'parallel',
    'parallel_multiple',
    'irrelevance',
]
LIVE_CATEGORY = [
    'live_simple',
    'live_multiple',
    'live_parallel',
    'live_parallel_multiple',
    'live_irrelevance',
    'live_relevance',
]
MULTI_TURN_CATEGORY = [
    'multi_turn_base',
    'multi_turn_miss_func',
    'multi_turn_miss_param',
    'multi_turn_long_context',
]
WEB_SEARCH_CATEGORY = [
    'web_search_base',
    'web_search_no_snippet',
]

MEMORY_CATEGORY = [f'memory_{backend}' for backend in ALL_AVAILABLE_MEMORY_BACKENDS]
MEMORY_SCENARIO_NAME = [
    'student',
    'customer',
    'finance',
    'healthcare',
    'notetaker',
]

SINGLE_TURN_CATEGORY = NON_LIVE_CATEGORY + LIVE_CATEGORY
AGENTIC_CATEGORY = MEMORY_CATEGORY + WEB_SEARCH_CATEGORY
NON_SCORING_CATEGORY = ['format_sensitivity']

ALL_SCORING_CATEGORIES = SINGLE_TURN_CATEGORY + MULTI_TURN_CATEGORY + AGENTIC_CATEGORY
ALL_CATEGORIES = ALL_SCORING_CATEGORIES + NON_SCORING_CATEGORY

TEST_COLLECTION_MAPPING = {
    'all':
    ALL_CATEGORIES,
    'all_scoring':
    ALL_SCORING_CATEGORIES,
    'multi_turn':
    MULTI_TURN_CATEGORY,
    'single_turn':
    SINGLE_TURN_CATEGORY,
    'live':
    LIVE_CATEGORY,
    'non_live':
    NON_LIVE_CATEGORY,
    'non_python': [
        'simple_java',
        'simple_javascript',
    ],
    'python': [
        'simple_python',
        'irrelevance',
        'parallel',
        'multiple',
        'parallel_multiple',
        'live_simple',
        'live_multiple',
        'live_parallel',
        'live_parallel_multiple',
        'live_irrelevance',
        'live_relevance',
    ],
    'memory':
    MEMORY_CATEGORY,
    'web_search':
    WEB_SEARCH_CATEGORY,
    'agentic':
    AGENTIC_CATEGORY,
}


@register_benchmark(
    BenchmarkMeta(
        name='bfcl_v4',
        pretty_name='BFCL-v4',
        tags=[Tags.FUNCTION_CALLING],
        description='Berkeley Function Calling Leaderboard (BFCL), the **first comprehensive '
        'and executable function call evaluation** '
        'dedicated to assessing Large Language Models\' (LLMs) ability to invoke '
        'functions. Unlike previous evaluations, '
        'BFCL accounts for various forms of function calls, diverse scenarios, and executability. '
        'Need to run `pip install bfcl-eval==2025.10.27.1` before evaluating. '
        '[Usage Example](https://evalscope.readthedocs.io/en/latest/third_party/bfcl_v3.html)',
        dataset_id='https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard',
        subset_list=ALL_SCORING_CATEGORIES,
        metric_list=['acc'],
        eval_split='train',
        extra_params={
            'underscore_to_dot': True,
            'is_fc_model': True,
        }
    )
)
class BFCLAdapter(DefaultDataAdapter):
    """
    BFCL adapter using the new data processing framework.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        check_import('bfcl_eval', package='bfcl-eval==2025.10.27.1', raise_error=True, feature_name=self.pretty_name)

        self.add_overall_metric = False
        self.add_aggregation_name = False

        self.underscore_to_dot = self.extra_params.get('underscore_to_dot', True)
        self.is_fc_model = self.extra_params.get('is_fc_model', True)
        self.model_result_dir = self.extra_params.get(Path(self._task_config.work_dir))

    def load(self):
        """Load and process the BFCL dataset."""
        from bfcl_eval.utils import parse_test_category_argument
        datasets = {}
        all_test_categories = parse_test_category_argument(self.subset_list)

        test_entries_by_cat, ground_truth_by_cat = self._load_bfcl_data(all_test_categories)

        for category in all_test_categories:
            test_entries = test_entries_by_cat.get(category, [])
            ground_truth_entries = ground_truth_by_cat.get(category, [])

            if not test_entries:
                continue

            datasets[category] = self._create_dataset_for_category(category, test_entries, ground_truth_entries)

        test_dataset = DatasetDict(datasets)
        return test_dataset, None

    def _load_bfcl_data(self, categories: List[str]):
        """Load test entries and ground truth data from bfcl_eval."""
        from bfcl_eval.utils import is_relevance_or_irrelevance, load_dataset_entry, load_ground_truth_entry

        test_entries_by_cat = defaultdict(list)
        ground_truth_by_cat = defaultdict(list)

        for category in categories:
            test_entries_by_cat[category] = load_dataset_entry(
                category, include_prereq=False, include_language_specific_hint=False
            )
            if not is_relevance_or_irrelevance(category):
                ground_truth_by_cat[category] = load_ground_truth_entry(category)

        return test_entries_by_cat, ground_truth_by_cat

    def _prepare_ground_truth_map(self, category: str, ground_truth_entries: List[Dict]) -> Dict[str, Dict]:
        """Prepare a map of ground truth entries with category-specific ID adjustments."""
        from bfcl_eval.utils import is_memory, is_web_search

        if is_memory(category):
            return {entry['id'].replace('memory', category): entry for entry in ground_truth_entries}
        if is_web_search(category):
            return {entry['id'].replace('web_search', category): entry for entry in ground_truth_entries}

        return {entry['id']: entry for entry in ground_truth_entries}

    def _create_dataset_for_category(
        self, category: str, test_entries: List[Dict], ground_truth_entries: List[Dict]
    ) -> DatasetDict:
        """Create a dataset for a single category by merging test and ground truth data."""
        from bfcl_eval.utils import (
            populate_initial_settings_for_memory_test_cases,
            populate_initial_settings_for_web_search_test_cases,
        )

        ground_truth_map = self._prepare_ground_truth_map(category, ground_truth_entries)

        test_entries = populate_initial_settings_for_web_search_test_cases(test_entries)
        test_entries = populate_initial_settings_for_memory_test_cases(
            test_entries, model_result_dir=self.model_result_dir
        )

        processed_entries = []
        for entry in test_entries:
            entry_id = entry['id']
            entry['category'] = category
            entry['ground_truth'] = ground_truth_map.get(entry_id, {}).get('ground_truth', {})
            processed_entries.append(entry)

        return DictDataLoader(
            dict_list=processed_entries,
            limit=self.limit,
            repeats=self.repeats,
            sample_fields=self.record_to_sample,
            shuffle=self.shuffle,
        ).load()

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Convert a data record to a Sample object."""
        return Sample(
            input=[ChatMessageUser(content=json.dumps(record['question']))],
            target=json.dumps(record['ground_truth']),  # Will use the record for evaluation
            metadata=record  # Store the full record for evaluation
        )

    def _on_inference(self, model: Model, sample: Sample) -> ModelOutput:
        from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler
        from openai import OpenAI

        handler = OpenAICompletionsHandler(
            model_name=model.name,
            temperature=model.config.temperature,
            registry_name=model.name,
            is_fc_model=self.is_fc_model,
        )
        handler.client = OpenAI(api_key=model.api.api_key, base_url=model.api.base_url)
        result = handler.inference(deepcopy(sample.metadata), include_input_log=False, exclude_state_log=False)
        return ModelOutput(output=result['model_output'], metadata=result)

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:

        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        try:

            score.value = {
                'acc': float(score_result['valid']),
            }
            score.explanation = score_result.get('error_message', 'Evaluation completed')
            score.metadata = {
                'raw_score_result': score_result,
                'test_category': test_category,
                'underscore_to_dot': self.underscore_to_dot,
                'is_fc_model': self.is_fc_model
            }
            score.main_score_name = 'acc'

        except Exception:
            logger.error(f'Evaluation failed for sample: {task_state.sample_id}\n{traceback.format_exc()}')
            score.value = {'acc': 0.0}
            score.explanation = 'Evaluation failed with an unexpected error.'
            score.metadata = {'error': traceback.format_exc()}
            score.main_score_name = 'acc'
        return score

    def _on_generate_report_end(self, report: Report, output_dir, **kwargs):
        """
        Finalize the report generation process. Calculate the overall score.

        Track the number of each category.
        - step1: simple_python, java, javascript unweighted average as simple_ast
        - step2.1: simple_ast, multiple, parallel, parallel_multiple unweighted average as ast_non_live
        - step2.2: live_simple, live_multiple, live_parallel, live_parallel_multiple weighted average as ast_live
        - step2.3: irrelevance as hallucination_non_live
        - step2.4: live_irrelevance, live_relevance weighted average as hallucination_live
        - step2.5: multi_turn_base as multi_turn_base
        - step2.6: multi_turn_miss_func, multi_turn_miss_param, multi_turn_long_context weighted average as multi_turn_augmented
        - step3.1: ast_non_live, hallucination_non_live unweighted average as non_live
        - step3.2: ast_live, hallucination_live weighted average as live
        - step3.3: multi_turn_base, multi_turn_augmented unweighted average as multi_turn
        - step4: non_live, live, multi_turn unweighted average as overall
        Args:
            report (Report): The generated evaluation report.
            output_dir (str): The directory to save the report.

        Returns:
            None
        """  # noqa: E501
        for metric in report.metrics:
            # Collect all subsets in a dictionary for easy access
            subset_dict: Dict[str, Subset] = {}
            for category in metric.categories:
                for subset in category.subsets:
                    subset_dict[subset.name] = subset

            # Step 1: Calculate simple_ast (simple_python, java, javascript unweighted average)
            simple_subsets = ['simple_python', 'java', 'javascript']
            simple_ast = unweighted_average_from_subsets(simple_subsets, subset_dict)
            subset_dict['simple_ast'] = simple_ast

            # Step 2.1: Calculate ast_non_live
            # (simple_ast, multiple, parallel, parallel_multiple unweighted average)
            ast_non_live_subsets = ['simple_ast', 'multiple', 'parallel', 'parallel_multiple']
            ast_non_live = unweighted_average_from_subsets(ast_non_live_subsets, subset_dict)
            subset_dict['ast_non_live'] = ast_non_live

            # Step 2.2: Calculate ast_live
            # (live_simple, live_multiple, live_parallel, live_parallel_multiple weighted average)
            live_subsets = ['live_simple', 'live_multiple', 'live_parallel', 'live_parallel_multiple']
            ast_live = weighted_average_from_subsets(live_subsets, subset_dict)
            subset_dict['ast_live'] = ast_live

            # Step 2.3: hallucination_non_live (irrelevance)
            if 'irrelevance' in subset_dict:
                subset_dict['hallucination_non_live'] = subset_dict['irrelevance']
            else:
                subset_dict['hallucination_non_live'] = Subset(name='hallucination_non_live', score=0, num=0)

            # Step 2.4: Calculate hallucination_live (live_irrelevance, live_relevance weighted average)
            hallucination_live_subsets = ['live_irrelevance', 'live_relevance']
            hallucination_live = weighted_average_from_subsets(hallucination_live_subsets, subset_dict)
            subset_dict['hallucination_live'] = hallucination_live

            # Step 2.5: multi_turn_base
            if 'multi_turn_base' not in subset_dict:
                subset_dict['multi_turn_base'] = Subset(name='multi_turn_base', score=0, num=0)

            # Step 2.6: Calculate multi_turn_augmented
            # (multi_turn_miss_func, multi_turn_miss_param, multi_turn_long_context weighted average)
            multi_turn_augmented_subsets = ['multi_turn_miss_func', 'multi_turn_miss_param', 'multi_turn_long_context']
            multi_turn_augmented = weighted_average_from_subsets(multi_turn_augmented_subsets, subset_dict)
            subset_dict['multi_turn_augmented'] = multi_turn_augmented

            # Step 3.1: Calculate non_live (ast_non_live, hallucination_non_live unweighted average)
            non_live_subsets = ['ast_non_live', 'hallucination_non_live']
            non_live = unweighted_average_from_subsets(non_live_subsets, subset_dict)
            subset_dict['non_live'] = non_live

            # Step 3.2: Calculate live (ast_live, hallucination_live weighted average)
            live_agg_subsets = ['ast_live', 'hallucination_live']
            live = weighted_average_from_subsets(live_agg_subsets, subset_dict)
            subset_dict['live'] = live

            # Step 3.3: Calculate multi_turn (multi_turn_base, multi_turn_augmented unweighted average)
            multi_turn_subsets = ['multi_turn_base', 'multi_turn_augmented']
            multi_turn = unweighted_average_from_subsets(multi_turn_subsets, subset_dict)
            subset_dict['multi_turn'] = multi_turn

            # Step 4: Calculate overall (non_live, live, multi_turn unweighted average)
            overall_subsets = ['non_live', 'live', 'multi_turn']
            overall = unweighted_average_from_subsets(overall_subsets, subset_dict)
            subset_dict['overall'] = overall

            # Add computed scores to the category
            computed_subset_names = ['non_live', 'live', 'multi_turn', 'overall']

            # Add the computed scores as new subsets in the metric
            dummy_subsets = []
            for subset_name in computed_subset_names:
                if subset_name in subset_dict:
                    subset = subset_dict[subset_name]
                    subset.name = subset_name.upper()
                    dummy_subsets.append(subset)
            dummy_category = Category(name='-', subsets=dummy_subsets)
            metric.categories.append(dummy_category)
