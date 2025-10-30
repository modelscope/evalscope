import json
import os
import traceback
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm
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
from evalscope.report import (
    Category,
    Report,
    Subset,
    percentage_weighted_average_from_subsets,
    unweighted_average_from_subsets,
    weighted_average_from_subsets,
)
from evalscope.utils.function_utils import thread_safe
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

ALL_SCORING_CATEGORIES = SINGLE_TURN_CATEGORY + MULTI_TURN_CATEGORY + AGENTIC_CATEGORY

DUMMY_MODEL_UNDERSCORE_TO_DOT = 'gpt-4o-2024-11-20-FC'
DUMMY_MODEL_NO_UNDERSCORE_TO_DOT = 'meta-llama/Llama-3.3-70B-Instruct-FC'


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
            'SERPAPI_API_KEY': 'SERPAPI_API_KEY must be set in environment variables for `web_search` categories.',
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
        self.model_result_dir = Path(self._task_config.work_dir)
        self.handler = None
        self.prereq_entries = []
        self.prereq_finished = False

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
            test_entries_by_cat[category] = load_dataset_entry(category)
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
            clean_up_memory_prereq_entries,
            is_memory_prereq,
            populate_initial_settings_for_memory_test_cases,
            populate_initial_settings_for_web_search_test_cases,
        )

        ground_truth_map = self._prepare_ground_truth_map(category, ground_truth_entries)

        test_entries = clean_up_memory_prereq_entries(test_entries)
        self.prereq_entries = [entry for entry in test_entries if is_memory_prereq(entry['id'])]
        test_entries = populate_initial_settings_for_web_search_test_cases(test_entries)
        test_entries = populate_initial_settings_for_memory_test_cases(
            test_entries, model_result_dir=self.model_result_dir
        )
        test_entries = [entry for entry in test_entries if not is_memory_prereq(entry['id'])]

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

    @thread_safe
    def _on_inference_start(self, model, sample):
        if self.handler is not None:
            return  # Handler already initialized

        from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler

        # Set env variables for OpenAI API
        os.environ['OPENAI_API_KEY'] = model.api.api_key
        os.environ['OPENAI_BASE_URL'] = model.api.base_url

        self.handler = OpenAICompletionsHandler(
            model_name=model.name,
            temperature=model.config.temperature,
            registry_name=model.name,
            is_fc_model=self.is_fc_model,
        )

        self._prereq_inference()

    def _prereq_inference(self):
        if self.prereq_finished:
            return

        from bfcl_eval.utils import get_directory_structure_by_id

        for entry in tqdm(self.prereq_entries, desc='Running prereq inferences for memory snapshots...'):
            memory_snapshot_folder = (
                self.model_result_dir / get_directory_structure_by_id(entry['id']) / 'memory_snapshot'
                / 'prereq_checkpoints'
            )
            existing_filenames = {f.name for f in memory_snapshot_folder.rglob('*.json')}
            if (entry['id'] + '.json') in existing_filenames:

                logger.info(f'Skipping prereq inference for entry ID {entry["id"]} as result already exists.')
                continue

            try:
                self.handler.inference(deepcopy(entry), include_input_log=False, exclude_state_log=False)
            except Exception as e:
                logger.error(f'Error during prereq inference for entry ID {entry.get("id")}: {e}')
                logger.error(traceback.format_exc())

        self.prereq_finished = True

    def _on_inference(self, model: Model, sample: Sample) -> ModelOutput:
        try:
            result, _ = self.handler.inference(
                deepcopy(sample.metadata), include_input_log=False, exclude_state_log=False
            )

            output = ModelOutput.from_content(
                model=model.name,
                content=json.dumps(result),
            )
        except Exception as e:
            # This is usually the case when the model getting stuck on one particular test case.
            # For example, timeout error or FC model returning invalid JSON response.
            # Since temperature is already set to 0.001, retrying the same test case will not help.
            # So we continue the generation process and record the error message as the model response
            logger.error(f'Error during inference for sample ID {sample.metadata.get("id")}: {e}')
            logger.error(traceback.format_exc())

            output = ModelOutput.from_content(
                model=model.name,
                content=json.dumps({
                    'error': str(e),
                    'error_message': traceback.format_exc(),
                }),
            )
        return output

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        from bfcl_eval.constants.enums import Language, ReturnFormat
        from bfcl_eval.eval_checker.eval_runner import (
            _evaluate_single_agentic_entry,
            _evaluate_single_ast_entry,
            _evaluate_single_multi_turn_entry,
            _evaluate_single_relevance_entry,
        )
        from bfcl_eval.utils import is_agentic, is_java, is_js, is_multi_turn, is_relevance_or_irrelevance

        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )
        model_result = json.loads(filtered_prediction)
        prompt = task_state.metadata
        test_category = prompt['category']
        index = prompt['id']
        ground_truth = prompt.get('ground_truth', {})
        # NOTE: This is hardcoded dummy model since its only use is to infer underscore_to_dot
        model_name = DUMMY_MODEL_UNDERSCORE_TO_DOT if self.underscore_to_dot else DUMMY_MODEL_NO_UNDERSCORE_TO_DOT

        if is_relevance_or_irrelevance(test_category):
            entry_result = _evaluate_single_relevance_entry(
                handler=self.handler,
                index=index,
                model_result_item=model_result,
                prompt_entry=prompt,
                model_name=model_name,
                test_category=test_category
            )
        elif is_multi_turn(test_category):
            entry_result = _evaluate_single_multi_turn_entry(
                handler=self.handler,
                test_entry_id=index,
                model_result_list=model_result,
                ground_truth_list=ground_truth,
                prompt_entry=prompt,
                model_name=model_name,
                test_category=test_category
            )
        elif is_agentic(test_category):
            entry_result = _evaluate_single_agentic_entry(
                handler=self.handler,
                index=index,
                model_result_list=model_result,
                possible_answer_item=ground_truth,
                prompt_entry=prompt,
                model_name=model_name,
                test_category=test_category
            )
        else:
            if is_java(test_category):
                language = Language.JAVA
                return_format = ReturnFormat.JAVA
            elif is_js(test_category):
                language = Language.JAVASCRIPT
                return_format = ReturnFormat.JAVASCRIPT
            else:
                language = Language.PYTHON
                return_format = ReturnFormat.PYTHON
            entry_result = _evaluate_single_ast_entry(
                handler=self.handler,
                index=prompt['id'],
                model_result_item=model_result,
                possible_answer_item=ground_truth,
                prompt_entry=prompt,
                model_name=model_name,
                test_category=test_category,
                language=language,
                return_format=return_format,
            )

        valid = 1 if entry_result['valid'] else 0
        score.value = {'acc': valid}
        score.metadata = {
            'valid': bool(entry_result.get('valid')),
            'error': str(entry_result.get('error')),
            'error_message': str(entry_result.get('error_message')),
            'error_type': str(entry_result.get('error_type')),
        }
        return score

    def _on_generate_report_end(self, report: Report, output_dir, **kwargs):
        """
        Finalize the report generation process. Calculate the overall score.

        Track the number of each category.
        - step1: simple_python, java, javascript unweighted average as simple_ast
        - step2.1: simple_ast, multiple, parallel, parallel_multiple unweighted average as ast_non_live
        - step2.2: live_simple, live_multiple, live_parallel, live_parallel_multiple weighted average as ast_live
        - step2.3: irrelevance, live_irrelevance as hallucination
        - step2.4: multi_turn_base as multi_turn_base
        - step2.5: multi_turn_miss_func, multi_turn_miss_param, multi_turn_long_context weighted average as multi_turn_augmented
        - step3.1: ast_non_live, hallucination unweighted average as non_live
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

            # Step 1: Calculate simple_ast (simple_python, simple_java, simple_javascript unweighted average)
            simple_subsets = ['simple_python', 'simple_java', 'simple_javascript']
            simple_ast = unweighted_average_from_subsets(simple_subsets, subset_dict)
            subset_dict['simple_ast'] = simple_ast

            # Step 2.1: Calculate non_live
            # (simple_ast, multiple, parallel, parallel_multiple unweighted average)
            non_live_subsets = ['simple_ast', 'multiple', 'parallel', 'parallel_multiple']
            non_live = unweighted_average_from_subsets(non_live_subsets, subset_dict)
            subset_dict['non_live'] = non_live

            # Step 2.2: Calculate live
            # (live_simple, live_multiple, live_parallel, live_parallel_multiple weighted average)
            live_subsets = ['live_simple', 'live_multiple', 'live_parallel', 'live_parallel_multiple']
            live = weighted_average_from_subsets(live_subsets, subset_dict)
            subset_dict['live'] = live

            # Step 2.3: Calculate hallucination_live (live_irrelevance, irrelevance weighted average)
            hallucination_subsets = ['live_irrelevance', 'irrelevance']
            hallucination = unweighted_average_from_subsets(hallucination_subsets, subset_dict)
            subset_dict['hallucination'] = hallucination

            # Step 2.4: Calculate multi_turn (multi_turn_base, multi_turn_augmented unweighted average)
            multi_turn_subsets = [
                'multi_turn_base', 'multi_turn_miss_func', 'multi_turn_miss_param', 'multi_turn_long_context'
            ]
            multi_turn = unweighted_average_from_subsets(multi_turn_subsets, subset_dict)
            subset_dict['multi_turn'] = multi_turn

            # Step 2.5 Calculate web_search (web_search_base, web_search_no_snippet unweighted average)
            web_search_subsets = ['web_search_base', 'web_search_no_snippet']
            web_search = unweighted_average_from_subsets(web_search_subsets, subset_dict)
            subset_dict['web_search'] = web_search

            # Step 2.6 Calculate memory (memory_kv, memory_vector, memory_rec_sum unweighted average)
            memory_subsets = ['memory_kv', 'memory_vector', 'memory_rec_sum']
            memory = unweighted_average_from_subsets(memory_subsets, subset_dict)
            subset_dict['memory'] = memory

            # Step 2.7 Calculate agentic (web_search, memory unweighted average)
            agentic_subsets = ['web_search', 'memory']
            agentic = unweighted_average_from_subsets(agentic_subsets, subset_dict)
            subset_dict['agentic'] = agentic

            # Step 4: Calculate overall (non_live, live, multi_turn percentage weighted average)
            overall_subsets = ['agentic', 'multi_turn', 'non_live', 'live', 'hallucination']
            overall = percentage_weighted_average_from_subsets(
                overall_subsets, subset_dict, weights=[40, 30, 10, 10, 10]
            )
            subset_dict['overall'] = overall

            # Add computed scores to the category
            computed_subset_names = ['agentic', 'multi_turn', 'non_live', 'live', 'hallucination', 'overall']

            # Add the computed scores as new subsets in the metric
            dummy_subsets = []
            for subset_name in computed_subset_names:
                if subset_name in subset_dict and subset_dict[subset_name].num > 0:
                    subset = subset_dict[subset_name]
                    subset.name = subset_name.upper()
                    dummy_subsets.append(subset)
            dummy_category = Category(name='-', subsets=dummy_subsets)
            metric.categories.append(dummy_category)
