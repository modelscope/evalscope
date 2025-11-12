from __future__ import annotations

import traceback
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm
from typing import Any, Dict, List, Tuple

from evalscope.report import (
    Category,
    Report,
    Subset,
    percentage_weighted_average_from_subsets,
    unweighted_average_from_subsets,
    weighted_average_from_subsets,
)

# ----------------------------
# Public constants (extracted)
# ----------------------------

ALL_AVAILABLE_MEMORY_BACKENDS: List[str] = [
    'kv',
    'vector',
    'rec_sum',
]

NON_LIVE_CATEGORY: List[str] = [
    'simple_python',
    'simple_java',
    'simple_javascript',
    'multiple',
    'parallel',
    'parallel_multiple',
    'irrelevance',
]
LIVE_CATEGORY: List[str] = [
    'live_simple',
    'live_multiple',
    'live_parallel',
    'live_parallel_multiple',
    'live_irrelevance',
    'live_relevance',
]
MULTI_TURN_CATEGORY: List[str] = [
    'multi_turn_base',
    'multi_turn_miss_func',
    'multi_turn_miss_param',
    'multi_turn_long_context',
]
WEB_SEARCH_CATEGORY: List[str] = [
    'web_search_base',
    'web_search_no_snippet',
]

MEMORY_CATEGORY: List[str] = [f'memory_{backend}' for backend in ALL_AVAILABLE_MEMORY_BACKENDS]
MEMORY_SCENARIO_NAME = [
    'student',
    'customer',
    'finance',
    'healthcare',
    'notetaker',
]

SINGLE_TURN_CATEGORY: List[str] = NON_LIVE_CATEGORY + LIVE_CATEGORY
AGENTIC_CATEGORY: List[str] = MEMORY_CATEGORY + WEB_SEARCH_CATEGORY

ALL_SCORING_CATEGORIES: List[str] = SINGLE_TURN_CATEGORY + MULTI_TURN_CATEGORY + AGENTIC_CATEGORY

# Dummy models used only to infer underscore_to_dot behavior
DUMMY_MODEL_UNDERSCORE_TO_DOT = 'gpt-4o-2024-11-20-FC'
DUMMY_MODEL_NO_UNDERSCORE_TO_DOT = 'meta-llama/Llama-3.3-70B-Instruct-FC'

# ----------------------------
# Data preparation helpers
# ----------------------------


def load_bfcl_data(categories: List[str]) -> Tuple[Dict[str, List[Dict]], Dict[str, List[Dict]]]:
    """
    Load test entries and ground truth data from bfcl_eval for given categories.
    """
    from bfcl_eval.utils import is_relevance_or_irrelevance, load_dataset_entry, load_ground_truth_entry

    test_entries_by_cat: Dict[str, List[Dict]] = defaultdict(list)
    ground_truth_by_cat: Dict[str, List[Dict]] = defaultdict(list)

    for category in categories:
        test_entries_by_cat[category] = load_dataset_entry(
            category, include_prereq=True, include_language_specific_hint=False
        )
        if not is_relevance_or_irrelevance(category):
            ground_truth_by_cat[category] = load_ground_truth_entry(category)

    return test_entries_by_cat, ground_truth_by_cat


def prepare_ground_truth_map(category: str, ground_truth_entries: List[Dict]) -> Dict[str, Dict]:
    """
    Map ground truth entries to IDs with category-specific adjustments.
    """
    from bfcl_eval.utils import is_memory, is_web_search

    if not ground_truth_entries:
        return {}

    if is_memory(category):
        return {entry['id'].replace('memory', category): entry for entry in ground_truth_entries}
    if is_web_search(category):
        return {entry['id'].replace('web_search', category): entry for entry in ground_truth_entries}
    return {entry['id']: entry for entry in ground_truth_entries}


def process_test_entries(
    category: str,
    test_entries: List[Dict[str, Any]],
    ground_truth_entries: List[Dict[str, Any]],
    model_result_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Clean and enrich test entries, return processed entries and prereq entries.

    Returns:
        processed_entries: entries ready to be mapped to Samples
        prereq_entries: entries requiring prereq inference (memory snapshots)
    """
    from bfcl_eval.utils import (
        clean_up_memory_prereq_entries,
        is_memory_prereq,
        populate_initial_settings_for_memory_test_cases,
        populate_initial_settings_for_web_search_test_cases,
    )

    ground_truth_map = prepare_ground_truth_map(category, ground_truth_entries)

    test_entries = clean_up_memory_prereq_entries(test_entries)
    test_entries = populate_initial_settings_for_web_search_test_cases(test_entries)
    test_entries = populate_initial_settings_for_memory_test_cases(test_entries, model_result_dir=model_result_dir)

    prereq_entries = [entry for entry in test_entries if is_memory_prereq(entry['id'])]
    main_entries = [entry for entry in test_entries if not is_memory_prereq(entry['id'])]

    processed_entries: List[Dict[str, Any]] = []
    for entry in main_entries:
        entry_id = entry['id']
        entry['category'] = category
        entry['ground_truth'] = ground_truth_map.get(entry_id, {}).get('ground_truth', {})
        processed_entries.append(entry)

    return processed_entries, prereq_entries


def run_prereq_inference(
    handler: Any,
    prereq_entries: List[Dict[str, Any]],
    model_result_dir: Path,
    batch_size: int,
    logger: Any,
) -> None:
    """
    Run prerequisite inferences for memory snapshot creation if results are missing.
    Optimized to run different (backend, scenario) groups in parallel while preserving in-group order.
    """
    import re
    from bfcl_eval.utils import get_directory_structure_by_id
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not prereq_entries:
        return

    def _parse_backend_scenario_idx(entry_id: str) -> Tuple[str, str, int]:
        """
        Extract backend, scenario, and scenario index from an entry id.
        Expected format:
          memory_{backend}_prereq_{total_index}-{scenario}-{scenario_index}
        Returns ('unknown', 'unknown', 0) on failure.
        """
        backend = 'unknown'
        scenario = 'unknown'
        idx = 0

        m_backend = re.search(r'^memory_(?P<backend>.+?)_prereq_', entry_id)
        if m_backend:
            backend = m_backend.group('backend')

        m_tail = re.search(r'-(?P<scenario>[a-zA-Z_]+)-(?P<idx>\d+)$', entry_id)
        if m_tail:
            scenario = m_tail.group('scenario')
            idx = int(m_tail.group('idx'))

        return backend, scenario, idx

    # Group entries by (backend, scenario)
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for entry in prereq_entries:
        backend, scenario, idx = _parse_backend_scenario_idx(entry['id'])
        entry['_group_backend'] = backend
        entry['_group_scenario'] = scenario
        entry['_scenario_idx'] = idx
        groups.setdefault((backend, scenario), []).append(entry)

    # Sort entries within each group by scenario index to keep order
    for group_entries in groups.values():
        group_entries.sort(key=lambda e: e.get('_scenario_idx', 0))

    # Worker to process a single (backend, scenario) group sequentially
    def _process_group_entries(group_entries: List[Dict[str, Any]], progress: Any) -> None:
        for entry in group_entries:
            try:
                memory_snapshot_folder = (
                    model_result_dir / get_directory_structure_by_id(entry['id']) / 'memory_snapshot'
                    / 'prereq_checkpoints'
                )
                existing_filenames = {f.name for f in memory_snapshot_folder.rglob('*.json')}
                if (entry['id'] + '.json') in existing_filenames:
                    logger.info(f'Skipping prereq inference for entry ID {entry["id"]} as result already exists.')
                else:
                    handler.inference(deepcopy(entry), include_input_log=False, exclude_state_log=False)
            except Exception as e:
                logger.error(f'Error during prereq inference for entry ID {entry.get("id")}: {e}')
                logger.error(traceback.format_exc())
            finally:
                # tqdm is thread-safe; each worker updates shared progress bar
                progress.update(1)

    # Run each (backend, scenario) group in parallel; preserve in-group order
    total = len(prereq_entries)
    with tqdm(total=total, desc='Running prereq inferences for memory snapshots...') as progress:
        max_workers = min(batch_size, len(groups))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_process_group_entries, group_entries, progress) for group_entries in groups.values()
            ]
            for _ in as_completed(futures):
                # Errors are logged within workers
                pass

    # Cleanup temp keys
    for group_entries in groups.values():
        for entry in group_entries:
            entry.pop('_group_backend', None)
            entry.pop('_group_scenario', None)
            entry.pop('_scenario_idx', None)


# ----------------------------
# Scoring helpers
# ----------------------------


def compute_entry_result(
    handler: Any,
    model_result: Any,
    prompt_entry: Dict[str, Any],
    underscore_to_dot: bool,
) -> Dict[str, Any]:
    """
    Compute evaluation result for a single entry across BFCL categories.
    """
    from bfcl_eval.constants.enums import Language, ReturnFormat
    from bfcl_eval.eval_checker.eval_runner import (
        _evaluate_single_agentic_entry,
        _evaluate_single_ast_entry,
        _evaluate_single_multi_turn_entry,
        _evaluate_single_relevance_entry,
    )
    from bfcl_eval.utils import is_agentic, is_java, is_js, is_multi_turn, is_relevance_or_irrelevance

    test_category = prompt_entry['category']
    index = prompt_entry['id']
    ground_truth = prompt_entry.get('ground_truth', {})

    model_name = (DUMMY_MODEL_UNDERSCORE_TO_DOT if underscore_to_dot else DUMMY_MODEL_NO_UNDERSCORE_TO_DOT)

    if is_relevance_or_irrelevance(test_category):
        return _evaluate_single_relevance_entry(
            handler=handler,
            index=index,
            model_result_item=model_result,
            prompt_entry=prompt_entry,
            model_name=model_name,
            test_category=test_category,
        )

    elif is_multi_turn(test_category):
        return _evaluate_single_multi_turn_entry(
            handler=handler,
            test_entry_id=index,
            model_result_list=model_result,
            ground_truth_list=ground_truth,
            prompt_entry=prompt_entry,
            model_name=model_name,
            test_category=test_category,
        )

    elif is_agentic(test_category):
        return _evaluate_single_agentic_entry(
            handler=handler,
            index=index,
            model_result_list=model_result,
            possible_answer_item=ground_truth,
            prompt_entry=prompt_entry,
            model_name=model_name,
            test_category=test_category,
        )
    else:
        # AST categories (python/java/js)
        if is_java(test_category):
            language = Language.JAVA
            return_format = ReturnFormat.JAVA
        elif is_js(test_category):
            language = Language.JAVASCRIPT
            return_format = ReturnFormat.JAVASCRIPT
        else:
            language = Language.PYTHON
            return_format = ReturnFormat.PYTHON

        return _evaluate_single_ast_entry(
            handler=handler,
            index=index,
            model_result_item=model_result,
            possible_answer_item=ground_truth,
            prompt_entry=prompt_entry,
            model_name=model_name,
            test_category=test_category,
            language=language,
            return_format=return_format,
        )


# ----------------------------
# Report aggregation helpers
# ----------------------------


def compute_aggregate_subsets(report: Report) -> None:
    """
    Compute aggregated subsets and overall score for BFCL report.
    Modifies the report in-place.
    """
    for metric in report.metrics:
        # Collect all subsets in a dictionary for easy access
        subset_dict: Dict[str, Subset] = {}
        for category in metric.categories:
            for subset in category.subsets:
                subset_dict[subset.name] = subset

        # Step 1: simple_ast
        simple_subsets = ['simple_python', 'simple_java', 'simple_javascript']
        simple_ast = unweighted_average_from_subsets(simple_subsets, subset_dict)
        subset_dict['simple_ast'] = simple_ast

        # Step 2.1: non_live (simple_ast, multiple, parallel, parallel_multiple)
        non_live_subsets = ['simple_ast', 'multiple', 'parallel', 'parallel_multiple']
        non_live = unweighted_average_from_subsets(non_live_subsets, subset_dict)
        subset_dict['non_live'] = non_live

        # Step 2.2: live (weighted)
        live_subsets = ['live_simple', 'live_multiple', 'live_parallel', 'live_parallel_multiple']
        live = weighted_average_from_subsets(live_subsets, subset_dict)
        subset_dict['live'] = live

        # Step 2.3: hallucination (unweighted)
        hallucination_subsets = ['live_irrelevance', 'irrelevance']
        hallucination = unweighted_average_from_subsets(hallucination_subsets, subset_dict)
        subset_dict['hallucination'] = hallucination

        # Step 2.4: multi_turn (unweighted)
        multi_turn_subsets = [
            'multi_turn_base',
            'multi_turn_miss_func',
            'multi_turn_miss_param',
            'multi_turn_long_context',
        ]
        multi_turn = unweighted_average_from_subsets(multi_turn_subsets, subset_dict)
        subset_dict['multi_turn'] = multi_turn

        # Step 2.5: web_search (unweighted)
        web_search_subsets = ['web_search_base', 'web_search_no_snippet']
        web_search = unweighted_average_from_subsets(web_search_subsets, subset_dict)
        subset_dict['web_search'] = web_search

        # Step 2.6: memory (unweighted)
        memory_subsets = ['memory_kv', 'memory_vector', 'memory_rec_sum']
        memory = unweighted_average_from_subsets(memory_subsets, subset_dict)
        subset_dict['memory'] = memory

        # Step 2.7: agentic (unweighted)
        agentic_subsets = ['web_search', 'memory']
        agentic = unweighted_average_from_subsets(agentic_subsets, subset_dict)
        subset_dict['agentic'] = agentic

        # Step 4: overall (percentage weighted average)
        overall_subsets = ['agentic', 'multi_turn', 'non_live', 'live', 'hallucination']
        overall = percentage_weighted_average_from_subsets(overall_subsets, subset_dict, weights=[40, 30, 10, 10, 10])
        subset_dict['overall'] = overall

        # Add computed scores to the category
        computed_subset_names = ['agentic', 'multi_turn', 'non_live', 'live', 'hallucination', 'overall']

        # Add the computed scores as new subsets in the metric
        dummy_subsets: List[Subset] = []
        for subset_name in computed_subset_names:
            if subset_name in subset_dict and subset_dict[subset_name].num > 0:
                subset = subset_dict[subset_name]
                subset.name = subset_name.upper()
                dummy_subsets.append(subset)
        dummy_category = Category(name='-', subsets=dummy_subsets)
        metric.categories.append(dummy_category)
