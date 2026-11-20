from __future__ import annotations

import json
import re
import uuid
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import copy, deepcopy
from pathlib import Path
from sys import platform
from typing import Any, Dict, Iterator, List, Tuple

from tqdm import tqdm

from evalscope.api.messages.chat_message import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)
from evalscope.api.tool import ToolCall, ToolFunction
from evalscope.report import Category, Report, Subset

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
# Handler compatibility helpers
# ----------------------------


@contextmanager
def _isolated_execution(entry: Dict[str, Any], *, evaluation: bool) -> Iterator[str]:
    """Keep BFCL's module-global simulators local to one rollout or scoring call."""
    from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_utils

    namespace = f'evalscope_{uuid.uuid4().hex}'
    model_names = (f'{namespace}_eval', f'{namespace}_ground_truth_eval') if evaluation else (namespace,)
    # The pinned harness caches all involved classes, including stateless ones,
    # under these exact names. Never clear another invocation's live instances.
    keys = [
        re.sub(r'[-./:]', '_', f'{model_name}_{entry["id"]}_{class_name}_instance')
        for model_name in model_names
        for class_name in entry.get('involved_classes', [])
    ]
    try:
        yield namespace
    finally:
        for key in keys:
            vars(multi_turn_utils).pop(key, None)


@contextmanager
def _limit_vector_search_threads() -> Iterator[None]:
    # Load the encoder before FAISS, as required by the upstream memory module.
    from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.memory_vector import faiss

    # The PyTorch and FAISS wheels on macOS can load separate OpenMP runtimes.
    # Keep FAISS single-threaded for this rollout without limiting the API call pool.
    previous = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(1)
    try:
        yield
    finally:
        faiss.omp_set_num_threads(previous)


def run_isolated_inference(
    handler: Any,
    entry: Dict[str, Any],
    *,
    include_input_log: bool,
    exclude_state_log: bool,
) -> Any:
    """Run BFCL inference with independent simulator state and original checkpoint IDs."""
    from bfcl_eval.utils import contain_multi_turn_interaction

    entry = deepcopy(entry)
    if not contain_multi_turn_interaction(entry['id']):
        return handler.inference(entry, include_input_log=include_input_log, exclude_state_log=exclude_state_log)

    # Only the backend instance name changes. The API model/client, registry and
    # memory snapshot paths retain their original identity on this shallow copy.
    isolated_handler = copy(handler)
    vector_search = (
        _limit_vector_search_threads()
        if (platform == 'darwin' and entry['id'].startswith('memory_vector_'))
        else nullcontext()
    )
    with vector_search, _isolated_execution(entry, evaluation=False) as namespace:
        isolated_handler.model_name_underline_replaced = namespace
        return isolated_handler.inference(
            entry, include_input_log=include_input_log, exclude_state_log=exclude_state_log
        )


def patch_openai_completions_handler_empty_tool_calls(handler_cls: type[Any]) -> None:
    """Work around a bfcl_eval bug where a text-only assistant turn loses its content.

    ``OpenAICompletionsHandler._parse_query_response_FC`` only falls back to
    ``message.content`` when iterating ``message.tool_calls`` *raises* (i.e. when it
    is ``None``). Many OpenAI-compatible servers (e.g. vLLM, SGLang) return
    ``tool_calls: []`` instead of omitting/nulling it for text-only completions, so the
    list comprehension over an empty list succeeds and silently produces
    ``model_responses == []``, discarding the model's real final answer. This has been
    reported upstream (ShishirPatil/gorilla#1316, #1351) and still occurs in
    bfcl-eval 2026.3.23.
    """

    def _parse_query_response_FC(self, api_response):
        message = api_response.choices[0].message
        tool_calls = message.tool_calls or []
        if tool_calls:
            model_responses = [{call.function.name: call.function.arguments} for call in tool_calls]
            tool_call_ids = [call.id for call in tool_calls]
        else:
            model_responses = message.content or ''
            tool_call_ids = []
        return {
            'model_responses': model_responses,
            'model_responses_message_for_chat_history': message,
            'tool_call_ids': tool_call_ids,
            'input_token': api_response.usage.prompt_tokens,
            'output_token': api_response.usage.completion_tokens,
        }

    handler_cls._parse_query_response_FC = _parse_query_response_FC


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
) -> set[Tuple[str, str]]:
    """
    Run prerequisite inferences for memory snapshot creation if results are missing.
    Optimized to run different (backend, scenario) groups in parallel while preserving in-group order.
    Return groups whose prerequisite chain could not complete.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from bfcl_eval.utils import get_directory_structure_by_id

    if not prereq_entries:
        return set()

    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for entry in prereq_entries:
        prefix, scenario, _ = entry['id'].rsplit('-', 2)
        backend = prefix.removeprefix('memory_').split('_prereq_', 1)[0]
        groups[backend, scenario].append(entry)

    def _process_group(backend: str, scenario: str, entries: List[Dict[str, Any]], progress: Any) -> bool:
        entries.sort(key=lambda entry: int(entry['id'].rsplit('-', 1)[1]))
        folder = model_result_dir / get_directory_structure_by_id(entries[0]['id']) / 'memory_snapshot'
        snapshot_path = folder / f'{scenario}_final.json'
        checkpoints = folder / 'prereq_checkpoints'
        resume = _memory_resume_index(snapshot_path, checkpoints, backend, entries)
        progress.update(resume)
        for entry in entries[resume:]:
            try:
                run_isolated_inference(handler, entry, include_input_log=False, exclude_state_log=False)
            except Exception:
                logger.exception(f'BFCL memory prerequisite {entry["id"]} failed. Later entries depend on this step.')
                return False
            finally:
                progress.update(1)

        try:
            has_content = _memory_snapshot_has_content(snapshot_path.read_bytes(), backend)
        except FileNotFoundError:
            logger.warning(f'BFCL memory {backend}/{scenario} has no final snapshot at {snapshot_path}.')
            return False
        except (OSError, ValueError) as exc:
            logger.warning(f'BFCL memory {backend}/{scenario} has an unreadable snapshot at {snapshot_path}: {exc}')
            return False
        if not has_content:
            logger.warning(
                f'BFCL memory {backend}/{scenario} has an empty snapshot at {snapshot_path}. '
                'Check whether the model called the memory write tools and whether those calls succeeded.'
            )
        return True

    # Run each (backend, scenario) group in parallel; preserve in-group order
    total = len(prereq_entries)
    with tqdm(total=total, desc='Running prereq inferences for memory snapshots...') as progress:
        max_workers = min(batch_size, len(groups))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_process_group, backend, scenario, entries, progress): (backend, scenario)
                for (backend, scenario), entries in groups.items()
            }
            return {futures[future] for future in as_completed(futures) if not future.result()}


def _memory_snapshot_has_content(data: bytes, backend: str) -> bool:
    snapshot = json.loads(data)
    if not isinstance(snapshot, dict):
        raise ValueError('snapshot must be a JSON object')
    if backend == 'rec_sum':
        memory = snapshot.get('memory')
        if not isinstance(memory, str):
            raise ValueError('memory must be a string')
        return bool(memory.strip())

    stores = [snapshot.get(key) for key in ('core_memory', 'archival_memory')]
    if backend == 'vector':
        if not all(isinstance(store, dict) and isinstance(store.get('next_id'), int) for store in stores):
            raise ValueError('vector memory must contain its next entry ID')
        stores = [store.get('store') for store in stores]
    if not all(isinstance(store, dict) for store in stores):
        raise ValueError('core and archival memory stores must be JSON objects')
    return any(stores)


def _memory_resume_index(snapshot_path: Path, checkpoints: Path, backend: str, entries: List[Dict[str, Any]]) -> int:
    """Resume only from a checkpoint matching the final snapshot used by BFCL."""
    resume = 0
    try:
        snapshot = snapshot_path.read_bytes()
        _memory_snapshot_has_content(snapshot, backend)
        for index, entry in enumerate(entries, start=1):
            if (checkpoints / f'{entry["id"]}.json').read_bytes() == snapshot:
                resume = index
    except (OSError, ValueError):
        # A gap ends the usable prefix. A bad final snapshot requires a fresh start.
        pass
    return resume


# ----------------------------
# Visualization helpers
# ----------------------------


def synthesize_agent_messages(
    prompt_entry: Dict[str, Any],
    model_result: Any,
    is_fc_model: bool,
    inference_log: Any = None,
) -> List[ChatMessage]:
    """Synthesize a ChatMessage list for review/visualization from a BFCL entry.

    BFCL's ``handler.inference`` returns ``model_result`` (per-turn / per-step
    response payloads) but does NOT expose the underlying chat history. This
    helper reconstructs an approximate conversation from:

    * ``prompt_entry['question']`` — list of turns, each a list of role/content
      dicts (the genuine user/system messages sent into the handler);
    * ``model_result`` — list of turns, each a list of step responses. In FC
      mode each step response is a list of ``{name: args}`` dicts; in
      prompting mode it is a string.
    * ``inference_log`` (optional) — multi-turn inference log used to recover
      per-step ``tool`` role messages (function execution results). Single-turn
      AST categories don't execute tools, so passing ``None`` is fine.

    Reasoning content is intentionally not recovered.
    """

    def _as_str(content: Any) -> str:
        if isinstance(content, str):
            return content
        try:
            return json.dumps(content, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(content)

    def _user_or_system(entry: Dict[str, Any]) -> ChatMessage:
        role = entry.get('role')
        content = _as_str(entry.get('content', ''))
        if role == 'system':
            return ChatMessageSystem(content=content)
        return ChatMessageUser(content=content)

    def _build_tool_calls(step_response: Any) -> List[ToolCall]:
        if not isinstance(step_response, list):
            return []
        tool_calls: List[ToolCall] = []
        for item in step_response:
            if not isinstance(item, dict) or len(item) != 1:
                return []
            name, args = next(iter(item.items()))
            if isinstance(args, str):
                try:
                    args_dict = json.loads(args)
                except (TypeError, ValueError):
                    args_dict = {'_raw': args}
            elif isinstance(args, dict):
                args_dict = args
            else:
                return []
            tool_calls.append(
                ToolCall(
                    id=f'call_{uuid.uuid4().hex[:16]}',
                    function=ToolFunction(name=name, arguments=args_dict),
                )
            )
        return tool_calls

    def _assistant_from_step(step_response: Any) -> ChatMessageAssistant:
        if is_fc_model:
            tool_calls = _build_tool_calls(step_response)
            if tool_calls:
                return ChatMessageAssistant(content='', tool_calls=tool_calls)
        return ChatMessageAssistant(content=_as_str(step_response))

    messages: List[ChatMessage] = []
    questions: List[List[Dict[str, Any]]] = prompt_entry.get('question', []) or []
    turns: List[Any] = model_result if isinstance(model_result, list) else []
    # Inference_log mixes per-turn dicts and bare state-log lists; keep only
    # the per-turn dicts so they align positionally with ``turns``.
    turn_logs: List[Dict[str, Any]] = (
        [t for t in inference_log if isinstance(t, dict)] if isinstance(inference_log, list) else []
    )
    n_turns = max(len(questions), len(turns))

    for i in range(n_turns):
        if i < len(questions):
            for q in questions[i] or []:
                if isinstance(q, dict):
                    messages.append(_user_or_system(q))
        if i >= len(turns):
            continue
        turn_steps = turns[i] if isinstance(turns[i], list) else [turns[i]]
        turn_log = turn_logs[i] if i < len(turn_logs) else {}
        for step_idx, step in enumerate(turn_steps):
            asst = _assistant_from_step(step)
            messages.append(asst)
            step_entries = turn_log.get(f'step_{step_idx}', []) if isinstance(turn_log, dict) else []
            if not isinstance(step_entries, list):
                continue
            tool_entries = [e for e in step_entries if isinstance(e, dict) and e.get('role') == 'tool']
            tool_calls = asst.tool_calls or []
            for j, entry in enumerate(tool_entries):
                tc = tool_calls[j] if j < len(tool_calls) else None
                messages.append(
                    ChatMessageTool(
                        content=_as_str(entry.get('content', '')),
                        tool_call_id=tc.id if tc is not None else None,
                        function=tc.function.name if tc is not None else None,
                    )
                )
    return messages


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

    model_name = DUMMY_MODEL_UNDERSCORE_TO_DOT if underscore_to_dot else DUMMY_MODEL_NO_UNDERSCORE_TO_DOT

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
        with _isolated_execution(prompt_entry, evaluation=True) as namespace:
            result = _evaluate_single_multi_turn_entry(
                handler=handler,
                test_entry_id=index,
                model_result_list=model_result,
                ground_truth_list=ground_truth,
                prompt_entry=deepcopy(prompt_entry),
                model_name=namespace,
                test_category=test_category,
            )
        # Keep diagnostics stable; the private simulator identity is not a model.
        if 'model_name' in result:
            result['model_name'] = model_name
        return result

    elif is_agentic(test_category):
        return _evaluate_single_agentic_entry(
            handler=handler,
            index=index,
            model_result_list=model_result,
            possible_answer_item=ground_truth,
            prompt_entry=prompt_entry.copy(),
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
    """Aggregate BFCL scores with the official treatment of unevaluated categories."""
    from bfcl_eval.eval_checker.eval_runner_helper import (
        calculate_percentage_weighted_accuracy,
        calculate_unweighted_accuracy,
        calculate_weighted_accuracy,
        get_category_score,
    )

    for metric in report.metrics:
        subset_dict: Dict[str, Subset] = {}
        for category in metric.categories:
            for subset in category.subsets:
                subset_dict[subset.name] = subset

        official_scores = {
            subset.name: {
                'accuracy': subset.score,
                'total_count': subset.num,
                'display_accuracy': subset.score,
            }
            for subset in subset_dict.values()
            if subset.num > 0
        }

        def category_score(name: str) -> dict:
            if name in official_scores:
                return official_scores[name]
            return get_category_score(official_scores, name)

        def add_aggregate(name: str, sources: list[str], result: dict) -> None:
            official_scores[name] = result
            # Official counts include unevaluated entries; Report.num remains the actual evaluated count.
            evaluated_num = sum(subset_dict[source].num for source in sources if source in subset_dict)
            subset_dict[name] = Subset(name=name, score=result['accuracy'], num=evaluated_num)

        simple_subsets = ['simple_python', 'simple_java', 'simple_javascript']
        simple_ast = calculate_unweighted_accuracy([category_score(name) for name in simple_subsets])
        add_aggregate('simple_ast', simple_subsets, simple_ast)

        non_live_subsets = ['simple_ast', 'multiple', 'parallel', 'parallel_multiple']
        non_live = calculate_unweighted_accuracy([category_score(name) for name in non_live_subsets])
        add_aggregate('non_live', non_live_subsets, non_live)

        live_subsets = ['live_simple', 'live_multiple', 'live_parallel', 'live_parallel_multiple']
        live = calculate_weighted_accuracy([category_score(name) for name in live_subsets])
        add_aggregate('live', live_subsets, live)

        hallucination_subsets = ['live_irrelevance', 'irrelevance']
        hallucination = calculate_unweighted_accuracy([category_score(name) for name in hallucination_subsets])
        add_aggregate('hallucination', hallucination_subsets, hallucination)

        multi_turn_subsets = [
            'multi_turn_base',
            'multi_turn_miss_func',
            'multi_turn_miss_param',
            'multi_turn_long_context',
        ]
        multi_turn = calculate_unweighted_accuracy([category_score(name) for name in multi_turn_subsets])
        add_aggregate('multi_turn', multi_turn_subsets, multi_turn)

        web_search_subsets = ['web_search_base', 'web_search_no_snippet']
        web_search = calculate_unweighted_accuracy([category_score(name) for name in web_search_subsets])
        add_aggregate('web_search', web_search_subsets, web_search)

        memory_subsets = ['memory_kv', 'memory_vector', 'memory_rec_sum']
        memory = calculate_unweighted_accuracy([category_score(name) for name in memory_subsets])
        add_aggregate('memory', memory_subsets, memory)

        agentic_subsets = ['web_search', 'memory']
        agentic = calculate_unweighted_accuracy([category_score(name) for name in agentic_subsets])
        add_aggregate('agentic', agentic_subsets, agentic)

        overall_subsets = ['agentic', 'multi_turn', 'non_live', 'live', 'hallucination']
        overall = calculate_percentage_weighted_accuracy(
            [category_score(name) for name in overall_subsets],
            [40, 30, 10, 10, 10],
            display_na_if_category_missing=False,
        )
        add_aggregate('overall', overall_subsets, overall)

        computed_subset_names = ['agentic', 'multi_turn', 'non_live', 'live', 'hallucination', 'overall']
        dummy_subsets: List[Subset] = []
        for subset_name in computed_subset_names:
            if subset_name in subset_dict and subset_dict[subset_name].num > 0:
                subset = subset_dict[subset_name]
                subset.name = subset_name.upper()
                subset.is_aggregate = True
                dummy_subsets.append(subset)
        dummy_category = Category(name='-', subsets=dummy_subsets)
        metric.categories.append(dummy_category)
