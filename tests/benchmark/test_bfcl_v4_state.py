"""Regression tests using BFCL's pinned executable multi-turn checker."""

import ast
import importlib
import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterator

import pytest

pytest.importorskip('bfcl_eval')

from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_utils
from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler
from bfcl_eval.utils import load_dataset_entry, load_ground_truth_entry

from evalscope.benchmarks.bfcl.v4 import utils as bfcl_utils
from evalscope.benchmarks.bfcl.v4.utils import (
    DUMMY_MODEL_NO_UNDERSCORE_TO_DOT,
    DUMMY_MODEL_UNDERSCORE_TO_DOT,
    compute_entry_result,
)


@pytest.fixture
def official_entry() -> Iterator[dict[str, Any]]:
    entry = next(
        item for item in load_dataset_entry('multi_turn_base', include_language_specific_hint=False)
        if item['id'] == 'multi_turn_base_0'
    )
    entry['category'] = 'multi_turn_base'
    entry['ground_truth'] = next(
        item['ground_truth'] for item in load_ground_truth_entry('multi_turn_base') if item['id'] == entry['id']
    )
    # Clean only this fixture's legacy keys when exercising the pre-fix code.
    keys = [
        re.sub(r'[-./]', '_', f'{model}{suffix}_{entry["id"]}_{class_name}_instance')
        for model in (DUMMY_MODEL_UNDERSCORE_TO_DOT, DUMMY_MODEL_NO_UNDERSCORE_TO_DOT)
        for suffix in ('_eval', '_ground_truth_eval')
        for class_name in entry['involved_classes']
    ]
    saved = {key: vars(multi_turn_utils)[key] for key in keys if key in vars(multi_turn_utils)}
    for key in keys:
        vars(multi_turn_utils).pop(key, None)
    yield entry
    for key in keys:
        vars(multi_turn_utils).pop(key, None)
    vars(multi_turn_utils).update(saved)


def _handler(fc: bool) -> OpenAICompletionsHandler:
    handler = object.__new__(OpenAICompletionsHandler)
    handler.is_fc_model = fc
    return handler


def _correct_prediction(entry: dict[str, Any], fc: bool) -> list[list[Any]]:
    calls = deepcopy(entry['ground_truth'])
    # BFCL's prompting decoder requires named arguments; the official reference
    # uses the semantically equivalent positional form for this one call.
    assert calls[2] == ["sort('final_report.pdf')"]
    calls[2] = ["sort(file_name='final_report.pdf')"]
    if not fc:
        return [['[' + ', '.join(turn) + ']'] for turn in calls]

    result = []
    for turn in calls:
        functions = []
        for text in turn:
            call = ast.parse(text, mode='eval').body
            assert isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and not call.args
            arguments = {item.arg: ast.literal_eval(item.value) for item in call.keywords}
            functions.append({call.func.id: json.dumps(arguments)})
        result.append([functions])
    return result


@pytest.mark.parametrize('fc', [False, True], ids=['prompt', 'fc'])
@pytest.mark.parametrize('underscore_to_dot', [False, True])
def test_bad_prediction_cannot_change_a_later_correct_verdict(
    official_entry: dict[str, Any], fc: bool, underscore_to_dot: bool,
) -> None:
    handler = _handler(fc)
    prediction = _correct_prediction(official_entry, fc)
    bad = deepcopy(prediction)
    bad[0] = [[]] if fc else ['[]']

    before = set(vars(multi_turn_utils))
    failed = compute_entry_result(handler, bad, deepcopy(official_entry), underscore_to_dot)
    assert failed['valid'] is False
    assert failed['error']['error_type'] == 'multi_turn:empty_turn_model_response'
    expected_model = (
        DUMMY_MODEL_UNDERSCORE_TO_DOT if underscore_to_dot else DUMMY_MODEL_NO_UNDERSCORE_TO_DOT
    )
    assert failed['model_name'] == expected_model

    for _ in range(2):
        actual = compute_entry_result(handler, deepcopy(prediction), deepcopy(official_entry), underscore_to_dot)
        assert actual['valid'] is True, actual
    assert set(vars(multi_turn_utils)) == before


def test_scoring_preserves_the_original_prompt(official_entry: dict[str, Any]) -> None:
    original = deepcopy(official_entry)
    result = compute_entry_result(_handler(True), _correct_prediction(official_entry, True), official_entry, True)
    assert result['valid'] is True
    assert official_entry == original


@pytest.mark.parametrize('underscore_to_dot', [False, True])
def test_ast_scoring_still_uses_the_registered_dummy_model(underscore_to_dot: bool) -> None:
    entry = next(
        item for item in load_dataset_entry('simple_python', include_language_specific_hint=False)
        if item['id'] == 'simple_python_1'
    )
    entry['category'] = 'simple_python'
    entry['ground_truth'] = [{'math.factorial': {'number': [5]}}]
    function_name = 'math_factorial' if underscore_to_dot else 'math.factorial'
    result = compute_entry_result(_handler(True), [{function_name: '{"number": 5}'}], entry, underscore_to_dot)
    assert result['valid'] is True, result


def test_partial_simulation_initialization_is_cleaned_after_exception(official_entry: dict[str, Any]) -> None:
    entry = deepcopy(official_entry)
    entry['involved_classes'] = ['MathAPI', *entry['involved_classes'], 'NotAnOfficialAPI']
    before = set(vars(multi_turn_utils))
    sentinel = object()
    key = 'another_active_evaluation_instance'
    vars(multi_turn_utils)[key] = sentinel
    try:
        with pytest.raises(KeyError, match='NotAnOfficialAPI'):
            compute_entry_result(_handler(True), _correct_prediction(official_entry, True), entry, True)
        assert vars(multi_turn_utils)[key] is sentinel
        assert set(vars(multi_turn_utils)) == before | {key}
    finally:
        vars(multi_turn_utils).pop(key, None)


@pytest.mark.parametrize('fc', [False, True], ids=['prompt', 'fc'])
def test_concurrent_grading_keeps_another_same_id_execution_alive(
    official_entry: dict[str, Any], monkeypatch: pytest.MonkeyPatch, fc: bool,
) -> None:
    checker = importlib.import_module('bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker')
    execute = checker.execute_multi_turn_func_call
    initialized = threading.Event()
    release = threading.Event()
    local = threading.local()

    def held_execution(*args: Any, **kwargs: Any) -> Any:
        result = execute(*args, **kwargs)
        if threading.current_thread().name.startswith('held-bfcl') and not getattr(local, 'held', False):
            local.held = True
            initialized.set()
            assert release.wait(timeout=15), 'The concurrent scorer did not release the held execution.'
        return result

    monkeypatch.setattr(checker, 'execute_multi_turn_func_call', held_execution)
    prediction = _correct_prediction(official_entry, fc)
    before = set(vars(multi_turn_utils))

    def score() -> dict[str, Any]:
        return compute_entry_result(_handler(fc), deepcopy(prediction), deepcopy(official_entry), True)

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='held-bfcl') as executor:
        pending = executor.submit(score)
        try:
            assert initialized.wait(timeout=15)
            completed = score()
            assert completed['valid'] is True, completed
        finally:
            release.set()
        result = pending.result(timeout=15)
        assert result['valid'] is True, result
    assert set(vars(multi_turn_utils)) == before


class _ExecutableHandler:
    """Exercise the real simulated filesystem across two steps without model I/O."""

    def __init__(self) -> None:
        self.model_name = 'real/model-name'
        self.model_name_underline_replaced = 'real_model_name'
        self.registry_name = 'original-registry'
        self.client = object()

    def inference(self, entry: dict[str, Any], include_input_log: bool, exclude_state_log: bool) -> Any:
        results = []
        instances = []
        for calls in (["cd(folder='document')"], ["mkdir(dir_name='isolated')"]):
            output, current = multi_turn_utils.execute_multi_turn_func_call(
                calls, entry['initial_config'], entry['involved_classes'],
                self.model_name_underline_replaced, entry['id'],
            )
            results.extend(output)
            instances.append(current['GorillaFileSystem'])
        assert instances[0] is instances[1], 'State must persist between steps of this invocation.'
        entry['question'].clear()
        return results, {'client': self.client, 'model': self.model_name, 'registry': self.registry_name}


def test_inference_preserves_within_run_state_without_mutating_shared_handler_or_entry(
    official_entry: dict[str, Any],
) -> None:
    handler = _ExecutableHandler()
    original_handler = vars(handler).copy()
    original_entry = deepcopy(official_entry)
    before = set(vars(multi_turn_utils))
    results = [
        bfcl_utils.run_isolated_inference(
            handler, official_entry, include_input_log=False, exclude_state_log=True,
        )
        for _ in range(2)
    ]
    assert results[0][0] == results[1][0]
    assert all('error' not in item.lower() for result, _ in results for item in result)
    for _, metadata in results:
        assert metadata == {'client': handler.client, 'model': handler.model_name, 'registry': handler.registry_name}
    assert vars(handler) == original_handler
    assert official_entry == original_entry
    assert set(vars(multi_turn_utils)) == before


def test_inference_exception_releases_its_simulated_instances(
    official_entry: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = _ExecutableHandler()
    original = _ExecutableHandler.inference

    def failing_inference(
        self: _ExecutableHandler, entry: dict[str, Any], include_input_log: bool, exclude_state_log: bool,
    ) -> Any:
        original(self, entry, include_input_log, exclude_state_log)
        raise RuntimeError('scripted transport failure')

    monkeypatch.setattr(_ExecutableHandler, 'inference', failing_inference)
    before = set(vars(multi_turn_utils))
    with pytest.raises(RuntimeError, match='scripted transport failure'):
        bfcl_utils.run_isolated_inference(handler, official_entry, include_input_log=False, exclude_state_log=True)
    assert set(vars(multi_turn_utils)) == before


class _MemorySnapshotHandler(_ExecutableHandler):

    def inference(self, entry: dict[str, Any], include_input_log: bool, exclude_state_log: bool) -> Any:
        assert include_input_log is False and exclude_state_log is False
        _, instances = multi_turn_utils.execute_multi_turn_func_call(
            [], entry['initial_config'], entry['involved_classes'], self.model_name_underline_replaced, entry['id'],
        )
        memory = instances['MemoryAPI_kv']
        if entry['id'].endswith('-1'):
            assert memory.core_memory == {'first_checkpoint': 'persisted'}
        key = 'first_checkpoint' if entry['id'].endswith('-0') else 'second_checkpoint'
        memory.core_memory_add(key, 'persisted')
        memory._flush_memory_to_local_file()
        return [], {}


def test_prerequisite_cleanup_preserves_ordered_memory_snapshots(tmp_path: Path) -> None:
    handler = _MemorySnapshotHandler()
    entries = []
    for index in range(2):
        entry_id = f'memory_kv_prereq_{index}-student-{index}'
        entries.append({
            'id': entry_id,
            'involved_classes': ['MemoryAPI_kv'],
            'initial_config': {
                'MemoryAPI_kv': {'model_result_dir': tmp_path, 'test_id': entry_id, 'scenario': 'student'},
            },
        })
    before = set(vars(multi_turn_utils))
    bfcl_utils.run_prereq_inference(handler, entries, tmp_path, 2, logging.getLogger(__name__))
    checkpoints = {path.name: json.loads(path.read_text()) for path in tmp_path.rglob('*.json')}
    for entry in entries:
        assert entry['id'] + '.json' in checkpoints
        assert '_scenario_idx' not in entry
    assert checkpoints[entries[1]['id'] + '.json']['core_memory'] == {
        'first_checkpoint': 'persisted', 'second_checkpoint': 'persisted',
    }
    assert 'student_final.json' in checkpoints
    assert set(vars(multi_turn_utils)) == before
    assert handler.model_name_underline_replaced == 'real_model_name'
