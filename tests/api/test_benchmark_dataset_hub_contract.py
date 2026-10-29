"""Static contracts for hub-aware benchmark dataset loading."""

import ast
import re
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).parents[2] / 'evalscope' / 'benchmarks'
DIRECT_MODELSCOPE_APIS = {'MsDataset', 'dataset_file_download', 'dataset_snapshot_download'}
REVISION_AWARE_CALLS = {'DatasetHub', 'download_dataset_file', 'download_dataset_snapshot'}


def _benchmark_sources():
    for path in BENCHMARK_ROOT.rglob('*.py'):
        yield path, ast.parse(path.read_text(encoding='utf-8'))


def _call_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return ''


def test_benchmarks_do_not_bypass_shared_hub_with_modelscope_dataset_apis() -> None:
    violations = []
    for path, tree in _benchmark_sources():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == 'modelscope':
                imported = DIRECT_MODELSCOPE_APIS & {alias.name for alias in node.names}
                if imported:
                    violations.append(f'{path.relative_to(BENCHMARK_ROOT)}:{node.lineno}: {sorted(imported)}')
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if re.search(r'modelscope\.cn/datasets/.+/resolve/', node.value):
                    violations.append(f'{path.relative_to(BENCHMARK_ROOT)}:{node.lineno}: hard-coded resolve URL')

    assert not violations, 'Benchmark data must load through evalscope.api.dataset:\n' + '\n'.join(violations)


def test_custom_dataset_hub_calls_propagate_revision() -> None:
    violations = []
    for path, tree in _benchmark_sources():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _call_name(node) not in REVISION_AWARE_CALLS:
                continue
            keyword_names = {keyword.arg for keyword in node.keywords}
            if 'revision' not in keyword_names:
                violations.append(f'{path.relative_to(BENCHMARK_ROOT)}:{node.lineno}: {_call_name(node)}')

    assert not violations, 'Custom dataset downloads must propagate the resolved revision:\n' + '\n'.join(violations)


def test_custom_remote_data_loaders_propagate_hub_and_revision() -> None:
    violations = []
    for path, tree in _benchmark_sources():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _call_name(node) != 'RemoteDataLoader':
                continue
            keyword_names = {keyword.arg for keyword in node.keywords}
            missing = {'data_source', 'version'} - keyword_names
            if missing:
                violations.append(f'{path.relative_to(BENCHMARK_ROOT)}:{node.lineno}: missing {sorted(missing)}')

    assert not violations, 'Custom RemoteDataLoader calls must preserve hub identity:\n' + '\n'.join(violations)
