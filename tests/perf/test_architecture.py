import ast
from pathlib import Path


def test_core_layers_do_not_depend_on_entrypoints_or_reporting() -> None:
    root = Path('evalscope/perf')
    core_dirs = ['config', 'domain', 'transport', 'protocols', 'workloads', 'metrics', 'results']
    forbidden = ('evalscope.cli', 'evalscope.service', 'evalscope.perf.reporting', 'evalscope.perf.sla')
    violations = []
    for directory in core_dirs:
        for path in (root / directory).rglob('*.py'):
            tree = ast.parse(path.read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                else:
                    continue
                if any(name.startswith(forbidden) for name in names):
                    violations.append(f'{path}:{node.lineno}')
    assert not violations


def test_perf_core_never_exits_the_process() -> None:
    for path in Path('evalscope/perf').rglob('*.py'):
        assert 'sys.exit(' not in path.read_text(encoding='utf-8'), path
