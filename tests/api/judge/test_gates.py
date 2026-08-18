"""Structural gates that stop the judge-parsing debt from growing back.

A benchmark adapter must not call a judge model, must not touch a raw judge response, and must
not define its own parser. ``PENDING_MIGRATION`` is empty: every adapter that scores with a judge
now goes through ``evalscope.api.judge``, and nothing may be added back.
"""
import ast
import os
from typing import Dict, List, Set, Tuple

BENCHMARKS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    'evalscope',
    'benchmarks',
)

RAW_RESPONSE_NAMES = ('judge_response', 'grading_response', 'judge_raw', 'judgment', 'last_response')
PARSER_CALLS = ('search', 'match', 'fullmatch', 'findall', 'finditer', 'loads')

# Empty by design -- a new entry here means a regression, not a to-do. Note the ``set()`` call:
# ``{}`` would be a dict and would silently break the set algebra below.
PENDING_MIGRATION: Set[str] = set()

# These adapters use the judge as a data transformer (not a verdict/score producer) or have a
# fundamentally different scoring architecture that does not fit the OutputContract model.
PERMANENTLY_EXEMPT: Set[str] = {
    'job_bench/job_bench_adapter.py',
    'wide_search/wide_search_adapter.py',
}


def adapter_files() -> List[str]:
    paths = []
    for dirpath, _, filenames in os.walk(BENCHMARKS_ROOT):
        for name in filenames:
            if name.endswith('_adapter.py'):
                paths.append(os.path.join(dirpath, name))
    return sorted(paths)


def relative(path: str) -> str:
    return os.path.relpath(path, BENCHMARKS_ROOT)


def scan(path: str) -> List[str]:
    """Return the gate violations in one adapter file."""
    with open(path, encoding='utf-8') as handle:
        tree = ast.parse(handle.read(), filename=path)

    aliases = _llm_judge_aliases(tree)
    violations: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == 'judge' and _is_llm_judge(node.func.value, aliases):
                violations.append('calls a judge model directly')
            elif node.func.attr in PARSER_CALLS and _touches_raw_response(node.args):
                violations.append(f'parses a raw judge response via {node.func.attr}()')
        elif isinstance(node, ast.Attribute) and node.attr == 'judge' and _is_llm_judge(node.value, aliases):
            # Handing the bound method to a helper reaches the judge just the same.
            violations.append('passes the judge model to a helper')
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            params = {arg.arg for arg in node.args.args + node.args.kwonlyargs}
            if params & set(RAW_RESPONSE_NAMES):
                violations.append(f'{node.name}() takes a raw judge response')
    return sorted(set(violations))


def _llm_judge_aliases(tree: ast.AST) -> Set[str]:
    """Local names bound to ``self.llm_judge``, e.g. ``judge = self.llm_judge``."""
    aliases = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if isinstance(node.value, ast.Attribute) and node.value.attr == 'llm_judge':
            for target in node.targets:
                if isinstance(target, ast.Name):
                    aliases.add(target.id)
    return aliases


def _is_llm_judge(receiver: ast.expr, aliases: Set[str]) -> bool:
    """True when the attribute is read off the LLM judge, not off an unrelated ``judger``."""
    if isinstance(receiver, ast.Attribute):
        return receiver.attr == 'llm_judge'
    if isinstance(receiver, ast.Name):
        return receiver.id in aliases
    return False


def _touches_raw_response(args: List[ast.expr]) -> bool:
    for arg in args:
        for node in ast.walk(arg):
            if isinstance(node, ast.Name) and any(key in node.id for key in RAW_RESPONSE_NAMES):
                return True
            if isinstance(node, ast.Attribute) and any(key in node.attr for key in RAW_RESPONSE_NAMES):
                return True
    return False


def current_violations() -> Dict[str, List[str]]:
    return {relative(path): found for path in adapter_files() if (found := scan(path))}


def test_no_new_adapter_touches_judge_output():
    allowed = PENDING_MIGRATION | PERMANENTLY_EXEMPT
    offenders = {name: found for name, found in current_violations().items() if name not in allowed}

    assert not offenders, (
        'These adapters call a judge model or parse its response directly. Use '
        'evalscope.api.judge contracts instead:\n' + '\n'.join(f'  {name}: {found}' for name, found in offenders.items())
    )


def test_pending_migration_list_has_no_stale_entries():
    """A migrated adapter must be removed from the list, so the gate can only tighten."""
    violating = set(current_violations())
    stale = sorted(PENDING_MIGRATION - violating)

    assert not stale, f'These adapters no longer violate the gate; remove them from PENDING_MIGRATION: {stale}'


def test_gate_detects_a_synthetic_violation(tmp_path):
    source = '''
class Adapter:
    def llm_match_score(self, prediction, reference, task_state):
        judge_response = self.llm_judge.judge(prompt='x')
        return re.search(r'(A|B)', judge_response)
'''
    path = tmp_path / 'synthetic_adapter.py'
    path.write_text(source, encoding='utf-8')

    found = scan(str(path))

    assert 'calls a judge model directly' in found
    assert any('parses a raw judge response' in item for item in found)


def test_gate_catches_a_judge_handed_to_a_helper(tmp_path):
    """Passing the bound method reaches the judge just as calling it does."""
    source = '''
class Adapter:
    def llm_match_score(self, prediction, reference, task_state):
        judge = self.llm_judge
        return Scorer(judge=judge.judge).run(prediction)
'''
    path = tmp_path / 'helper_adapter.py'
    path.write_text(source, encoding='utf-8')

    assert 'passes the judge model to a helper' in scan(str(path))


def test_gate_ignores_an_unrelated_judger(tmp_path):
    """``olympiad_bench`` uses a rule-based MathJudger; it must not be flagged."""
    source = '''
class Adapter:
    def match_score(self, prediction, reference, task_state):
        judger = MathJudger()
        return judger.judge(prediction, reference)
'''
    path = tmp_path / 'rule_adapter.py'
    path.write_text(source, encoding='utf-8')

    assert scan(str(path)) == []
