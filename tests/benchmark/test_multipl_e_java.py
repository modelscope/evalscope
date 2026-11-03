"""Regression tests for complete Java classes and official MultiPL-E test suffixes."""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from evalscope.benchmarks.multipl_e.utils import build_full_code
from tests.benchmark.multipl_e_java_test_utils import java_major_version, run_java

FIXTURES = Path(__file__).parent / 'fixtures' / 'multipl_e_java'
TASKS = ['humaneval_53_add.json', 'mbpp_3_is_not_prime.json']
METHODS = {
    'humaneval_53_add.json': (
        'public static long add(long x, long y)',
        'return x + y;',
    ),
    'mbpp_3_is_not_prime.json': (
        'public static boolean isNotPrime(long n)',
        'for (long i = 2; i * i <= n; i++) if (n % i == 0) return true; return n < 2;',
    ),
}
# Hand-written expected class members, checked against the unchanged official rows.
# These expectations do not call any production parsing or assembly helper.
EXPECTED_MAINS = {
    'humaneval_53_add.json': '''
    public static void main(String[] args) {
        assert(add((0l), (1l)) == (1l));
        assert(add((1l), (0l)) == (1l));
        assert(add((2l), (3l)) == (5l));
        assert(add((5l), (7l)) == (12l));
        assert(add((7l), (5l)) == (12l));
    }
''',
    'mbpp_3_is_not_prime.json': '''
    public static void main(String[] args) {
        assert(isNotPrime((2l)) == (false));
        assert(isNotPrime((10l)) == (true));
        assert(isNotPrime((35l)) == (true));
        assert(isNotPrime((37l)) == (false));
    }
''',
}
EXPECTED_ENTRY_POINT = '''class Main {
    public static void main(String[] args) throws Exception { Problem.main(args); }
}'''


def _assemble(code: str, fixture_name: str = 'humaneval_53_add.json') -> str:
    record = json.loads((FIXTURES / fixture_name).read_text(encoding='utf-8'))
    return build_full_code(f'```java\n{code}\n```', record['language'], record['stop_tokens'], record['tests'])


def _compact(code: str) -> str:
    return re.sub(r'\s+', '', code)


def _solution(fixture_name: str, style: str) -> tuple[str, str]:
    signature, body = METHODS[fixture_name]
    if style == 'one_line_class':
        method = f'{signature} {{ {body} }}'
        return f'class Problem {{ {method} }}', method
    if style == 'one_line_method':
        method = f'    {signature} {{ {body} }}'
    else:
        indent = {'two_spaces': '  ', 'four_spaces': '    ', 'tabs': '\t', 'crlf': '    '}[style]
        method = f'{indent}{signature} {{\n{indent}{indent}{body}\n{indent}}}'
    source = f'class Problem {{\n{method}\n}}'
    return (source.replace('\n', '\r\n') if style == 'crlf' else source), method


STYLES = ['one_line_class', 'one_line_method', 'two_spaces', 'four_spaces', 'tabs', 'crlf']


@pytest.mark.parametrize('fixture_name', TASKS)
@pytest.mark.parametrize('style', STYLES)
def test_complete_class_receives_tests_before_its_closing_brace(fixture_name: str, style: str) -> None:
    source, method = _solution(fixture_name, style)

    assembled = _assemble(source, fixture_name)

    expected_class = f'class Problem {{\n{method}\n{EXPECTED_MAINS[fixture_name]}\n}}\n{EXPECTED_ENTRY_POINT}'
    assert _compact(assembled).endswith(_compact(expected_class))
    assert 'import java.util.*;' in assembled
    assert 'import org.javatuples.*;' in assembled


@pytest.mark.parametrize('fixture_name', TASKS)
@pytest.mark.parametrize('style', STYLES)
def test_official_assertions_pass_for_every_format(fixture_name: str, style: str) -> None:
    source, _ = _solution(fixture_name, style)

    result = run_java(_assemble(source, fixture_name))

    assert result['status'] == 'success', result


@pytest.mark.parametrize(
    'fixture_name,source',
    [
        ('humaneval_53_add.json', 'class Problem { public static long add(long x, long y) { return x - y; } }'),
        ('mbpp_3_is_not_prime.json', 'class Problem { public static boolean isNotPrime(long n) { return false; } }'),
    ],
)
def test_wrong_algorithm_still_fails_official_assertions(fixture_name: str, source: str) -> None:
    result = run_java(_assemble(source, fixture_name))

    assert result['stage'] == 'run', result
    assert result['status'] == 'error', result
    assert 'AssertionError' in result['stderr']


HELPER_SOLUTION = '''import java.util.function.LongBinaryOperator;
class Problem {
    public static long add(long x, long y) {
        if (x >= 0) {
            if (y >= 0) {
                return helper(x, y);
            }
        }
        return helper(x, y);
    }
    private static long helper(long x, long y) {
        LongBinaryOperator operation = (a, b) -> { return Adder.apply(a, b); };
        return operation.applyAsLong(x, y);
    }
    private static class Adder {
        static long apply(long x, long y) { return x + y; }
    }
}'''


def test_nested_blocks_and_helpers_after_target_method_are_preserved() -> None:
    assembled = _assemble(HELPER_SOLUTION)

    assert 'import java.util.function.LongBinaryOperator;' in assembled
    assert 'LongBinaryOperator operation = (a, b) -> { return Adder.apply(a, b); };' in assembled
    assert 'static long apply(long x, long y) { return x + y; }' in assembled
    assert assembled.index('static long apply') < assembled.index('public static void main')


def test_preserved_helpers_compile_and_pass() -> None:
    result = run_java(_assemble(HELPER_SOLUTION))

    assert result['status'] == 'success', result


def test_problem_constructor_and_type_references_are_preserved() -> None:
    source = '''class Problem {
    Problem() {}
    public static long add(long x, long y) { return new Problem().sum(x, y); }
    private long sum(long x, long y) { return x + y; }
}'''

    assembled = _assemble(source)

    assert 'class Main {' in assembled
    assert 'Problem() {}' in assembled
    assert 'return new Problem().sum(x, y);' in assembled
    result = run_java(assembled)
    assert result['status'] == 'success', result


def test_existing_public_main_class_name_is_preserved() -> None:
    source = '''public class Main {
    public static long add(long x, long y) { return x + y; }
}'''

    assembled = _assemble(source)

    assert 'public class Main {' in assembled
    assert 'class Solution' not in assembled
    result = run_java(assembled)
    assert result['status'] == 'success', result


def test_a_generated_main_is_not_silently_removed_to_make_the_solution_pass() -> None:
    source = '''public class Main {
    public static long add(long x, long y) { return x + y; }
    public static void main(String[] args) { throw new AssertionError("model main"); }
}'''

    assembled = _assemble(source)

    assert 'throw new AssertionError("model main");' in assembled
    result = run_java(assembled)
    assert result['status'] == 'error', result


def test_dollar_sign_helper_class_name_is_not_replaced_as_a_prefix() -> None:
    source = '''class Problem$Helper {
    static long sum(long x, long y) { return x + y; }
}
class Problem {
    public static long add(long x, long y) { return Problem$Helper.sum(x, y); }
}'''

    assembled = _assemble(source)

    assert 'class Problem$Helper {' in assembled
    assert 'return Problem$Helper.sum(x, y);' in assembled
    assert 'class Main$Helper' not in assembled
    result = run_java(assembled)
    assert result['status'] == 'success', result


LEXICAL_SOLUTION = r'''/* class Problem { a comment with unmatched braces: } } */
// class Problem { an unmatched opening brace
class Problem {
    static final String DESCRIPTION = "class Problem { } and an escaped quote: \" }";
    static final char CLOSE = '}';
    static final char QUOTE = '\'';
    static final char BACKSLASH = '\\';
    // class Problem { } }
    /* class Problem { nested-looking { braces } */
    public static long add(long x, long y) { return x + y; }
    static class Problematic {}
}'''


def test_comments_and_literals_do_not_change_class_boundary_or_get_renamed() -> None:
    assembled = _assemble(LEXICAL_SOLUTION)

    for preserved in [
        '/* class Problem { a comment with unmatched braces: } } */',
        '// class Problem { an unmatched opening brace',
        '"class Problem { } and an escaped quote: \\" }"',
        "static final char CLOSE = '}';",
        "static final char QUOTE = '\\'';",
        "static final char BACKSLASH = '\\\\';",
        'static class Problematic {}',
    ]:
        assert preserved in assembled
    assert '\nclass Main {' in assembled
    assert assembled.index('static class Problematic') < assembled.index('public static void main')


def test_comments_and_literals_still_compile_and_pass() -> None:
    result = run_java(_assemble(LEXICAL_SOLUTION))

    assert result['status'] == 'success', result


TEXT_BLOCK_SOLUTION = '''class Problem {
    static final String DESCRIPTION = """
        class Problem { unmatched braces: } } }
        // class Problem {
        """;
    public static long add(long x, long y) { return x + y; }
}'''


def test_text_block_contents_are_ignored_when_locating_class_boundary() -> None:
    assembled = _assemble(TEXT_BLOCK_SOLUTION)

    assert 'class Problem { unmatched braces: } } }' in assembled
    assert '// class Problem {' in assembled
    assert 'public static long add(long x, long y) { return x + y; }' in assembled
    assert assembled.index('""";') < assembled.index('public static void main')


def test_text_blocks_compile_and_pass_when_supported_by_local_jdk() -> None:
    if java_major_version() < 15:
        pytest.skip('Java text-block integration requires JDK 15 or newer; its assembly check still runs.')

    result = run_java(_assemble(TEXT_BLOCK_SOLUTION))

    assert result['status'] == 'success', result


UNICODE_SOLUTION = r'''class Probl\u0065m \u007b
    static final String MESSAGE = "\u007b \u007d";
    public static long add(long x, long y) \u007b return x + y; \u007d
\u007d'''


def test_unicode_escape_strings_are_preserved_and_structural_escapes_are_recognized() -> None:
    assembled = _assemble(UNICODE_SOLUTION)

    assert r'static final String MESSAGE = "\u007b \u007d";' in assembled
    assert r'public static long add(long x, long y) \u007b return x + y; \u007d' in assembled
    assert r'class Probl\u0065m \u007b' in assembled
    assert assembled.index('return x + y;') < assembled.index('public static void main')


def test_legal_unicode_escape_source_compiles_and_passes() -> None:
    result = run_java(_assemble(UNICODE_SOLUTION))

    assert result['status'] == 'success', result


def test_unicode_newline_cannot_hide_a_premature_class_closure_in_a_comment() -> None:
    source = r'''class Problem {
    // This Unicode escape ends the comment: \u000a }
    public static long add(long x, long y) { return x + y; }
}'''

    result = run_java(_assemble(source))

    assert result['status'] == 'error', result
    assert result['stage'] == 'compile', result


def test_incomplete_completion_keeps_the_official_method_closure() -> None:
    assembled = _assemble('class Problem {\n    public static long add(long x, long y) {\n        return x + y;')

    expected = 'class Problem { public static long add(long x, long y) { return x + y; }'
    expected += EXPECTED_MAINS['humaneval_53_add.json'] + '}' + EXPECTED_ENTRY_POINT
    assert _compact(assembled).endswith(_compact(expected))


@pytest.mark.parametrize(
    'tests',
    [
        '',
        'public static void main(String[] args) {}\n}',
        '}\n}',
        '    }\n    public static void main(String[] args) {\n    assert(add(2, 3) == 5);\n',
    ],
    ids=['empty', 'missing-method-closure', 'missing-test-main', 'unclosed-test-main'],
)
def test_malformed_dataset_test_suffix_is_reported_as_a_data_error(tests: str) -> None:
    with pytest.raises(ValueError):
        build_full_code(
            '```java\nclass Problem { public static long add(long x, long y) { return x + y; } }\n```',
            'java',
            ['\n    }\n'],
            tests,
        )


@pytest.mark.parametrize(
    'source',
    [
        'class Problem { public static long add(long x, long y) { return x + y; } }\nnot Java code',
        'class Problem { public static long add(long x, long y) { return x + y; } }\n/* unclosed comment',
        'class Problem { public static long add(long x, long y) { if (true) { return x + y;',
    ],
    ids=['trailing-garbage', 'unterminated-trailing-comment', 'unfinished-nested-block'],
)
def test_malformed_output_is_not_truncated_into_a_passing_program(source: str) -> None:
    result = run_java(_assemble(source))

    assert result['status'] == 'error', result
    assert result['stage'] == 'compile', result


def test_non_java_stop_token_and_test_suffix_behavior_is_unchanged() -> None:
    assembled = build_full_code(
        '```python\ndef add(x, y):\n    return x + y\n# STOP\nraise RuntimeError("discard me")\n```',
        'python',
        ['\n# STOP'],
        'assert add(2, 3) == 5',
    )

    assert assembled.endswith('\ndef add(x, y):\n    return x + y\nassert add(2, 3) == 5')
    assert 'discard me' not in assembled


@pytest.mark.parametrize(
    'source',
    [
        'class Problem { public static long add(long x, long y) { '
        'long Problem = x; long Main = y; return Problem + Main; } }',
        'class Problem { public static long add(long x, long y) { '
        'return Problem.class.getName().equals("Problem") ? x + y : 0; } }',
    ],
    ids=['local-identifier-collision', 'reflective-class-name'],
)
def test_solution_identifier_and_class_name_semantics_are_preserved(source: str) -> None:
    result = run_java(_assemble(source))

    assert result['status'] == 'success', result


@pytest.mark.parametrize(
    'source',
    [
        'public final class Problem { Problem() {} '
        'public static long add(long x, long y) { return new Problem().sum(x, y); } '
        'private long sum(long x, long y) { return x + y; } }',
        'class Problem { static class Main {} '
        'public static long add(long x, long y) { return x + y; } }',
        '@interface Marker { int[] value(); }\n'
        'public @Marker({1, 2}) final class Problem { '
        'public static long add(long x, long y) { return x + y; } }',
    ],
    ids=['public-class-and-constructor', 'nested-main-name', 'public-before-annotation-array'],
)
def test_problem_class_can_keep_its_name_with_a_main_executor_entry_point(source: str) -> None:
    result = run_java(_assemble(source))

    assert result['status'] == 'success', result


def test_repeated_public_modifier_is_not_repaired_into_a_passing_program() -> None:
    result = run_java(_assemble(
        'public public class Problem { public static long add(long x, long y) { return x + y; } }'
    ))

    assert result['status'] == 'error', result
    assert result['stage'] == 'compile', result


def test_unterminated_text_block_with_backslashes_finishes_assembly() -> None:
    record = json.loads((FIXTURES / 'humaneval_53_add.json').read_text(encoding='utf-8'))
    source = 'class Problem { static String text = """\n' + '\\' * 64
    script = (
        'import json, sys; from evalscope.benchmarks.multipl_e.utils import build_full_code; '
        'source, tests = json.load(sys.stdin); '
        'code = build_full_code("```java\\n" + source + "\\n```", "java", [], tests); '
        'assert "#error" in code'
    )
    try:
        completed = subprocess.run(
            [sys.executable, '-c', script],
            input=json.dumps([source, record['tests']]),
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )
    except subprocess.TimeoutExpired:
        pytest.fail('Malformed model output must not hang assembly before sandbox execution.')

    assert completed.returncode == 0, completed.stdout + completed.stderr
