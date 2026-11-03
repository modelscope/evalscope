"""Regression tests for complete Java classes and official MultiPL-E test suffixes."""

import json
import re
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import pytest

from evalscope.api.evaluator import TaskState
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.multipl_e.utils import build_full_code
from evalscope.config import TaskConfig

# Official MultiPL-E Java rows, unchanged from revision 28441b6024e71d4a1c1c0f6bf171c935cd5a43f2.
# Source: https://huggingface.co/datasets/nuprl/MultiPL-E/tree/28441b6024e71d4a1c1c0f6bf171c935cd5a43f2
# HumanEval source: https://github.com/openai/human-eval
# MBPP source: https://github.com/google-research/google-research/tree/master/mbpp
# MBPP attribution: Austin et al., Program Synthesis with Large Language Models (2021), CC BY 4.0.
# License: https://creativecommons.org/licenses/by/4.0/
# The upstream license notices below apply to the copied fixtures, not to EvalScope as a whole.
#
# The MIT License
#
# Copyright (c) OpenAI (https://openai.com)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# BSD 3-Clause License with Machine Learning Restriction
#
# Copyright (c) 2022, Northeastern University, Oberlin College, Roblox Inc,
# Stevens Institute of Technology, University of Massachusetts Amherst, and
# Wellesley College.
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# 4.  The contents of this repository may not be used as training data for any
#     machine learning model, including but not limited to neural networks.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
FIXTURES = {
    'humaneval_53_add': {
        'name': 'HumanEval_53_add',
        'language': 'java',
        'prompt': (
            'import java.util.*;\n'
            'import java.lang.reflect.*;\n'
            'import org.javatuples.*;\n'
            'import java.security.*;\n'
            'import java.math.*;\n'
            'import java.io.*;\n'
            'import java.util.stream.*;\n'
            'class Problem {\n'
            '    // Add two numbers x and y\n'
            '    // >>> add((2l), (3l))\n'
            '    // (5l)\n'
            '    // >>> add((5l), (7l))\n'
            '    // (12l)\n'
            '    public static long add(long x, long y) {\n'
        ),
        'doctests': 'transform',
        'original': (
            '/work/arjunguha-research-group/arjun/repos/nuprl/MultiPL-E/datasets/../datasets/originals-'
            'with-cleaned-doctests/HumanEval_53_add.py'
        ),
        'prompt_terminology': 'reworded',
        'tests': (
            '    }\n'
            '    public static void main(String[] args) {\n'
            '    assert(add((0l), (1l)) == (1l));\n'
            '    assert(add((1l), (0l)) == (1l));\n'
            '    assert(add((2l), (3l)) == (5l));\n'
            '    assert(add((5l), (7l)) == (12l));\n'
            '    assert(add((7l), (5l)) == (12l));\n'
            '    }\n'
            '\n'
            '}\n'
        ),
        'stop_tokens': ['\n    }\n'],
    },
    'mbpp_3_is_not_prime': {
        'name': 'mbpp_3_is_not_prime',
        'language': 'java',
        'prompt': (
            'import java.util.*;\n'
            'import java.lang.reflect.*;\n'
            'import org.javatuples.*;\n'
            'import java.security.*;\n'
            'import java.math.*;\n'
            'import java.io.*;\n'
            'import java.util.stream.*;\n'
            'class Problem {\n'
            '    // Write a javathon function to identify non-prime numbers.\n'
            '    public static boolean isNotPrime(long n) {\n'
        ),
        'doctests': 'transform',
        'original': '/Users/arjun/repos/nuprl/MultiPL-E/datasets/../datasets/mbpp-typed/mbpp_3_is_not_prime.py',
        'prompt_terminology': 'reworded',
        'tests': (
            '    }\n'
            '    public static void main(String[] args) {\n'
            '    assert(isNotPrime((2l)) == (false));\n'
            '    assert(isNotPrime((10l)) == (true));\n'
            '    assert(isNotPrime((35l)) == (true));\n'
            '    assert(isNotPrime((37l)) == (false));\n'
            '    }\n'
            '\n'
            '}\n'
        ),
        'stop_tokens': ['\n    }\n'],
    },
}
TASKS = ['humaneval_53_add', 'mbpp_3_is_not_prime']
METHODS = {
    'humaneval_53_add': (
        'public static long add(long x, long y)',
        'return x + y;',
    ),
    'mbpp_3_is_not_prime': (
        'public static boolean isNotPrime(long n)',
        'for (long i = 2; i * i <= n; i++) if (n % i == 0) return true; return n < 2;',
    ),
}
# Hand-written expected class members, checked against the unchanged official rows.
# These expectations do not call any production parsing or assembly helper.
EXPECTED_MAINS = {
    'humaneval_53_add': '''
    public static void main(String[] args) {
        assert(add((0l), (1l)) == (1l));
        assert(add((1l), (0l)) == (1l));
        assert(add((2l), (3l)) == (5l));
        assert(add((5l), (7l)) == (12l));
        assert(add((7l), (5l)) == (12l));
    }
''',
    'mbpp_3_is_not_prime': '''
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


@lru_cache(maxsize=1)
def _java_major_version() -> int:
    if not shutil.which('javac') or not shutil.which('java'):
        pytest.skip('Java integration requires local javac and java; assembly tests still run.')
    result = subprocess.run(['javac', '-version'], capture_output=True, text=True, timeout=10)
    if result.returncode:
        pytest.skip('Java integration requires a working local JDK; assembly tests still run.')
    match = re.search(r'javac (\d+)(?:\.(\d+))?', result.stdout + result.stderr)
    assert match is not None, result.stdout + result.stderr
    return int(match.group(2) if match.group(1) == '1' else match.group(1))


def _run_java(code: str) -> dict:
    """Run fixed arithmetic fixtures with native assertions, without a sandbox service.

    An unused package marker resolves the javatuples import; it provides no tuple API.
    """
    _java_major_version()
    with tempfile.TemporaryDirectory(prefix='evalscope-multipl-java-') as directory:
        workdir = Path(directory)
        source = workdir / 'Main.java'
        source.write_text(code, encoding='utf-8')
        marker = workdir / 'org' / 'javatuples' / 'UnusedImport.java'
        marker.parent.mkdir(parents=True)
        marker.write_text('package org.javatuples; public final class UnusedImport {}\n', encoding='utf-8')
        compiled = subprocess.run(
            ['javac', '-cp', str(workdir), '-d', str(workdir), str(marker), str(source)],
            capture_output=True, text=True, timeout=20,
        )
        if compiled.returncode:
            return {'status': 'error', 'stage': 'compile', 'stdout': compiled.stdout, 'stderr': compiled.stderr}
        result = subprocess.run(
            ['java', '-ea', '-cp', str(workdir), 'Main'], capture_output=True, text=True, timeout=20
        )
        return {
            'status': 'success' if result.returncode == 0 else 'error',
            'stage': 'run', 'stdout': result.stdout, 'stderr': result.stderr,
        }


def _assemble(code: str, fixture_name: str = 'humaneval_53_add') -> str:
    record = FIXTURES[fixture_name]
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

    result = _run_java(_assemble(source, fixture_name))

    assert result['status'] == 'success', result


@pytest.mark.parametrize(
    'fixture_name,source',
    [
        ('humaneval_53_add', 'class Problem { public static long add(long x, long y) { return x - y; } }'),
        ('mbpp_3_is_not_prime', 'class Problem { public static boolean isNotPrime(long n) { return false; } }'),
    ],
)
def test_wrong_algorithm_still_fails_official_assertions(fixture_name: str, source: str) -> None:
    result = _run_java(_assemble(source, fixture_name))

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
    result = _run_java(_assemble(HELPER_SOLUTION))

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
    result = _run_java(assembled)
    assert result['status'] == 'success', result


def test_existing_public_main_class_name_is_preserved() -> None:
    source = '''public class Main {
    public static long add(long x, long y) { return x + y; }
}'''

    assembled = _assemble(source)

    assert 'public class Main {' in assembled
    assert 'class Solution' not in assembled
    result = _run_java(assembled)
    assert result['status'] == 'success', result


def test_a_generated_main_is_not_silently_removed_to_make_the_solution_pass() -> None:
    source = '''public class Main {
    public static long add(long x, long y) { return x + y; }
    public static void main(String[] args) { throw new AssertionError("model main"); }
}'''

    assembled = _assemble(source)

    assert 'throw new AssertionError("model main");' in assembled
    result = _run_java(assembled)
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
    result = _run_java(assembled)
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
    result = _run_java(_assemble(LEXICAL_SOLUTION))

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
    if _java_major_version() < 15:
        pytest.skip('Java text-block integration requires JDK 15 or newer; its assembly check still runs.')

    result = _run_java(_assemble(TEXT_BLOCK_SOLUTION))

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
    result = _run_java(_assemble(UNICODE_SOLUTION))

    assert result['status'] == 'success', result


def test_unicode_newline_cannot_hide_a_premature_class_closure_in_a_comment() -> None:
    source = r'''class Problem {
    // This Unicode escape ends the comment: \u000a }
    public static long add(long x, long y) { return x + y; }
}'''

    result = _run_java(_assemble(source))

    assert result['status'] == 'error', result
    assert result['stage'] == 'compile', result


def test_incomplete_completion_keeps_the_official_method_closure() -> None:
    assembled = _assemble('class Problem {\n    public static long add(long x, long y) {\n        return x + y;')

    expected = 'class Problem { public static long add(long x, long y) { return x + y; }'
    expected += EXPECTED_MAINS['humaneval_53_add'] + '}' + EXPECTED_ENTRY_POINT
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
    result = _run_java(_assemble(source))

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
    result = _run_java(_assemble(source))

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
    result = _run_java(_assemble(source))

    assert result['status'] == 'success', result


def test_repeated_public_modifier_is_not_repaired_into_a_passing_program() -> None:
    result = _run_java(_assemble(
        'public public class Problem { public static long add(long x, long y) { return x + y; } }'
    ))

    assert result['status'] == 'error', result
    assert result['stage'] == 'compile', result


def test_unterminated_text_block_with_backslashes_finishes_assembly() -> None:
    record = FIXTURES['humaneval_53_add']
    source = 'class Problem { static String text = """\n' + '\\' * 64
    script = (
        'import json, sys; from evalscope.benchmarks.multipl_e.utils import build_full_code; '
        'from evalscope.utils.code_utils import default_extract_helper; '
        'prediction, source, tests = json.load(sys.stdin); '
        'assert default_extract_helper(prediction, "java") == source; '
        'code = build_full_code(prediction, "java", [], tests); '
        'assert "#error" in code'
    )
    try:
        completed = subprocess.run(
            [sys.executable, '-c', script],
            input=json.dumps([f'```java\n{source}\n```', source, record['tests']]),
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )
    except subprocess.TimeoutExpired:
        pytest.fail('Malformed model output must not hang assembly before sandbox execution.')

    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize('benchmark', ['multiple_humaneval', 'multiple_mbpp'])
@pytest.mark.parametrize('correct', [True, False], ids=['correct', 'wrong'])
def test_adapters_grade_complete_java_solutions(
    benchmark: str, correct: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = TaskConfig(model='mock', datasets=[benchmark], sandbox={'enabled': True})
    adapter = get_benchmark(benchmark, config)
    fixture = 'humaneval_53_add' if benchmark == 'multiple_humaneval' else 'mbpp_3_is_not_prime'
    sample = adapter.record_to_sample(FIXTURES[fixture])
    assert 'Write the complete code and wrap it in markdown fenced code.' in adapter.format_prompt_template(sample)
    assert adapter.benchmark_meta.evaluation_version == 'v1.1'
    source, _ = _solution(fixture, 'one_line_class')
    if not correct:
        wrong_body = 'return 0;' if benchmark == 'multiple_humaneval' else 'return false;'
        source = source.replace(METHODS[fixture][1], wrong_body)
    prediction = f'```java\n{source}\n```'

    def execute(code: str, timeout: int, language: str) -> dict:
        assert language == 'java'
        assert timeout == 30
        return _run_java(code)

    # Execute fixed fixtures natively; no sandbox service or model endpoint is used.
    monkeypatch.setattr(adapter, 'execute_code_in_sandbox', execute)
    state = TaskState(model='mock', sample=sample)
    score = adapter.match_score(prediction, adapter.extract_answer(prediction, state), '', state)
    assert score.value == {'acc': correct}
    assert score.metadata['run_language'] == 'java'
    assert score.metadata['task_id'] == FIXTURES[fixture]['name']
    execution = score.metadata['execution_result']
    assert execution['stage'] == 'run', execution
    if not correct:
        assert 'AssertionError' in execution['stderr']
