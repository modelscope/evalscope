"""Regression coverage for the public metrics and benchmark numeric policies."""

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig
from evalscope.constants import ScoreStatus
from evalscope.metrics.math.contracts import InvalidMathReference
from evalscope.metrics.math.parser import compare_answers, extract_answer, extract_boxed_answers, math_equal
from evalscope.metrics.nlp.metrics import Accuracy, ExactMatch, MathAcc


@pytest.mark.parametrize(('prediction', 'reference', 'expected'), [
    ('1800', '18', False), ('0.18', '18', False), ('18%', '0.18', True), ('10%', '10', True),
    ('10%', '10.0', False), ('', '', False), (' ', '0', False), ('0', '0', True),
    (r'\frac{1}{2}', '0.5', True), (r'\sqrt{4}', '2', True), ('x+x', '2x', True),
    (r'\{1,2\}', r'\{2,1\}', True), ('[0,1)', '[0,1)', True), ('[0,1)', '(0,1)', False),
    (r'\begin{pmatrix}1&2\\3&4\end{pmatrix}', r'\begin{pmatrix}1&2\\3&4\end{pmatrix}', True),
    ('x=1', 'x=1', True), (r'\frac{', r'\frac{', False),
])
def test_public_math_equal(prediction: Any, reference: Any, expected: Any) -> None:
    assert math_equal(prediction, reference) is expected


def test_issue_paths_and_display() -> None:
    assert Accuracy(numeric=True).apply(['1800', '', '0', '18%'], ['18', '0', '0', '.18']) == [0, 0, 1, 1]
    assert MathAcc().apply([r'work 999; \boxed{18%}', ''], ['.18', '0']) == [1, 0]
    assert extract_answer(r'work 999; \boxed{\boxed{18%}}') == '18%'
    assert extract_answer('Answer: 18%') == '18%'
    assert extract_answer('18 percent') == '18%'
    assert extract_answer('no answer available') == ''
    # Text exact matching is a separate contract.
    assert ExactMatch().apply([''], ['']) == [1]


def test_plain_percent_survives_adapter_extraction() -> None:
    score = adapter('gsm8k').calculate_metrics(state('Answer: 18%', '.18')).score
    assert score.extracted_prediction == '18%'
    assert score.main_value == 1


def test_cmmu_fill_answer_extracts_the_complete_output() -> None:
    score = adapter('cmmu').calculate_metrics(
        state(r'Work: 2 * 9 = 18. Final answer: \boxed{18}', '18', {'type': 'fill-in-the-blank'})).score
    assert score.main_value == 1


def test_fallback_text_cannot_establish_equality() -> None:
    assert not math_equal(r'\frac{', r'\frac{')
    with pytest.raises(InvalidMathReference):
        Accuracy(numeric=True).apply(['0'], [r'\frac{'])


def test_legacy_switches_warn_instead_of_maintaining_another_grader() -> None:
    with pytest.warns(DeprecationWarning):
        assert math_equal('18%', '.18', include_percentage=False, is_close=False, timeout=True)
    with pytest.warns(DeprecationWarning):
        assert extract_answer('Answer: 2', use_last_number=False) == '2'


def test_ordered_boxes_keep_duplicates_and_ignore_truncation() -> None:
    assert extract_boxed_answers(r'\boxed{1}, \boxed{1}, \boxed{\frac{1}{2}} \boxed{') == [
        '1', '1', r'\frac{1}{2}',
    ]
    assert extract_boxed_answers(r'\boxed{\{1,2\}}') == [r'\{1,2\}']


def test_truncated_outer_box_cannot_promote_nested_answers() -> None:
    prediction = r'\boxed{1} \boxed{unfinished{ \boxed{2}'
    assert extract_boxed_answers(prediction) == ['1']
    benchmark = get_benchmark('hipho', TaskConfig(datasets=['hipho'], judge={'strategy': 'llm'}))
    metadata = {'answers': ['1', '2'], 'marking': []}
    assert benchmark._build_answer_cases(metadata, prediction) == []
    assert benchmark._reduce_answer([], metadata, prediction).value == {'acc': 0.5}
    assert extract_boxed_answers(r'\boxed{\boxed{1}} \boxed{2}') == ['1', '2']


@pytest.mark.parametrize(('prediction', 'reference', 'absolute', 'expected'), [
    ('1.005', '1', [0.01], True), ('1.01', '1', [0.01], True), ('1.010001', '1', [0.01], False),
    ('1.02', '1', [0.01], False),
    ('1800', '18', [1e-8], False), ('0.18', '18', [1e-8], False),
    ('2.05,1.005', '2,1', [0.1,0.01], True), ('2.05,1.05', '2,1', [0.1,0.01], False),
    ('1,1', '1,2', [1e-8], False),
    ('1', '1,1', [1e-8], False), ('1,1', '1,1', [1e-8], True),
])
def test_olympiad_tolerance_and_multiple_answers(prediction: Any, reference: Any, absolute: Any, expected: Any) -> None:
    assert compare_answers(prediction, reference, absolute_tolerance=absolute,
                           multiple_answers=',' in reference).matched is expected


def state(prediction: Any, reference: Any, metadata: Any=None, choices: Any=None) -> Any:
    return TaskState(model='offline', sample=Sample(input='question', target=reference, metadata=metadata or {}, choices=choices or []),
                     output=ModelOutput.from_content('offline', prediction), completed=True)


def adapter(name: Any) -> Any:
    return get_benchmark(name, config=TaskConfig(datasets=[name], judge={'strategy':'rule'}))


@pytest.mark.parametrize(('prediction','expected'), [('3',1),('003',1),('3.0',1),('1+2',0),
                                                        (r'\frac{6}{2}',0),('3.2',0),('',0)])
def test_aime_integer_format(prediction: Any, expected: Any) -> None:
    score = adapter('aime24').calculate_metrics(state(r'\boxed{' + prediction + '}', '3')).score
    assert score.main_value == expected


def test_aime_noninteger_reference_is_excluded() -> None:
    score = adapter('aime24').calculate_metrics(state(r'\boxed{3}', 'x')).score
    assert score.status is ScoreStatus.EXCLUDED
    assert score.value == {}


def test_docmath_symbolic_reference_is_excluded_on_numeric_type() -> None:
    score = adapter('docmath').calculate_metrics(state(r'\boxed{x}', 'x', {'answer_type': 'float'})).score
    assert score.status is ScoreStatus.EXCLUDED
    assert score.value == {}


@pytest.mark.parametrize(('prediction','reference','answer_type','expected'), [
    ('500','5','int',0),('.05','5','int',0),('','0','int',0),('no answer','0','int',0),
    ('0','0','int',1),('100.1','100','float',1),('100.2','100','float',0),
    ('True','True','bool',1),('False','False','bool',1),('True','False','bool',0),
    ('no answer','False','bool',0),('', 'False', 'bool',0),('Therefore, the answer is no.', 'False','bool',1),
])
def test_docmath_rule_path(prediction: Any, reference: Any, answer_type: Any, expected: Any) -> None:
    result = adapter('docmath').calculate_metrics(state(prediction, reference, {'answer_type':answer_type})).score
    assert result.main_value == expected


@pytest.mark.parametrize('reference', ['', r'\frac{'])
def test_invalid_reference_is_excluded_in_real_adapter(reference: str) -> None:
    result = adapter('gsm8k').calculate_metrics(state('0', reference)).score
    assert result.status is ScoreStatus.EXCLUDED
    assert result.value == {}


@pytest.mark.parametrize(('name','metadata','prediction'), [
    ('math_vista', {'question_type':'multi_choice'}, 'Answer: B'),
    ('math_vision', {'question_type':'multi_choice'}, 'Answer: B'),
    ('math_verse', {'question_type':'multi-choice'}, r'\boxed{B}'),
    ('math_verse', {'question_type':'multi-choice'}, 'ANSWER: B'),
    ('cmmu', {'type':'multiple-choice'}, '答案：B'),
    ('agieval', {'subset':'lsat-ar'}, 'Answer: B'),
])
def test_mixed_benchmarks_keep_categorical_scores(name: Any, metadata: Any, prediction: Any) -> None:
    task_state = state(prediction,'B',metadata, choices=['first','second','third','fourth'])
    score = adapter(name).calculate_metrics(task_state).score
    assert score.main_value == 1
    if name in ('math_vista','math_vision','math_verse','cmmu'):
        assert score.value == {'accuracy':1}


def test_parallel_matches_sequential() -> None:
    cases = [('1800','18'),('18%','.18'),(r'\frac{1}{2}','.5'),('x+x','2x')] * 12
    sequential = [math_equal(*case) for case in cases]
    with ThreadPoolExecutor(max_workers=12) as executor:
        parallel = list(executor.map(lambda case: math_equal(*case), cases))
    assert parallel == sequential


def test_olympiad_delimited_components_and_reference_group() -> None:
    metadata = {'final_answer': [r'$x+x$, $0$', r'$2x$, $0$'],
                'answer_type': 'Expression,Numerical', 'is_multiple_answer': True, 'error': ',0'}
    score = adapter('olympiad_bench').calculate_metrics(
        state(r'\boxed{2x, 0}', ','.join(metadata['final_answer']), metadata)).score
    assert score.main_value == 1
    assert compare_answers('1/3,0', r'$\frac{1}{3}$ , 0', multiple_answers=True).matched


@pytest.mark.parametrize(('prediction', 'expected'), [
    (r'\Delta=3.629\cdot10^{-11}', 1), (r'\Delta=3\cdot10^{-11}', 0),
    (r'\Delta=3.73\cdot10^{-11}', 1), (r'\Delta=3.7301\cdot10^{-11}', 0),
])
def test_olympiad_labeled_numeric_component_tolerance(prediction: str, expected: int) -> None:
    metadata = {'final_answer': [r'\Delta=3.63\cdot10^{-11}'], 'answer_type': 'Numerical', 'error': '1e-12'}
    score = adapter('olympiad_bench').calculate_metrics(
        state(r'\boxed{' + prediction + '}', metadata['final_answer'][0], metadata)).score
    assert score.main_value == expected


@pytest.mark.parametrize(('error', 'expected'), [(0, 0), (0.0, 0), ('0', 0), (None, 1), ('', 1)])
def test_olympiad_zero_tolerance_is_not_missing(error: Any, expected: int) -> None:
    metadata = {'final_answer': ['1'], 'answer_type': 'Numerical', 'error': error}
    score = adapter('olympiad_bench').calculate_metrics(state(r'\boxed{1.000000001}', '1', metadata)).score
    assert score.main_value == expected
    assert adapter('olympiad_bench').calculate_metrics(state(r'\boxed{1}', '1', metadata)).score.main_value == 1


def test_olympiad_multiple_answers_keep_zero_component_tolerance() -> None:
    metadata = {'final_answer': ['1,2'], 'answer_type': 'Numerical,Numerical',
                'is_multiple_answer': True, 'error': '0,0.1'}
    score = adapter('olympiad_bench').calculate_metrics(state(r'\boxed{1.000000001,2.05}', '1,2', metadata)).score
    assert score.main_value == 0
    assert adapter('olympiad_bench').calculate_metrics(state(r'\boxed{1,2.05}', '1,2', metadata)).score.main_value == 1


def test_scientific_numeric_literals_are_formatted_for_upstream() -> None:
    assert math_equal('1e-3', '.001')
    assert not math_equal('1e-3', '1')
    assert MathAcc().apply([r'\boxed{3.63e-11}'], ['3.63e-11']) == [1]
    assert extract_answer(r'\boxed{3.63e-11}') == '3.63e-11'


def test_legacy_helpers_delegate_upstream() -> None:
    from evalscope.metrics.math.parser import is_digit, numeric_equal, parse_digits, symbolic_equal

    assert parse_digits(r'\frac{1}{2}') == 0.5
    assert parse_digits('not a number') is None
    assert is_digit('0') and not is_digit('x')
    assert numeric_equal(0.5, 0.5)
    assert symbolic_equal('x+x', '2x')


def test_tir_numeric_errors_are_excluded(monkeypatch: pytest.MonkeyPatch) -> None:
    from evalscope.metrics.math import parser
    from evalscope.metrics.math.contracts import MathEvaluationError

    def fail(*args: object, **kwargs: object) -> None:
        raise MathEvaluationError('verification failed')

    monkeypatch.setattr(parser, 'compare_answers', fail)
    score = adapter('tir_bench').calculate_metrics(state('2', '2', {'task': 'math'})).score
    assert score.status is ScoreStatus.EXCLUDED
    assert score.value == {}


@pytest.mark.parametrize(('name', 'metadata'), [
    ('gsm8k', {}), ('aime24', {}), ('docmath', {'answer_type': 'float'}),
    ('olympiad_bench', {'answer_type': 'Expression', 'is_multiple_answer': False}),
    ('agieval', {'subset': 'math'}), ('agieval', {'subset': 'gaokao-mathcloze'}),
])
def test_invalid_reference_cannot_be_recovered_by_judge(
    name: str, metadata: dict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = get_benchmark(name, TaskConfig(datasets=[name], judge={'strategy': 'llm_recall'}))

    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail('Unavailable mathematical references must not request judge recall')

    monkeypatch.setattr(benchmark, 'score_with_judge_contracts', unexpected)
    score = benchmark.calculate_metrics(state(r'\boxed{0}', r'\frac{', metadata)).score
    assert score.status is ScoreStatus.EXCLUDED
    assert score.value == {}
    assert score.metadata['metric_unavailable'] is True


def test_execution_failure_is_excluded_before_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    from evalscope.metrics.math import parser
    from evalscope.metrics.math.contracts import MathEvaluationError

    benchmark = get_benchmark('gsm8k', TaskConfig(datasets=['gsm8k'], judge={'strategy': 'llm_recall'}))

    def fail(*args: object, **kwargs: object) -> None:
        raise MathEvaluationError('verification failed')

    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail('Execution failure must not request judge recall')

    monkeypatch.setattr(parser, 'compare_answers', fail)
    monkeypatch.setattr(benchmark, 'score_with_judge_contracts', unexpected)
    score = benchmark.calculate_metrics(state(r'\boxed{0}', '0')).score
    assert score.status is ScoreStatus.EXCLUDED and score.value == {}


def test_valid_incorrect_answer_still_allows_judge_recall(monkeypatch: pytest.MonkeyPatch) -> None:
    from evalscope.api.metric import Score

    benchmark = get_benchmark('gsm8k', TaskConfig(datasets=['gsm8k'], judge={'strategy': 'llm_recall'}))
    requested = []

    def recall(*args: object, **kwargs: object) -> Score:
        requested.append(True)
        return Score(value={'accuracy': 1}, main_score_name='accuracy')

    monkeypatch.setattr(benchmark, 'score_with_judge_contracts', recall)
    score = benchmark.calculate_metrics(state(r'\boxed{2}', '1')).score
    assert requested == [True]
    assert score.main_value == 1


def test_math_runs_in_threads_without_signal_timers_or_subprocesses(monkeypatch: pytest.MonkeyPatch) -> None:
    import signal
    import subprocess

    from evalscope.metrics.math.parser import parse_digits

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail('Direct mathematical scoring must not create subprocesses or install signal timers')

    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    monkeypatch.setattr(signal, 'signal', forbidden)
    monkeypatch.setattr(signal, 'alarm', forbidden, raising=False)

    def evaluate(_: int) -> tuple[bool, str, float | None]:
        return math_equal('18%', '.18'), extract_answer(r'\boxed{\frac{3}{4}}'), parse_digits(r'\frac{3}{4}')

    with ThreadPoolExecutor(max_workers=4) as executor:
        assert list(executor.map(evaluate, range(12))) == [(True, r'\frac{3}{4}', 0.75)] * 12


def test_upstream_verification_errors_remain_excluded_before_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    import math_verify

    benchmark = get_benchmark('gsm8k', TaskConfig(datasets=['gsm8k'], judge={'strategy': 'llm_recall'}))

    def fail(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError('upstream verification failed')

    def unexpected(*args: Any, **kwargs: Any) -> None:
        pytest.fail('Failed mathematical execution must not request judge recall')

    monkeypatch.setattr(math_verify, 'verify', fail)
    monkeypatch.setattr(benchmark, 'score_with_judge_contracts', unexpected)
    score = benchmark.calculate_metrics(state(r'\boxed{2}', '2')).score
    assert score.status is ScoreStatus.EXCLUDED and score.value == {}
    assert score.metadata['metric_unavailable'] is True
