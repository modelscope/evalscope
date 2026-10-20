import json
from pathlib import Path
from typing import Any, Dict

import pytest

import evalscope  # noqa: F401  # imported for benchmark and metric registration side effects
from evalscope.api.benchmark import DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import SampleScore
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.finqa.utils import eval_program, parse_program, program_tokenization, score_program
from evalscope.config import TaskConfig

TABLE = [['', '2019', '2018'], ['net revenue', '$ 5829', '$ 5735'], ['operating expenses', '1,200', '(300)']]


def _record(program: str, answer: Any, uid: str = 'ETR/2016/page_23.pdf-2') -> Dict[str, Any]:
    return {
        'id': uid,
        'pre_text': ['entergy corporation management discussion .', 'results of operations .'],
        'post_text': ['see note 2 .'],
        'table': TABLE,
        'qa': {'question': 'what is the change in net revenue?', 'program': program, 'exe_ans': answer},
    }


def _adapter(**dataset_args: Any) -> DefaultDataAdapter:
    args = {'finqa': dataset_args} if dataset_args else {}
    return get_benchmark('finqa', TaskConfig(datasets=['finqa'], dataset_args=args))


def _state_score(adapter: DefaultDataAdapter, sample: Sample, completion: str):
    state = TaskState(
        model='mock',
        sample=sample,
        messages=[],
        output=ModelOutput.from_content(model='mock', content=completion),
        completed=True,
    )
    return adapter.calculate_metrics(state).score


def test_registered_meta() -> None:
    adapter = _adapter()

    assert adapter.subset_list == ['default']
    assert adapter.metric_list == ['execution_accuracy', 'program_accuracy']


def test_load_from_local_directory(tmp_path: Path) -> None:
    records = [_record('subtract(5829, 5735)', 94.0, 'a-1'), _record('greater(5829, 5735)', 'yes', 'a-2')]
    (tmp_path / 'test.json').write_text(json.dumps(records), encoding='utf-8')
    data = _adapter(dataset_id=str(tmp_path)).load_dataset()

    assert len(data['default']) == 2
    sample = data['default'][0]
    assert sample.target == 'subtract(5829, 5735)'
    content = sample.input[-1].content
    assert 'Text:\nentergy corporation management discussion .\nresults of operations .' in content
    assert 'Table:\n | 2019 | 2018\nnet revenue | $ 5829 | $ 5735' in content
    assert content.endswith('Program: <program>')


def test_tokenization() -> None:
    assert program_tokenization('subtract(5829, 5735), divide(#0, 5735)') == [
        'subtract(', '5829', '5735', ')', 'divide(', '#0', '5735', ')', 'EOF'
    ]


@pytest.mark.parametrize(
    'program, expected',
    [
        ('subtract(5829, 5735)', 94.0),
        ('subtract(5829, 5735), divide(#0, 5735)', round(94 / 5735, 5)),
        ('multiply(2, const_100)', 200.0),
        ('add(const_m1, 1)', 0.0),
        ('divide(50%, 2)', 0.25),
        ('greater(5829, 5735)', 'yes'),
        ('table_sum(net revenue, none)', 11564.0),
        ('table_average(net revenue, none)', 5782.0),
    ],
)
def test_eval_program(program: str, expected: Any) -> None:
    assert eval_program(program_tokenization(program), TABLE) == (0, expected)


@pytest.mark.parametrize(
    'program',
    [
        'foo(1, 2)',
        'subtract(1, x)',
        'divide(1, 0)',
        'table_max(missing row, none)',
        'subtract(1,2)',
        'add(#1, 2)',
        # the official code reads a cell such as "(300)" as empty, so the row has no valid numbers
        'table_average(operating expenses, none)',
    ],
)
def test_invalid_programs(program: str) -> None:
    assert eval_program(program_tokenization(program), TABLE)[0] == 1


def test_parse_program() -> None:
    assert parse_program('So...\nProgram: subtract(5829, 5735), divide(#0, 5735)') == (
        'subtract(5829, 5735), divide(#0, 5735)'
    )
    assert parse_program('Program: `subtract( 5829 ,  5735 )`.') == 'subtract(5829, 5735)'
    assert parse_program('**Program:** add(1, 2)') == 'add(1, 2)'
    assert parse_program('Program: subtract(5829,5735), divide(#0,5735)') == 'subtract(5829, 5735), divide(#0, 5735)'
    # a comma inside a number is not a separator when the step already has one
    assert parse_program('Program: add(1,234, 5)') == 'add(1,234, 5)'
    assert parse_program('Program: add(1, 2)\nProgram: add(3, 4)') == 'add(3, 4)'
    assert parse_program('The answer is 94') is None


def test_score_program() -> None:
    gold = 'subtract(5829, 5735), divide(#0, 5735)'
    answer = round(94 / 5735, 5)

    assert score_program(gold, gold, answer, TABLE) == (1.0, 1.0)
    # same result through another program: execution counts, program does not
    assert score_program('divide(94, 5735)', gold, answer, TABLE) == (1.0, 0.0)
    # a different but symbolically equal program
    assert score_program(
        'add(5829, 5735), subtract(#0, 5735), subtract(#1, 5735), divide(#2, 5735)', gold, answer, TABLE
    ) == (1.0, 1.0)
    assert score_program('subtract(5829, 5735), multiply(#0, const_100), divide(#1, 5735)', gold, answer, TABLE) == (
        0.0, 0.0
    )
    assert score_program(None, gold, answer, TABLE) == (0.0, 0.0)


def test_program_accuracy_is_symbolic() -> None:
    gold = 'add(5829, 5735)'

    assert score_program('add(5735, 5829)', gold, 11564.0, TABLE) == (1.0, 1.0)


def test_scoring_end_to_end_and_aggregation() -> None:
    adapter = _adapter()
    sample = adapter.record_to_sample(_record('subtract(5829, 5735)', 94.0))
    scores = [
        SampleScore(score=_state_score(adapter, sample, 'reasoning\nProgram: subtract(5829, 5735)'), sample_id=0),
        SampleScore(score=_state_score(adapter, sample, 'Program: subtract(5735, 5829)'), sample_id=1),
        SampleScore(score=_state_score(adapter, sample, 'The change is 94.'), sample_id=2),
        SampleScore(score=_state_score(adapter, sample, 'Program: add(94, 0)'), sample_id=3),
    ]
    assert [s.score.value for s in scores] == [
        {'execution_accuracy': 1.0, 'program_accuracy': 1.0},
        {'execution_accuracy': 0.0, 'program_accuracy': 0.0},
        {'execution_accuracy': 0.0, 'program_accuracy': 0.0},
        {'execution_accuracy': 1.0, 'program_accuracy': 0.0},
    ]
    aggregates = {agg.metric_name: agg for agg in adapter.aggregate_scores(scores)}
    assert aggregates['execution_accuracy'].score == pytest.approx(0.5)
    assert aggregates['program_accuracy'].score == pytest.approx(0.25)
