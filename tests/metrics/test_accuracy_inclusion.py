import pytest

from evalscope.api.evaluator import ReviewResult, Target, TaskState
from evalscope.api.model import ModelOutput
from evalscope.api.registry import BENCHMARK_REGISTRY
from evalscope.benchmarks.mmlu_redux.mmlu_redux_adapter import MMLUReduxAdapter
from evalscope.benchmarks.trivia_qa.trivia_qa_adapter import TriviaQaAdapter
from evalscope.config import TaskConfig
from evalscope.metrics.nlp.metrics import Accuracy, MultiChoiceAcc
from evalscope.report.data_frames import _build_prediction_row


@pytest.mark.parametrize(
    ('prediction', 'reference', 'expected'),
    [
        ('A', ['A', 'B'], 1.0),
        ('B', ['A', 'B'], 1.0),
        ('AB', ['A', 'B'], 0.0),
        ('C', ['A', 'B'], 0.0),
        (' a\n', [' A ', 'B'], 1.0),
        ('PARIS', ['London', 'Paris'], 1.0),
        ('Paris', ['London', ' PARIS\t'], 1.0),
        ('York', ['New York'], 0.0),
        ('York', 'New York', 0.0),
        ('A', 'AB', 0.0),
        (' New York\n', 'new york', 1.0),
        ('New York', 'New York', 1.0),
        ('', [''], 0.0),
        (' ', [' '], 0.0),
        (' ', ' ', 0.0),
        ('', '', 0.0),
        ('A', [], 0.0),
    ],
)
def test_inclusion_matches_one_complete_normalized_answer(
    prediction: str, reference: str | list[str], expected: float
) -> None:
    assert Accuracy(allow_inclusion=True).apply([prediction], [reference]) == [expected]


def test_inclusion_scores_mixed_reference_types_in_order() -> None:
    assert Accuracy(allow_inclusion=True).apply(
        [' PARIS ', 'York', ' b '], [['London', 'Paris'], 'New York', ['A', 'B']]
    ) == [1.0, 0.0, 1.0]


def test_default_accuracy_keeps_exact_match_behavior() -> None:
    assert Accuracy().apply([' PARIS ', 'York', ''], ['paris', 'New York', '']) == [1.0, 0.0, 1.0]


def test_target_normalizes_alternatives_and_preserves_empty_targets() -> None:
    target = Target([' Paris ', 'Paris', 'London', ''])

    assert target.values == ('Paris', 'London')
    assert target.display == 'Paris\nLondon'
    assert Target(['', ' ']).values == ()


def test_metrics_prepare_references_explicitly() -> None:
    alternatives = Target(['A', 'B'])

    assert Accuracy(allow_inclusion=True).prepare_reference(alternatives) == ['A', 'B']
    assert MultiChoiceAcc().prepare_reference(alternatives) == 'AB'
    with pytest.raises(ValueError, match='requires one reference'):
        Accuracy().prepare_reference(alternatives)


def test_mmlu_redux_accepts_each_correct_single_choice() -> None:
    adapter = MMLUReduxAdapter(benchmark_meta=BENCHMARK_REGISTRY['mmlu_redux'])
    sample = adapter.record_to_sample(
        {
            'question': 'Which option is acceptable?',
            'choices': ['First', 'Second', 'Third', 'Fourth'],
            'answer': 0,
            'error_type': 'multiple_correct_answers',
            'correct_answer': '0 or 1',
        }
    )
    assert sample.target == ['A', 'B']
    assert Accuracy(allow_inclusion=True).apply(['A', 'B', 'AB'], [sample.target] * 3) == [1.0, 1.0, 0.0]


def test_mmlu_redux_scores_each_correct_choice_through_task_state() -> None:
    adapter = MMLUReduxAdapter(
        benchmark_meta=BENCHMARK_REGISTRY['mmlu_redux'],
        task_config=TaskConfig(datasets=['mmlu_redux']),
    )
    sample = adapter.record_to_sample(
        {
            'question': 'Which option is acceptable?',
            'choices': ['First', 'Second', 'Third', 'Fourth'],
            'answer': 0,
            'error_type': 'multiple_correct_answers',
            'correct_answer': '0 or 1',
        }
    )
    sample.id = 0
    state = TaskState(
        model='mock',
        sample=sample,
        output=ModelOutput.from_content('mock', 'B'),
        completed=True,
    )

    sample_score = adapter.calculate_metrics(state)

    assert sample_score.score.value['accuracy'] == 1.0


def test_trivia_qa_accepts_normalized_aliases() -> None:
    adapter = TriviaQaAdapter(benchmark_meta=BENCHMARK_REGISTRY['trivia_qa'])
    sample = adapter.record_to_sample(
        {
            'question': 'What city?',
            'question_id': 'example',
            'answer': {'aliases': ['New York City'], 'normalized_aliases': ['new york city', 'nyc']},
            'entity_pages': {'wiki_context': 'A city in the United States.'},
        }
    )
    assert Accuracy(allow_inclusion=True).apply([' NYC\n', 'York'], [sample.target] * 2) == [1.0, 0.0]


def test_trivia_qa_preserves_aliases_through_prediction_rows() -> None:
    adapter = TriviaQaAdapter(
        benchmark_meta=BENCHMARK_REGISTRY['trivia_qa'],
        task_config=TaskConfig(datasets=['trivia_qa']),
    )
    sample = adapter.record_to_sample(
        {
            'question': 'What city?',
            'question_id': 'example',
            'answer': {'aliases': ['New York City'], 'normalized_aliases': ['new york city', 'nyc']},
            'entity_pages': {'wiki_context': 'A city in the United States.'},
        }
    )
    sample.id = 0
    state = TaskState(
        model='mock',
        sample=sample,
        output=ModelOutput.from_content('mock', 'NYC'),
        completed=True,
    )

    sample_score = adapter.calculate_metrics(state)
    review_result = ReviewResult.from_score_state(sample_score, state)
    prediction_row = _build_prediction_row(review_result, None, [])

    assert sample_score.score.value['accuracy'] == 1.0
    assert review_result.target == ['New York City', 'new york city', 'nyc']
    assert prediction_row['Gold'] == ['New York City', 'new york city', 'nyc']


@pytest.mark.parametrize('benchmark_name', ['mmlu_redux', 'trivia_qa'])
def test_inclusion_benchmark_evaluation_version(benchmark_name: str) -> None:
    assert BENCHMARK_REGISTRY[benchmark_name].evaluation_version == 'v1.1'
