"""Unit tests for TIR-Bench multiple-choice answer extraction."""
import pytest

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, ContentText
from evalscope.api.model import ModelOutput
from evalscope.api.registry import BENCHMARK_REGISTRY, get_benchmark
from evalscope.benchmarks.tir_bench.tir_bench_adapter import TIRBenchAdapter
from evalscope.benchmarks.tir_bench.utils import extract_mcq_answer, judge_choice
from evalscope.config import TaskConfig

PROMPT = 'What is the reading on the gauge?\nA. 10\nB. 20\nC. 30\nD. 40'


def test_tir_bench_evaluation_version_reflects_scoring_change() -> None:
    metadata = BENCHMARK_REGISTRY['tir_bench']

    assert metadata.data_adapter is TIRBenchAdapter
    assert metadata.evaluation_version == 'v1.3'


@pytest.mark.parametrize(
    ('prediction', 'expected'),
    [
        # The pronoun 'I' is not one of the options.
        ('The correct option is C. I double-checked the dial markings.', 'C'),
        ('Option B matches the needle position. I am fairly confident.', 'B'),
        # 'H' is a valid letter in general, but not an option of this question.
        ('The reading is B, not the value at mark H.', 'B'),
    ],
)
def test_bare_letter_must_be_an_offered_option(prediction: str, expected: str) -> None:
    extracted = extract_mcq_answer(prediction, PROMPT)

    assert extracted == expected
    assert judge_choice(extracted, expected, PROMPT) == 1.0


@pytest.mark.parametrize(
    ('prediction', 'expected'),
    [
        ('Answer: A looks close, but the needle is past 25.\nAnswer: C', 'C'),
        ('The answer is A at first glance. Looking again, the answer is C.', 'C'),
        ('A is correct? No, C is correct.', 'C'),
    ],
)
def test_last_explicit_answer_marker_wins(prediction: str, expected: str) -> None:
    assert extract_mcq_answer(prediction) == expected


@pytest.mark.parametrize(
    ('prediction', 'expected'),
    [
        ('C', 'C'),
        ('Answer: B', 'B'),
        ('The answer is D.', 'D'),
        (r'The reading is 30, so \boxed{C}.', 'C'),
        ('(A) is too low; the needle is at (C).', 'C'),
        ('The needle points at 30, so C.', 'C'),
    ],
)
def test_clear_answers_are_unchanged(prediction: str, expected: str) -> None:
    assert extract_mcq_answer(prediction, PROMPT) == expected


@pytest.mark.parametrize(
    'prompt',
    [
        'What proportion of the image is the man?\nA: 65%\nB: 73%\nC: 81%',
        'What is the voltmeter reading in kV?\nA 200-205\nB 205-210\nC 210-215',
        'How many edges are there in this polygon?\nA.12\nB.13\nC.14',
        'Does the bar have a uniform color? Select from the following choices. (A) Hard to tell (B) Yes (C) No',
    ],
)
def test_option_labels_are_read_from_dataset_prompt_styles(prompt: str) -> None:
    assert extract_mcq_answer('It is B. I checked twice.', prompt) == 'B'


def test_all_letters_count_when_the_prompt_lists_no_options() -> None:
    assert extract_mcq_answer('It is H.', 'Which option is correct?') == 'H'


def test_reply_without_an_offered_letter_is_not_credited() -> None:
    # Raw text would reach the nearest-letter fallback in judge_choice, which picks 'A' here.
    prediction = 'After zooming in, I still cannot read the dial.'

    assert judge_choice(extract_mcq_answer(prediction, PROMPT), 'A', PROMPT) == 0.0


def test_adapter_passes_the_prompt_to_answer_extraction() -> None:
    adapter = get_benchmark('tir_bench', config=TaskConfig(datasets=['tir_bench'], judge={'strategy': 'rule'}))
    sample = Sample(
        input=[ChatMessageUser(content=[ContentText(text=PROMPT)])],
        target='C',
        metadata={'task': 'instrument', 'meta_data': {}, 'id': 1},
    )
    state = TaskState(
        model='offline',
        sample=sample,
        output=ModelOutput.from_content('offline', 'The correct option is C. I double-checked the dial markings.'),
        completed=True,
    )

    score = adapter.calculate_metrics(state).score

    assert score.extracted_prediction == 'C'
    assert score.main_value == 1
