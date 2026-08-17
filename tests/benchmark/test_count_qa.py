"""Unit tests for the CountQA adapter's count parsing and scoring.

A counting answer is a bare number, so a mis-parse produces a plausible-looking prediction and
silently corrupts the score instead of raising. These cases pin the parser and the two metrics.
"""
import io
from PIL import Image
from typing import Any, Dict, List

from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.count_qa.count_qa_adapter import SYSTEM_PROMPT, parse_count
from evalscope.config import TaskConfig


def _adapter():
    task_cfg = TaskConfig(model='mock', datasets=['count_qa'])
    return get_benchmark('count_qa', task_cfg)


def _image_bytes(image_format: str = 'JPEG') -> bytes:
    buffer = io.BytesIO()
    Image.new('RGB', (2, 2)).save(buffer, format=image_format)
    return buffer.getvalue()


def _record(**overrides: Any) -> Dict[str, Any]:
    record = {
        'image': {
            'bytes': _image_bytes(),
            'path': '0.jpg'
        },
        'questions': ['How many jackets are there?', 'How many hangers are there?'],
        'answers': ['15', '4'],
        'categories': ['Clothing & Wearables'],
        'is_focused': False,
    }
    record.update(overrides)
    return record


def _score(completion: str, reference: str) -> Dict[str, float]:
    adapter = _adapter()
    extracted = adapter.extract_answer(prediction=completion, task_state=None)
    return adapter.match_score(
        original_prediction=completion,
        filtered_prediction=extracted,
        reference=reference,
        task_state=None,
    ).value


def test_system_prompt_contains_nothing_the_parser_can_match() -> None:
    """A model echoing the instruction must not have a digit from it scored as its count."""
    assert parse_count(SYSTEM_PROMPT) is None


def test_compliant_reply_is_parsed() -> None:
    """The prompt asks for a bare integer; models still wrap it in markdown or punctuation."""
    assert parse_count('12') == 12
    assert parse_count(' 7\n') == 7
    assert parse_count('**23**') == 23
    assert parse_count('9.') == 9
    assert parse_count('0') == 0


def test_verbose_reply_falls_back_to_the_first_integer() -> None:
    """Matches the paper's rewriter LLM, which extracts the first numerical value."""
    assert parse_count('There are approximately 12 boxes visible.') == 12
    assert parse_count('I count 18 tiles on that wall.') == 18


def test_reply_without_a_count_yields_no_prediction() -> None:
    """An empty or truncated reply must score 0 rather than contribute an invented count."""
    assert parse_count('') is None
    assert parse_count('I cannot tell how many there are.') is None
    assert _score('', '15') == {'acc': 0.0, 'relaxed_acc': 0.0}


def test_digit_like_characters_are_not_read_as_a_count() -> None:
    """``str.isdigit()`` accepts characters ``int()`` rejects, which would raise mid-run."""
    assert parse_count('\u00b2') is None


def test_exact_match_and_relaxed_accuracy() -> None:
    """Relaxed accuracy credits a prediction within 5% of the ground truth, exact match does not."""
    assert _score('100', '100') == {'acc': 1.0, 'relaxed_acc': 1.0}
    assert _score('103', '100') == {'acc': 0.0, 'relaxed_acc': 1.0}
    assert _score('110', '100') == {'acc': 0.0, 'relaxed_acc': 0.0}
    # Small counts leave no room for tolerance, so relaxed accuracy collapses to exact match.
    assert _score('4', '5') == {'acc': 0.0, 'relaxed_acc': 0.0}


def test_zero_ground_truth_stays_exact() -> None:
    """A ground truth of 0 has no relative tolerance to spend."""
    assert _score('0', '0') == {'acc': 1.0, 'relaxed_acc': 1.0}
    assert _score('1', '0') == {'acc': 0.0, 'relaxed_acc': 0.0}


def test_record_expands_into_one_sample_per_question() -> None:
    adapter = _adapter()
    samples = adapter.record_to_sample(_record())

    assert [sample.target for sample in samples] == ['15', '4']
    for sample, question in zip(samples, _record()['questions']):
        contents: List[Any] = sample.input[0].content
        assert contents[0].image.startswith('data:image/jpeg;base64,')
        assert contents[1].text == question
        assert sample.metadata['is_focused'] is False
        assert sample.metadata['categories'] == ['Clothing & Wearables']


def test_image_mime_type_follows_the_actual_bytes() -> None:
    """The dataset mixes JPEG and PNG images, so the data-URI header must be sniffed."""
    record = _record(image={'bytes': _image_bytes('PNG'), 'path': '0.png'})
    sample = _adapter().record_to_sample(record)[0]
    assert sample.input[0].content[0].image.startswith('data:image/png;base64,')
