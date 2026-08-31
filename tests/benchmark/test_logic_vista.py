"""Unit tests for the LogicVista adapter's answer normalization and prompt safety.

LogicVista labels are read from an `ANSWER:` line and a few items are multi-select, so a
regression in target normalization or in the label alphabet silently turns correct answers
into misses without raising anything.
"""
import io
from typing import Any, Dict, List

from PIL import Image

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.logic_vista.logic_vista_adapter import OPTION_LABELS, PROMPT_TEMPLATE
from evalscope.config import TaskConfig


def _adapter():
    task_cfg = TaskConfig(model='mock', datasets=['logic_vista'])
    return get_benchmark('logic_vista', task_cfg)


def _image_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new('RGB', (2, 2)).save(buffer, format='PNG')
    return buffer.getvalue()


def _record(**overrides: Any) -> Dict[str, Any]:
    record = {
        'id': 'v1_0',
        'question': 'Which figure completes the pattern?',
        'answer': 'C',
        'skill': ['inductive'],
        'image': {
            'bytes': _image_bytes(),
            'path': 'v1_0.png'
        },
    }
    record.update(overrides)
    return record


def _extract(completion: str, target: str = 'C') -> str:
    adapter = _adapter()
    sample = adapter.record_to_sample(_record(answer=target))
    state = TaskState(model='mock', sample=sample, output=ModelOutput.from_content('mock', completion))
    return adapter.extract_answer(prediction=completion, task_state=state)


def test_instruction_contains_nothing_the_answer_parser_can_match() -> None:
    """A model echoing the format instruction must not have that echo scored as a valid label.

    `parse_answers` falls back to the last upper-case character when no `ANSWER:` line is
    present, so the instruction added here must not spell out a label sequence. The fallback
    still reads the question text, which some items label explicitly (e.g. 'What choice
    (A, B, C, or D) ...'), so an answerless reply that only restates the question is scored as
    a lenient guess; that leniency belongs to the shared parser rather than this benchmark.
    """
    instruction = PROMPT_TEMPLATE.format(question='')
    assert 'ANSWER: [LETTER]' in instruction
    assert _extract(instruction) not in OPTION_LABELS


def test_single_label_is_extracted() -> None:
    assert _extract('The square moves right.\nANSWER: C') == 'C'
    assert _extract('Step 1 ...\n\n**ANSWER: E**', target='E') == 'E'


def test_multi_select_target_and_prediction_share_a_normalized_form() -> None:
    """Ground truth 'B, D' and a prediction of 'DB' must compare equal."""
    adapter = _adapter()
    sample = adapter.record_to_sample(_record(answer='B, D'))
    assert sample.target == 'BD'
    assert _extract('Both fit.\nANSWER: DB', target='B, D') == 'BD'


def test_labels_outside_the_alphabet_are_not_extracted() -> None:
    """A hallucinated label must yield no prediction rather than a confident wrong one."""
    assert OPTION_LABELS[-1] == 'I'
    assert _extract('ANSWER: Z') == ''


def test_bracketed_label_is_read_instead_of_a_letter_from_the_option_text() -> None:
    """Real replies quote the label as printed in the image and append the option text.

    `parse_answers` reads the wrapped label, so the last capital letter of the reply ('F'
    below) is not mistaken for the answer. All four strings below are verbatim answer lines
    from a qwen-vl-plus run and are kept as regression coverage for this benchmark, whose
    labels are only visible in the image.
    """
    assert _extract('Therefore:\n\nANSWER: (D) B, D and F are turning anticlockwise', target='D') == 'D'
    assert _extract('Thus:\n\nANSWER: (E)', target='E') == 'E'
    assert _extract('So:\n\nANSWER: (E) None', target='E') == 'E'
    assert _extract('Hence:\n\nANSWER: (B) B and C, then A and D', target='B') == 'B'
    # Multi-select stays multi-select
    assert _extract('ANSWER: (B, D)', target='B, D') == 'BD'


def test_non_label_parentheses_yield_no_answer() -> None:
    """Only a label-shaped bracket group is read; bracketed prose must not become a label."""
    assert _extract('ANSWER: (see the diagram above)') not in OPTION_LABELS
    assert _extract('ANSWER: [LETTER]') not in OPTION_LABELS


def test_records_without_ground_truth_are_skipped() -> None:
    adapter = _adapter()
    assert adapter.record_to_sample(_record(answer='', question='')) == []
    assert adapter.record_to_sample(_record(skill=[])) == []


def test_sample_carries_the_label_alphabet_and_the_image() -> None:
    adapter = _adapter()
    sample: Sample = adapter.record_to_sample(_record())
    assert sample.choices == OPTION_LABELS
    contents: List[Any] = sample.input[0].content
    assert contents[0].image.startswith('data:image/png;base64,')
    assert 'Which figure completes the pattern?' in contents[1].text
