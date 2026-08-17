"""Unit tests for SLAKE answer parsing and normalization.

Scoring is normalized exact match, so a regression here fails silently: a mis-parsed or
over-normalized answer still looks like a plausible short answer and the run completes with a
wrong score.
"""
from evalscope.benchmarks.slake.slake_adapter import EN_PROMPT_TEMPLATE, ZH_PROMPT_TEMPLATE
from evalscope.benchmarks.slake.utils import normalize_answer, parse_answer


def test_parse_answer_reads_the_marker_line() -> None:
    assert parse_answer('The scan is axial.\nANSWER: CT') == 'CT'
    assert parse_answer('answer: 胸腔') == '胸腔'
    assert parse_answer('ANSWER: "Lung"') == 'Lung'


def test_parse_answer_ignores_a_restated_instruction() -> None:
    """The prompt contains the marker, so an echoed instruction must not shadow the answer."""
    prediction = f'{EN_PROMPT_TEMPLATE.format(question="Which organ is shown?")}\nANSWER: Lung'
    assert parse_answer(prediction) == 'Lung'


def test_parse_answer_keeps_one_line() -> None:
    assert parse_answer('ANSWER: Chest\n(the lower ribs are visible too)') == 'Chest'


def test_parse_answer_falls_back_to_the_whole_reply() -> None:
    # Models frequently answer with the bare short answer the prompt asks for.
    assert parse_answer('CT') == 'CT'
    assert parse_answer('') == ''
    assert parse_answer(None) == ''


def test_normalize_answer_ignores_case_and_punctuation() -> None:
    assert normalize_answer('Lung.') == 'lung'
    assert normalize_answer('CT (Computed Tomography)') == 'ct'
    assert normalize_answer('腹部。') == '腹部'
    assert normalize_answer('Lung, Spinal Cord') == normalize_answer('lung  spinal cord')


def test_normalize_answer_unifies_yes_no_surface_forms() -> None:
    # The Chinese references use several polarity spellings for the same closed-ended answer.
    for yes in ['Yes', 'yes.', '是的', '是', '有', '包含', '可以', '存在']:
        assert normalize_answer(yes) == 'yes', yes
    for no in ['No', '不是', '否', '没有', '不包含', '不可以', '不正常']:
        assert normalize_answer(no) == 'no', no


def test_normalize_answer_unifies_xray_spellings() -> None:
    # Modality references stay in English even for Chinese questions.
    for spelling in ['X-Ray', 'x-ray', 'Xray', 'X光', 'X射线']:
        assert normalize_answer(spelling) == 'xray', spelling


def test_normalize_answer_keeps_distinct_answers_distinct() -> None:
    assert normalize_answer('健康') != normalize_answer('是的')
    assert normalize_answer('异常') != normalize_answer('不是')
    assert normalize_answer('T2-weighted') != normalize_answer('T2')


def test_prompts_request_the_answer_marker() -> None:
    for template in [EN_PROMPT_TEMPLATE, ZH_PROMPT_TEMPLATE]:
        prompt = template.format(question='Which organ is shown?')
        assert prompt.startswith('Which organ is shown?')
        assert 'ANSWER:' in prompt
