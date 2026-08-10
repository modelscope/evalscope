"""Unit tests for the CC-OCR V2 scorers.

Every track uses a different metric and the answer conventions differ per scenario, so a
regression here silently shifts scores instead of raising. The expected values below were
cross-checked against the official ``src/evaluate_*.py`` scripts.
"""
import json
import os
import pytest
import tempfile

from evalscope.benchmarks.cc_ocr_v2.cc_ocr_v2_adapter import _index_images
from evalscope.benchmarks.cc_ocr_v2.utils import (
    parse_object_list,
    parse_point_box,
    parsing_op,
    score_kie,
    score_object_grounding,
    score_parsing,
    score_recognition,
    score_sample,
    score_text_grounding,
    score_vqa,
    strip_code_fence,
)


def test_strip_code_fence_only_unwraps_bare_and_json_fences() -> None:
    """Regression: the official scorers accept only bare and ``json`` fences.

    An ``html`` or ``latex`` fence is not recognised, so the tag itself leaks into the scored
    text and costs the prediction points -- which is why the parsing track unwraps those
    fences explicitly instead of relying on this helper.
    """
    assert strip_code_fence('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_code_fence('```\nplain\n```') == 'plain'
    assert strip_code_fence('```html\n<p>x</p>\n```') == 'html\n<p>x</p>'
    assert strip_code_fence('```latex\n\\alpha\n```') == 'latex\n\\alpha'
    # A fence that is not the whole reply is left alone
    assert strip_code_fence('text ```json\n1\n``` tail').startswith('text ```json')


def test_score_recognition_tokenizes_per_language() -> None:
    # English scene text: word level, alphanumeric only for multi_scene scenarios
    scenario = 'multi_scene_ocr_document_text_CORD_100'
    assert score_recognition('Lemon Tea 25.000', 'Lemon Tea 25.000', scenario) == pytest.approx(1.0)
    # Punctuation is dropped for multi_scene word-level data, so it cannot cost points
    assert score_recognition('Lemon, Tea 25000', 'Lemon Tea 25.000', scenario) == pytest.approx(1.0)
    # Chinese ground truth switches to character level
    zh_scenario = 'multi_scene_ocr_document_text_zh_doc_100'
    assert score_recognition('合计金额', '合计金额', zh_scenario) == pytest.approx(1.0)
    assert 0.0 < score_recognition('合计', '合计金额', zh_scenario) < 1.0
    # Non-multi_scene scenarios keep punctuation, so the token no longer matches
    assert score_recognition('bonjour!', 'bonjour', 'multi_lan_ocr_French_French_20') == 0.0


def test_score_kie_matches_fields() -> None:
    reference = json.dumps({'Name': '周志强', 'Total Amount': '1674.00'}, ensure_ascii=False)
    assert score_kie(reference, reference) == pytest.approx(1.0, abs=1e-5)
    assert score_kie(f'```json\n{reference}\n```', reference) == pytest.approx(1.0, abs=1e-5)
    # One field right, one wrong: that is 1 true positive plus a false positive and a false
    # negative, so F1 = 1 / (1 + 2/2) = 0.5
    half = json.dumps({'Name': '周志强', 'Total Amount': '0.00'}, ensure_ascii=False)
    assert score_kie(half, reference) == pytest.approx(0.5, abs=1e-5)
    # Unparseable prediction earns nothing but must not raise
    assert score_kie('I cannot read this image.', reference) == 0.0


def test_score_vqa_short_and_long_answers() -> None:
    # Short English answer: substring match, no partial credit
    assert score_vqa('The total is 1,234 dollars.', '1,234') == 1.0
    assert score_vqa('The total is 9,999 dollars.', '1,234') == 0.0
    assert score_vqa('total two thousand', 'total one thousand') == 0.0
    # Short Chinese answers fall back to ANLS -- the one asymmetry in the official scorers
    assert score_vqa('合计金额一千圆', '合计金额一千元') == pytest.approx(6 / 7)
    # Long answers are scored by ANLS with a 0.5 floor
    long_answer = 'the quick brown fox jumps over the lazy dog'
    assert score_vqa(long_answer, long_answer) == 1.0
    assert score_vqa('completely different words here entirely', long_answer) == 0.0
    # A list of acceptable answers takes the best match
    assert score_vqa('it is 42', json.dumps(['7', '42'])) == 1.0
    assert score_vqa('', '42') == 0.0


def test_parse_point_box_reads_the_first_tuple() -> None:
    assert parse_point_box('(845, 163, 1102, 307)') == [845.0, 163.0, 1102.0, 307.0]
    assert parse_point_box('The box is [0.1, 0.2, 0.3, 0.4].') == [0.1, 0.2, 0.3, 0.4]
    assert parse_point_box('I cannot locate that text.') is None


def test_score_text_grounding_without_image_compares_as_is() -> None:
    reference = '[100.0, 100.0, 200.0, 200.0]'
    assert score_text_grounding('(100, 100, 200, 200)', reference, '') == 1.0
    assert score_text_grounding('no box here', reference, '') == 0.0
    # A malformed reference cannot be scored
    assert score_text_grounding('(100, 100, 200, 200)', 'not a box', '') == 0.0


def test_score_text_grounding_maps_normalized_predictions_to_pixels(tmp_path) -> None:
    """Regression: nothing else exercises the rescaling branch.

    Every model observed so far answered in absolute pixels, so a run never reaches the
    ``scale=1000`` path -- a silent break here would zero out the score of every model that
    actually obeys the prompt, while leaving the rest of the suite green.
    """
    from PIL import Image

    image_path = str(tmp_path / 'page.png')
    Image.new('RGB', (1000, 500), 'white').save(image_path)

    reference = '[400.0, 100.0, 600.0, 300.0]'  # pixels
    # The same box expressed on the 0-1000 grid the prompt asks for, and unit normalized
    assert score_text_grounding('(400, 200, 600, 600)', reference, image_path) == pytest.approx(1.0)
    assert score_text_grounding('(0.4, 0.2, 0.6, 0.6)', reference, image_path) == pytest.approx(1.0)
    # Absolute pixels are read as 0-1000 values and therefore land elsewhere, matching the
    # official scorer, which normalizes the same way.
    assert score_text_grounding('(400, 100, 600, 300)', reference, image_path) < 0.5


def test_parse_object_list_accepts_common_bbox_notations() -> None:
    payload = json.dumps([
        {'bbox_2d': [1, 2, 3, 4], 'label': 'DATE'},
        {'box_2d': {'xmin': 5, 'ymin': 6, 'xmax': 7, 'ymax': 8}, 'category_name': 'CITY'},
        {'bbox': {'x': 10, 'y': 10, 'width': 5, 'height': 5}, 'label': 'ZIP'},
        {'label': 'no box'},
    ])
    assert parse_object_list(payload) == [[1, 2, 3, 4], [5, 6, 7, 8], [10, 10, 15, 15]]
    assert parse_object_list('not json at all') is None


def test_score_object_grounding_averages_matched_iou() -> None:
    reference = json.dumps([
        {'bbox_2d': [0, 0, 10, 10], 'label': 'A'},
        {'bbox_2d': [100, 100, 110, 110], 'label': 'B'},
    ])
    assert score_object_grounding(reference, reference, '') == 1.0
    # One of two boxes recovered -> mean IoU over ground-truth boxes is 0.5
    partial = json.dumps([{'bbox_2d': [0, 0, 10, 10], 'label': 'A'}])
    assert abs(score_object_grounding(partial, reference, '') - 0.5) < 1e-9
    assert score_object_grounding('sorry, nothing found', reference, '') == 0.0


def test_parsing_op_detection_covers_every_scenario_family() -> None:
    assert parsing_op('doc_parsing_formula_formula_handwriting_100') == 'formula'
    assert parsing_op('doc_parsing_molecular_molecular_handwriting_100') == 'molecular'
    assert parsing_op('doc_parsing_custom_info_board_8') == 'custom'
    assert parsing_op('doc_parsing_table_table_photo_150') == 'table'
    assert parsing_op('doc_parsing_doc_doc_scan_150') == 'doc'


def test_score_parsing_per_op() -> None:
    formula = r'\frac{1}{2}'
    assert score_parsing(f'```latex\n{formula}\n```', formula, 'doc_parsing_formula_formula_x_1') == 1.0
    smiles = 'O=C(N)CC'
    assert score_parsing(f'<smiles>{smiles}</smiles>', smiles, 'doc_parsing_molecular_molecular_x_1') == 1.0

    table = '<table><tr><td>a</td><td>b</td></tr></table>'
    assert score_parsing(f'```html\n{table}\n```', table, 'doc_parsing_table_table_photo_1') == 1.0
    assert score_parsing('no table at all', table, 'doc_parsing_table_table_photo_1') == 0.0

    doc = r'\section{Report} Total: 42'
    assert score_parsing(doc, doc, 'doc_parsing_doc_doc_scan_1') == 1.0
    assert score_parsing('', doc, 'doc_parsing_doc_doc_scan_1') == 0.0


def test_malformed_table_prediction_scores_zero_instead_of_raising() -> None:
    """A garbled table can leave a non-numeric ``colspan``; the official TEDS tree builder
    raises ``ValueError`` on it, which would abort the whole run instead of scoring 0."""
    reference = '<table><tr><td colspan="2">a</td></tr></table>'
    prediction = '<table><tr><td colspan="a b">x</td></tr></table>'
    assert score_parsing(prediction, reference, 'doc_parsing_table_table_photo_1') == 0.0


def test_score_sample_dispatches_on_metadata() -> None:
    metadata = {
        'task': 'recognition',
        'sub_task': 'natural_scene_recognition',
        'scenario': 'multi_scene_ocr_document_text_CORD_100',
        'image_paths': [],
    }
    assert score_sample('TOTAL 30', 'TOTAL 30', metadata) == pytest.approx(1.0)

    metadata = {'task': 'grounding', 'sub_task': 'text_grounding', 'scenario': 'x', 'image_paths': []}
    assert score_sample('(1, 1, 2, 2)', '[1, 1, 2, 2]', metadata) == 1.0


def test_index_images_handles_single_files_and_page_directories() -> None:
    with tempfile.TemporaryDirectory() as root:
        open(os.path.join(root, 'aaa.jpg'), 'wb').close()
        open(os.path.join(root, 'notes.txt'), 'wb').close()
        pages = os.path.join(root, 'bbb')
        os.makedirs(pages)
        for name in ('page_10.jpg', 'page_2.jpg', 'page_1.jpg'):
            open(os.path.join(pages, name), 'wb').close()

        index = _index_images(root)
        assert index['aaa'] == [os.path.join(root, 'aaa.jpg')]
        # Multi-page documents keep natural page order, not lexicographic order
        assert [os.path.basename(path) for path in index['bbb']] == ['page_1.jpg', 'page_2.jpg', 'page_10.jpg']
        assert 'notes' not in index
        assert _index_images(os.path.join(root, 'missing')) == {}
