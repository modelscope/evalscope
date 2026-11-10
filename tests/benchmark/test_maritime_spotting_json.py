"""maritime_ocr_bench: an example object before the spotting payload must not break parsing."""

import pytest

pytest.importorskip('shapely')  # maritime_ocr_bench extra

from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.maritime_ocr_bench.scoring.spotting_json_quantization_eval import normalize_spotting_payload
from evalscope.config import TaskConfig

V2_PAYLOAD = '{"images": [{"content": [{"type": "text", "text": "A1", "pos": [0, 0, 10, 0, 10, 10, 0, 10]}]}]}'


def test_example_object_before_the_payload_is_ignored() -> None:
    text = f'Example: {{"images": []}}. Answer: {V2_PAYLOAD}'

    payload = normalize_spotting_payload(text, expected_format='v2')

    assert payload['parse_ok'] is True
    assert payload['parse_error'] is None
    assert payload['items'] == [{'text': 'A1', 'quad': [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]}]


def test_payload_is_detected_automatically() -> None:
    payload = normalize_spotting_payload(f'Example: {{"images": []}}. Answer: {V2_PAYLOAD}')

    assert payload['detected_format'] == 'v2'
    assert payload['parse_ok'] is True


def test_text_without_a_payload_reports_no_object() -> None:
    payload = normalize_spotting_payload('no payload here', expected_format='v2')

    assert payload['parse_ok'] is False
    assert payload['parse_error'] == 'no json object found'


def test_evaluation_version_tracks_spotting_json_extraction() -> None:
    adapter = get_benchmark('maritime_ocr_bench', TaskConfig(model='mock', datasets=['maritime_ocr_bench']))

    assert adapter.benchmark_meta.evaluation_version == 'v1.2'
