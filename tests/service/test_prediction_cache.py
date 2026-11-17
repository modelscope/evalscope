"""Prediction fingerprints change with their source files and drive distinct ETags."""

from pathlib import Path

import pytest

from evalscope.service.prediction_cache import prediction_etag


def test_creating_a_missing_source_changes_the_etag(tmp_path: Path) -> None:
    review = tmp_path / 'review.jsonl'
    prediction = tmp_path / 'prediction.jsonl'
    before = prediction_etag('key', str(review), str(prediction))
    review.write_text('{}\n', encoding='utf-8')
    assert prediction_etag('key', str(review), str(prediction)) != before


@pytest.mark.parametrize('source', ['review', 'prediction'])
def test_rewriting_a_source_changes_the_etag(tmp_path: Path, source: str) -> None:
    review = tmp_path / 'review.jsonl'
    prediction = tmp_path / 'prediction.jsonl'
    review.write_text('{}\n', encoding='utf-8')
    prediction.write_text('{}\n', encoding='utf-8')

    before = prediction_etag('key', str(review), str(prediction))
    (tmp_path / f'{source}.jsonl').write_text('{}\n{}\n', encoding='utf-8')

    after = prediction_etag('key', str(review), str(prediction))
    assert after != before


def test_etag_is_stable_per_identity_and_sources(tmp_path: Path) -> None:
    review = tmp_path / 'review.jsonl'
    review.write_text('{}\n', encoding='utf-8')
    prediction = str(tmp_path / 'prediction.jsonl')
    first = prediction_etag('key-a', str(review), prediction)
    assert first == prediction_etag('key-a', str(review), prediction)
    assert first != prediction_etag('key-b', str(review), prediction)
