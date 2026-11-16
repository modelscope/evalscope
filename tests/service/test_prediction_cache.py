"""Prediction fingerprints change with their source files and drive distinct ETags."""

import os
from pathlib import Path

from evalscope.service.prediction_cache import prediction_etag, prediction_fingerprint


def test_missing_sources_fingerprint_as_zeros() -> None:
    fingerprint = prediction_fingerprint('/nonexistent/review.jsonl', '/nonexistent/prediction.jsonl')
    assert fingerprint == (('@review', 0, 0, 0), ('@prediction', 0, 0, 0))


def test_rewriting_a_source_changes_the_fingerprint(tmp_path: Path) -> None:
    review = tmp_path / 'review.jsonl'
    prediction = tmp_path / 'prediction.jsonl'
    review.write_text('{}\n', encoding='utf-8')
    prediction.write_text('{}\n', encoding='utf-8')

    before = prediction_fingerprint(str(review), str(prediction))

    review.write_text('{}\n{}\n', encoding='utf-8')
    os.utime(review, None)

    after = prediction_fingerprint(str(review), str(prediction))
    assert after != before
    assert after[1] == before[1]  # the untouched prediction file is unchanged


def test_etag_is_stable_per_identity_and_fingerprint(tmp_path: Path) -> None:
    review = tmp_path / 'review.jsonl'
    review.write_text('{}\n', encoding='utf-8')
    fingerprint = prediction_fingerprint(str(review), '/nonexistent/prediction.jsonl')

    first = prediction_etag('key-a', fingerprint)
    assert first == prediction_etag('key-a', fingerprint)
    assert first != prediction_etag('key-b', fingerprint)

    review.write_text('{}\n{}\n', encoding='utf-8')
    os.utime(review, None)
    assert prediction_etag('key-a', prediction_fingerprint(str(review), '/nonexistent/prediction.jsonl')) != first
