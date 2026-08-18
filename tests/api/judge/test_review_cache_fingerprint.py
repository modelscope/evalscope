"""A cached review must not be reused under a different judge configuration.

Doing so reports the old judge's scores as if the new configuration had produced them.
"""
import pytest

from evalscope.api.evaluator.cache import CacheManager, ReviewResult, compute_judge_fingerprint
from evalscope.api.metric import SampleScore, Score
from evalscope.utils.io_utils import OutputsStructure

JUDGE_ARGS = {'model_id': 'judge-a', 'generation_config': {'temperature': 0.0}}


def make_manager(tmp_path, fingerprint):
    outputs = OutputsStructure(str(tmp_path), is_make=True)
    return CacheManager(outputs=outputs, model_name='m', benchmark_name='b', judge_fingerprint=fingerprint)


def review(fingerprint):
    result = ReviewResult(index=0, sample_score=SampleScore(score=Score(value={'acc': 1.0}), sample_id=0))
    result.judge_fingerprint = fingerprint
    return result


def test_fingerprint_is_none_without_a_judge():
    assert compute_judge_fingerprint('rule', None, False) is None
    assert compute_judge_fingerprint('auto', {}, False) is None


def test_fingerprint_changes_with_the_judge_configuration():
    base = compute_judge_fingerprint('llm', JUDGE_ARGS, True)

    assert base != compute_judge_fingerprint('llm_recall', JUDGE_ARGS, True)
    assert base != compute_judge_fingerprint('llm', {**JUDGE_ARGS, 'model_id': 'judge-b'}, True)
    assert base != compute_judge_fingerprint('llm', {**JUDGE_ARGS, 'generation_config': {'temperature': 1.0}}, True)
    assert base != compute_judge_fingerprint('llm', JUDGE_ARGS, False)


def test_rotating_the_api_key_does_not_invalidate_reviews():
    """A credential change is not a scoring change, and the key must not reach the cache file."""
    with_key = compute_judge_fingerprint('llm', {**JUDGE_ARGS, 'api_key': 'k1'}, True)
    other_key = compute_judge_fingerprint('llm', {**JUDGE_ARGS, 'api_key': 'k2'}, True)

    assert with_key == other_key == compute_judge_fingerprint('llm', JUDGE_ARGS, True)


def test_matching_fingerprint_is_reused(tmp_path):
    manager = make_manager(tmp_path, 'abc123')

    manager._check_judge_fingerprint(review('abc123'), 'reviews.jsonl')


def test_mismatched_fingerprint_is_refused(tmp_path):
    manager = make_manager(tmp_path, 'abc123')

    with pytest.raises(ValueError, match='rerun_review=True'):
        manager._check_judge_fingerprint(review('def456'), 'reviews.jsonl')


def test_a_legacy_cache_without_a_fingerprint_only_warns(tmp_path):
    manager = make_manager(tmp_path, 'abc123')

    manager._check_judge_fingerprint(review(None), 'reviews.jsonl')


def test_rule_only_runs_never_check_fingerprints(tmp_path):
    manager = make_manager(tmp_path, None)

    manager._check_judge_fingerprint(review('anything'), 'reviews.jsonl')


def test_saved_reviews_carry_the_current_fingerprint(tmp_path):
    manager = make_manager(tmp_path, 'abc123')
    assert manager.judge_fingerprint == 'abc123'
