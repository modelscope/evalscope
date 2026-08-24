import pytest

from evalscope.backend.rag_eval.backend_manager import require_sentence_transformers_logit_score


def test_pre_logit_score_version_rejected():
    with pytest.raises(ImportError, match='LogitScore'):
        require_sentence_transformers_logit_score('5.3.0')


def test_logit_score_version_accepted():
    require_sentence_transformers_logit_score('5.4.0')


def test_logit_score_prerelease_accepted():
    # A release-candidate/dev build of 5.4.0 already ships LogitScore; only the
    # numeric release tuple should gate this, not a direct Version() comparison
    # (Version('5.4.0.dev0') < Version('5.4.0') would reject it wrongly).
    require_sentence_transformers_logit_score('5.4.0.dev0')


def test_newer_version_accepted():
    require_sentence_transformers_logit_score('5.6.0')


def test_unparseable_version_rejected():
    with pytest.raises(ImportError, match='LogitScore'):
        require_sentence_transformers_logit_score('not-a-version')
