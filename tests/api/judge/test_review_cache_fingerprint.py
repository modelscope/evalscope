"""Review reuse is permitted only for the exact Native judge semantics."""
import pytest

from evalscope.api.evaluator.cache import CacheManager, ReviewResult, compute_judge_fingerprint
from evalscope.api.metric import SampleScore, Score
from evalscope.api.mixin.llm_judge_mixin import LLMJudgeMixin
from evalscope.config import JudgeConfig
from evalscope.utils.io_utils import OutputsStructure


def fingerprint(**overrides):
    values = {'strategy': 'llm', 'models': [{'model_id': 'judge-a'}]}
    values.update(overrides)
    return compute_judge_fingerprint(JudgeConfig(**values), judge_revision='7')


def make_manager(tmp_path, value):
    return CacheManager(OutputsStructure(str(tmp_path), is_make=True), 'm', 'b', judge_fingerprint=value)


def review(value):
    result = ReviewResult(index=0, sample_score=SampleScore(score=Score(value={'acc': 1.0}), sample_id=0))
    result.judge_fingerprint = value
    return result


def test_rule_only_config_has_no_fingerprint():
    assert compute_judge_fingerprint(JudgeConfig(), judge_revision='1') is None


@pytest.mark.parametrize(
    'changed',
    [
        {'strategy': 'llm_recall'},
        {'models': [{'model_id': 'judge-b'}]},
        {'repeats': 2},
        {'position_swap': 'on'},
        {'aggregation': 'median'},
        {'models': [{'model_id': 'judge-a'}, {'model_id': 'judge-b'}], 'min_valid_judges': 2},
    ],
)
def test_every_scoring_semantic_changes_the_fingerprint(changed):
    assert fingerprint() != fingerprint(**changed)


def test_adapter_runtime_semantics_change_the_fingerprint():
    config = JudgeConfig(strategy='llm', models=[{'model_id': 'judge-a'}])

    assert compute_judge_fingerprint(config, '7', {'pass_threshold': 0.75}) != compute_judge_fingerprint(
        config, '7', {'pass_threshold': 0.8}
    )


def test_declared_semantic_helper_changes_the_adapter_cache_revision(tmp_path):
    helper = tmp_path / 'judge_helper.py'
    helper.write_text('prompt = "one"\n', encoding='utf-8')

    class Adapter:
        judge_revision = '1'
        judge_cache_dependencies = (str(helper), )
        judge_cache_revision = LLMJudgeMixin.judge_cache_revision

    adapter = Adapter()
    before = adapter.judge_cache_revision
    helper.write_text('prompt = "two"\n', encoding='utf-8')

    assert adapter.judge_cache_revision != before


def test_api_key_is_scrubbed_from_the_fingerprint():
    assert fingerprint(models=[{'model_id': 'judge-a', 'api_key': 'old'}]) == fingerprint(
        models=[{'model_id': 'judge-a', 'api_key': 'new'}]
    )


def test_missing_or_mismatched_fingerprint_is_refused(tmp_path):
    manager = make_manager(tmp_path, 'new')

    with pytest.raises(ValueError, match='rerun_review=True'):
        manager._check_judge_fingerprint(review(None), 'reviews.jsonl')
    with pytest.raises(ValueError, match='rerun_review=True'):
        manager._check_judge_fingerprint(review('old'), 'reviews.jsonl')


def test_judge_cache_is_not_reused_by_a_rule_run(tmp_path):
    manager = make_manager(tmp_path, None)

    with pytest.raises(ValueError, match='different judge configuration'):
        manager._check_judge_fingerprint(review('judge'), 'reviews.jsonl')


def test_rerun_is_atomic(tmp_path):
    manager = make_manager(tmp_path, 'new')
    target = manager.get_review_cache_path('default')
    with open(target, 'w', encoding='utf-8') as file:
        file.write('old\n')

    manager.delete_review_cache('default')

    with open(target, encoding='utf-8') as file:
        assert file.read() == 'old\n'
    manager.commit_review_reruns()
    with open(target, encoding='utf-8') as file:
        assert file.read() == 'old\n'
