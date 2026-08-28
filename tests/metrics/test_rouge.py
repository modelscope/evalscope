"""One-sample rouge helpers score every (prediction, reference) pair, not only the last one.

ToolBench passes a whole multi-turn conversation as the pair list, so an overwrite-per-pair
loop made Rouge-L reflect just the final turn.  The mean over pairs restores the official
ToolBench convention (`rouge.get_scores(..., avg=True)` in the vendored evaluator that these
helpers replaced).
"""
from evalscope.metrics.utils.rouge import compute_rouge_score_one_sample, compute_rouge_score_one_sample_zh

ENGLISH_KEYS = (
    'rouge-1-r',
    'rouge-1-p',
    'rouge-1-f',
    'rouge-2-r',
    'rouge-2-p',
    'rouge-2-f',
    'rouge-l-r',
    'rouge-l-p',
    'rouge-l-f',
)

CHINESE_KEYS = (
    'Rouge-1-R',
    'Rouge-1-P',
    'Rouge-1-F',
    'Rouge-2-R',
    'Rouge-2-P',
    'Rouge-2-F',
    'Rouge-L-R',
    'Rouge-L-P',
    'Rouge-L-F',
)


def test_english_averages_over_all_pairs() -> None:
    score = compute_rouge_score_one_sample(['aaa bbb', 'ccc'], ['aaa bbb', 'xxx'])

    # First pair is identical (1.0), second shares no token (0.0): mean 0.5 per metric.
    for key in ENGLISH_KEYS:
        assert score[key] == 0.5, key


def test_english_single_pair_is_its_own_mean() -> None:
    score = compute_rouge_score_one_sample(['aaa bbb'], ['aaa bbb'])

    assert score['rouge-l-f'] == 1.0


def test_english_reports_all_keys_when_no_pair_scored() -> None:
    score = compute_rouge_score_one_sample([], [])

    assert score == {key: 0.0 for key in ENGLISH_KEYS}


def test_english_unscorable_pair_neither_clobbers_nor_counts() -> None:
    """A pair the scorer cannot handle is skipped; earlier and later valid pairs remain.

    `None` makes the tokenizer raise, which the helper catches and logs per pair.
    """
    score = compute_rouge_score_one_sample([None, 'aaa bbb'], ['xxx yyy', 'aaa bbb'])

    assert score['rouge-1-f'] == 1.0


def test_chinese_averages_over_all_pairs() -> None:
    score = compute_rouge_score_one_sample_zh(['你好 世界', '完全不同'], ['你好 世界', '另外一句话'])

    # Identical first pair (1.0) averaged with a disjoint second pair (~0).
    assert abs(score['Rouge-1-F'] - 0.5) < 1e-3
    assert abs(score['Rouge-L-F'] - 0.5) < 1e-3


def test_chinese_reports_all_keys_when_no_pair_scored() -> None:
    score = compute_rouge_score_one_sample_zh([], [])

    assert score == {key: 0.0 for key in CHINESE_KEYS}
