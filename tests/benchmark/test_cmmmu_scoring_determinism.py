"""Regression tests for CMMMU's scoring fallbacks.

Both fallbacks guess when the answer cannot be extracted. Guessing is the benchmark's own
behaviour, but an unseeded guess made the same prediction score differently on every `rerun_review`.
"""

import random

from evalscope.benchmarks.cmmmu.utils import eval_cmmmu, get_multi_choice_prediction

CHOICES = ['A', 'B', 'C', 'D']
INDEX2ANS = {'A': 'alpha', 'B': 'beta', 'C': 'gamma', 'D': 'delta'}
UNPARSEABLE = 'the model rambled without naming an option'


def test_multi_choice_guess_is_stable_for_the_same_response() -> None:
    guesses = set()
    for seed in range(8):
        random.seed(seed)
        guesses.add(get_multi_choice_prediction(UNPARSEABLE, CHOICES, INDEX2ANS))

    assert len(guesses) == 1


def test_multi_choice_guess_still_varies_between_responses() -> None:
    responses = [f'{UNPARSEABLE} {i}' for i in range(40)]

    guesses = {get_multi_choice_prediction(r, CHOICES, INDEX2ANS) for r in responses}
    assert len(guesses) > 1


def test_true_false_tie_break_is_stable_for_the_same_prediction() -> None:
    """A tie between positive and negative keywords is decided by a guess."""
    entry = {'type': '判断', 'answer': '对', 'parsed_pred': ['对的', '错的']}

    results = set()
    for seed in range(8):
        random.seed(seed)
        results.add(eval_cmmmu(dict(entry)))

    assert len(results) == 1
