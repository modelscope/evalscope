# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright (c) EleutherAI. and its affiliates.
# Copyright (c) OpenAI. and its affiliates.

import itertools
import math
import numpy as np
import random
import sacrebleu
from collections import defaultdict
from collections.abc import Iterable
from typing import TYPE_CHECKING, Dict, List, Union


def mean(arr: list):
    if not arr:
        return 0.0

    if isinstance(arr[0], list):
        arr = [item for sublist in arr for item in sublist]
    return sum(arr) / len(arr)


def pass_at_k(arr: Union[List[int], List[List[int]]], k: int = 1) -> float:
    if not arr:
        return 0.0

    def sub_pass_at_k(sub_arr: List[int]) -> float:
        return 1.0 if any(sub_arr[:k]) else 0.0

    if isinstance(arr[0], list):
        return sum(sub_pass_at_k(sub_arr) for sub_arr in arr) / len(arr)
    else:
        return sum(arr) / len(arr)


def pop_stddev(arr):
    mu = mean(arr)
    return math.sqrt(sum([(x - mu)**2 for x in arr]) / len(arr))


def sample_stddev(arr):
    mu = mean(arr)
    return math.sqrt(sum([(x - mu)**2 for x in arr]) / (len(arr) - 1))


def mean_stderr(arr):
    return sample_stddev(arr) / math.sqrt(len(arr))


def median(arr):
    return arr[len(arr) // 2]


def matthews_corrcoef(items):
    import sklearn.metrics

    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    return sklearn.metrics.matthews_corrcoef(golds, preds)


def simple_f1_score(scores: tuple) -> float:
    score1 = scores[0]
    score2 = scores[1]
    score1 = np.mean(score1) if len(score1) > 0 else 0.0
    score2 = np.mean(score2) if len(score2) > 0 else 0.0

    if score1 == 0 and score2 == 0:
        return 0.0
    else:
        return 2 * score1 * score2 / (score1 + score2)


def f1_score(items):
    import sklearn.metrics

    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    fscore = sklearn.metrics.f1_score(golds, preds)

    return np.max(fscore)


def acc_all(items):
    # Only count as correct if all answers are labeled correctly for each question
    question_scoring_dict = {}
    preds = list(zip(*items))[0]
    docs = list(zip(*items))[1]

    for doc, pred in zip(docs, preds):
        paragraph_id = doc['idx']['paragraph']
        question_id = doc['idx']['question']
        if (paragraph_id, question_id) not in question_scoring_dict:
            question_scoring_dict[(paragraph_id, question_id)] = []

        gold_label = doc['label'] == 1

        question_scoring_dict[(paragraph_id, question_id)].append(gold_label == pred)
    acc = np.mean([int(all(x)) for x in question_scoring_dict.values()])
    return acc


def acc_all_stderr(items):
    # Only count as correct if all answers are labeled correctly for each question
    question_scoring_dict = {}
    preds = list(zip(*items))[0]
    docs = list(zip(*items))[1]

    for doc, pred in zip(docs, preds):
        question_id = doc['idx']['question']
        if question_id not in question_scoring_dict:
            question_scoring_dict[question_id] = []

        gold_label = doc['label'] == 1
        question_scoring_dict[question_id].append(gold_label == pred)

    acc = mean_stderr([int(all(x)) for x in question_scoring_dict.values()])
    return acc


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """Compute max metric between prediction and each ground truth."""
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def perplexity(items):
    return math.exp(-mean(items))


def weighted_mean(items: List) -> float:
    # e.g. [(0,1), (0.5,1), (1,1)]
    a, b = zip(*items)
    return sum(a) / sum(b)


def micro_mean(items):
    try:
        return sum([item.score * item.num for item in items]) / sum([item.num for item in items])
    except ZeroDivisionError:
        return 0.0


def macro_mean(items):
    try:
        return sum([item.score for item in items]) / len(items)
    except ZeroDivisionError:
        return 0.0


def weighted_perplexity(items):
    return math.exp(-weighted_mean(items))


def bits_per_byte(items):
    return -weighted_mean(items) / math.log(2)


def bleu(items):
    """The Bilingual Evaluation Understudy Score, or BLEU for short, is a metric
    for evaluating a generated sentence to a reference sentence. It counts matching
    n-grams in the candidate translation to n-grams in the reference text, where
    1-gram or unigram would be each token and a bigram comparison would be each
    word pair. The comparison is made regardless of word order
    Source: https://machinelearningmastery.com/calculate-bleu-score-for-text-python/
    Paper: https://www.aclweb.org/anthology/P02-1040/

    Higher is better
    """
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    refs, preds = _sacreformat(refs, preds)
    return sacrebleu.corpus_bleu(preds, refs).score


def bleu_ngram_one_sample(predict, reference):
    """
    Calculate BLEU-1, BLEU-2, BLEU-3, and BLEU-4 scores

    Args:
        items: [(ref, pred)]

    Returns:
        {
            'bleu-1': 0.8,
            'bleu-2': 0.45,
            'bleu-3': 0.0,
            'bleu-4': 0.0
        }

    """
    import jieba
    from nltk import word_tokenize
    from nltk.translate.bleu_score import sentence_bleu

    def is_contains_chinese(strs):
        for _char in strs:
            if '\u4e00' <= _char <= '\u9fa5':
                return True
        return False

    predict = list(jieba.cut(predict)) if is_contains_chinese(predict) else word_tokenize(predict)
    reference = [list(jieba.cut(reference))] if is_contains_chinese(reference) else [word_tokenize(reference)]

    result = dict()
    result['bleu-1'] = sentence_bleu(reference, predict, weights=(1, 0, 0, 0))
    result['bleu-2'] = sentence_bleu(reference, predict, weights=(0, 1, 0, 0))
    result['bleu-3'] = sentence_bleu(reference, predict, weights=(0, 0, 1, 0))
    result['bleu-4'] = sentence_bleu(reference, predict, weights=(0, 0, 0, 1))

    return result


def chrf(items):
    """chrF++ is a tool for automatic evaluation of machine translation output
    based on character n-gram precision and recall enhanced with word n-grams.
    Source: https://github.com/m-popovic/chrF
    Paper: https://www.aclweb.org/anthology/W15-3049.pdf

    Higher is better  # TODO I think
    """
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    refs, preds = _sacreformat(refs, preds)
    return sacrebleu.corpus_chrf(preds, refs).score


def ter(items):
    """Translation Error Rate is an error metric for machine translation that
    measures the number of edits required to change a system output into one
    of the references
    Source: http://www.cs.umd.edu/~snover/tercom/
    Paper: http://mt-archive.info/AMTA-2006-Snover.pdf

    Lower is better
    """
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    refs, preds = _sacreformat(refs, preds)
    return sacrebleu.corpus_ter(preds, refs).score


def is_non_str_iterable(obj):
    return isinstance(obj, Iterable) and not isinstance(obj, str)


def _sacreformat(refs, preds):
    """Format refs and preds for sacrebleu corpus calculation. It is very particular"""
    # Sacrebleu expects (List[str], List[List[str])
    #   e.g. sacrebleu.corpus_bleu([pred_t], [[ref1_stream], [ref2_stream], ...])

    # Note [ref1_stream] is the first reference for each pred.
    # So lists are size N and (M, N) for N preds and M possible refs for each pred
    # This is a different order of dimensions that I would expect

    # We expect refs to be List[str] or List[List[str]], the outer list corresponding to preds
    # Must become List[List[str]] with the inner list corresponding to preds
    if not is_non_str_iterable(refs):
        refs = list(refs)
    if not is_non_str_iterable(refs[0]):
        refs = [[ref] for ref in refs]
    refs = list(zip(*refs))
    # Note the number of refs in each ref list much match the number of preds

    # We expect preds to be List[str] or List[List[str]]. Must become List[str]
    if not is_non_str_iterable(preds):
        preds = list(preds)
    if is_non_str_iterable(preds[0]):
        assert len(preds[0]) == 1, f'Pred must be a str, was {preds[0]}'
        preds = [pred[0] for pred in preds]

    return refs, preds


class _bootstrap_internal:

    def __init__(self, f, n):
        self.f = f
        self.n = n

    def __call__(self, v):
        i, xs = v
        rnd = random.Random()
        rnd.seed(i)
        res = []
        for _ in range(self.n):
            res.append(self.f(rnd.choices(xs, k=len(xs))))
        return res


def bootstrap_stderr(f, xs, iters):
    import multiprocessing as mp

    pool = mp.Pool(mp.cpu_count())
    # this gives a biased estimate of the stderr (i.e w/ the mean, it gives something
    # equivalent to stderr calculated without Bessel's correction in the stddev.
    # Unfortunately, I haven't been able to figure out what the right correction is
    # to make the bootstrap unbiased - i considered multiplying by sqrt(n/(n-1)) but
    # that would be ad-hoc and I can't prove that that would actually be an unbiased estimator)
    # Thankfully, shouldn't matter because our samples are pretty big usually anyways
    res = []
    chunk_size = min(1000, iters)
    from tqdm import tqdm

    print('bootstrapping for stddev:', f.__name__)
    for bootstrap in tqdm(
            pool.imap(
                _bootstrap_internal(f, chunk_size),
                [(i, xs) for i in range(iters // chunk_size)],
            ),
            total=iters // chunk_size,
    ):
        # sample w replacement
        res.extend(bootstrap)

    pool.close()
    return sample_stddev(res)


def stderr_for_metric(metric, bootstrap_iters):
    bootstrappable = [
        median,
        matthews_corrcoef,
        f1_score,
        perplexity,
        bleu,
        chrf,
        ter,
    ]

    if metric in bootstrappable:
        return lambda x: bootstrap_stderr(metric, x, iters=bootstrap_iters)

    stderr = {mean: mean_stderr, acc_all: acc_all_stderr}

    return stderr.get(metric, None)


def yesno(x):
    if x:
        return 'yes'
    else:
        return 'no'


def compute_elo(battles,
                col_model_a='model_a',
                col_model_b='model_b',
                col_win='win',
                tie_values=['tie', 'tie (bothbad)'],
                k=32,
                scale=400,
                base=10,
                init_rating=1000):
    rating = defaultdict(lambda: init_rating)

    for rd, model_a, model_b, win in battles[[col_model_a, col_model_b, col_win]].itertuples():
        ra = rating[model_a]
        rb = rating[model_b]
        ea = 1 / (1 + base**((rb - ra) / scale))
        eb = 1 / (1 + base**((ra - rb) / scale))
        if win == col_model_a:
            sa = 1
        elif win == col_model_b:
            sa = 0
        elif win in tie_values:
            sa = 0.5
        else:
            raise Exception(f'unexpected vote {win}')
        rating[model_a] += k * (sa - ea)
        rating[model_b] += k * (1 - sa - eb)

    return rating


def exact_match(gold: str, pred: str) -> float:
    if not pred:
        return 0

    return 1 if gold.strip() == pred.strip() else 0


def calculate_arc_accuracy(question_answers: Dict[str, str], predictions: Dict[str, List[str]]) -> float:
    """
    Calculate accuracy for ARC benchmark.

    Args:
        question_answers: question_id -> answer mapping, e.g. {'abc_123': 'A'}
        predictions: question_id -> prediction mapping, e.g. {'abc_123': ['D'], 'xyz_456': ['A', 'C']}

    Returns:
        accuracy score (float)

    Notes:
        Each question is worth one point. Models are allowed to give multiple answers (e.g., "A;C"),
        in which case the model receives 1/N points credit if one of its N answers is correct.
        Refer to: https://leaderboard.allenai.org/arc/submissions/get-started
    """
    score = 0.0

    for question_id, answer in question_answers.items():
        try:
            predictions_for_q = predictions[question_id]
        except Exception as e:
            raise KeyError(f'Missing arc prediction: {e}')

        if answer in predictions_for_q:
            score += 1.0 / len(predictions_for_q)

        del predictions[question_id]

    if len(predictions) > 0:
        log_ex: str = ', '.join(list(predictions.keys())[:3])
        raise ValueError(f'Found {len(predictions)} extra predictions, for example: {log_ex}')

    return score / len(question_answers)


def calculate_pass_at_k(num_samples: Union[int, List[int], np.ndarray],
                        num_correct: Union[List[int], np.ndarray],
                        k: int = 1) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    Examples:
        >>> import numpy as np
        >>> from typing import Union
        >>> total = np.array([5, 5, 5])
        >>> correct = np.array([2, 4, 2])
        >>> calculate_pass_at_k(total, correct, 1)
        result: "array([0.4, 0.8, 0.4])"
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])
