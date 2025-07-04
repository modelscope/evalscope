import inspect
import math
import numpy as np
import pandas as pd
import re
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from evalscope.utils.logger import get_logger

logger = get_logger()


def process_review_item(review_item: dict) -> dict:
    """
    Process a single review item to extract relevant information.

    Args:
        review_item (dict): The review item to process.

    Returns:
        dict: Processed review item with necessary information.
    """
    res = []
    raw_input = review_item['raw_input']
    sample_index = review_item['index']
    question_keys = ['question', 'Question', 'prompt', 'Prompt', 'query', 'Query', 'problem', 'Problem']
    # Find the first non-empty question key in raw_input
    question = next((raw_input.get(key) for key in question_keys if raw_input.get(key)), None)
    for choice_index, choice in enumerate(review_item['choices']):
        raw_pred_answer = choice['message']['content']
        parsed_gold_answer = choice['review']['gold']
        parsed_pred_answer = choice['review']['pred']
        score = choice['review']['result']
        raw_d = {
            'Index': f'{sample_index}_{choice_index}',
            'Input': raw_input,
            'Question': question if question else '*No Question*',
            'Generated': raw_pred_answer,
            'Gold': parsed_gold_answer if parsed_gold_answer != raw_input else '*Same as Input*',
            'Pred': parsed_pred_answer,
            'Score': score,
        }
        res.append(raw_d)

    return res


def post_process_result(completion):
    result = re.findall(r'\[\[([AB<>=]+)\]\]', completion)
    if result:
        return result[0]
    else:
        return None


def get_judge_score(result, reverse=False):
    """
    Calculate the judge score, considering confidence weight.

    Args:
        result: Judgment result ('A=B', 'A>B', 'A>>B', 'B>A', 'B>>A')
        reverse: Whether to reverse the score

    Returns:
        float: Weighted score
    """

    # Base score mapping - using finer-grained scores
    if not reverse:
        score_mapping = {
            'A=B': 0.5,  # Tie
            'A>B': 0.75,  # A slightly wins
            'A>>B': 1.0,  # A significantly wins
            'B>A': 0.25,  # B slightly wins
            'B>>A': 0.0,  # B significantly wins
        }
    else:
        score_mapping = {
            'A=B': 0.5,  # Tie
            'A>B': 0.25,  # A slightly wins
            'A>>B': 0.0,  # A significantly wins
            'B>A': 0.75,  # B slightly wins
            'B>>A': 1.0,  # B significantly wins
        }

    base_score = score_mapping.get(result, 0.5)

    return base_score


def get_battles_from_row(row, first_game_only=False, multiplier=3):
    results = []

    game = row['games'][0]
    output = {'model_a': game['model_a'], 'model_b': game['model_b']}

    weight = 1
    if game['judgment'] == 'A=B':
        output['winner'] = 'tie'
    elif game['judgment'] == 'A>B':
        output['winner'] = 'model_a'
    elif game['judgment'] == 'A>>B':
        output['winner'] = 'model_a'
        weight = multiplier
    elif game['judgment'] == 'B>A':
        output['winner'] = 'model_b'
    elif game['judgment'] == 'B>>A':
        output['winner'] = 'model_b'
        weight = multiplier
    else:
        weight = 0

    if weight:
        results += [output] * weight

    if first_game_only:
        return pd.DataFrame(results)

    # Dont change the order of model_a and model_b
    output = {'model_a': game['model_a'], 'model_b': game['model_b']}

    # game 2
    game = row['games'][1]

    weight = 1
    if game['judgment'] == 'A=B':
        output['winner'] = 'tie'
    elif game['judgment'] == 'A>B':
        output['winner'] = 'model_b'
    elif game['judgment'] == 'A>>B':
        output['winner'] = 'model_b'
        weight = multiplier
    elif game['judgment'] == 'B>A':
        output['winner'] = 'model_a'
    elif game['judgment'] == 'B>>A':
        output['winner'] = 'model_a'
        weight = multiplier
    else:
        weight = 0

    if weight:
        results += [output] * weight

    return pd.DataFrame(results)


def compute_mle_elo(df, scale=400, base=10, init_rating=1000, baseline_model='gpt4-0314'):
    models = pd.concat([df['model_a'], df['model_b']]).unique()
    models = pd.Series(np.arange(len(models)), index=models)

    # duplicate battles
    df = pd.concat([df, df], ignore_index=True)
    p = len(models.index)
    n = df.shape[0]

    X = np.zeros([n, p])
    X[np.arange(n), models[df['model_a']]] = +math.log(base)
    X[np.arange(n), models[df['model_b']]] = -math.log(base)

    # one A win => two A win
    Y = np.zeros(n)
    Y[df['winner'] == 'model_a'] = 1.0

    # one tie => one A win + one B win
    # find tie + tie (both bad) index
    tie_idx = (df['winner'] == 'tie') | (df['winner'] == 'tie (bothbad)')
    tie_idx[len(tie_idx) // 2:] = False
    Y[tie_idx] = 1.0

    if len(np.unique(Y)) < 2:
        logger.info('Warning: Only one class in the data')
        elo_scores = pd.Series(init_rating, index=models.index)
        if np.all(Y == 1.0):
            elo_scores[df['model_a'].iloc[0]] += scale  # Boost the winning model
        elif np.all(Y == 0.0):
            elo_scores[df['model_b'].iloc[0]] += scale  # Boost the winning model
        return elo_scores.sort_values(ascending=False)

    lr = LogisticRegression(
        fit_intercept=False, penalty=None, tol=1e-8)  # May need to set a small value when not use GPT4 as judge model
    lr.fit(X, Y)

    elo_scores = scale * lr.coef_[0] + init_rating

    # set anchor 1000
    if baseline_model in models.index:
        elo_scores += 1000 - elo_scores[models[baseline_model]]
    return pd.Series(elo_scores, index=models.index).sort_values(ascending=False)


def get_bootstrap_result(battles, func_compute_elo, num_round, baseline_model='gpt-4-0314'):
    rows = []
    kwargs = {}
    if 'baseline_model' in inspect.signature(func_compute_elo).parameters:
        kwargs['baseline_model'] = baseline_model
    for _ in tqdm(range(num_round), desc='bootstrap'):
        res = func_compute_elo(battles.sample(frac=1.0, replace=True), **kwargs)
        if res is not None:
            rows.append(res)
    df = pd.DataFrame(rows)
    return df[df.median().sort_values(ascending=False).index]


def predict_win_rate(elo_ratings, scale=400, base=10, init_rating=1000):
    names = sorted(list(elo_ratings.keys()))
    wins = defaultdict(lambda: defaultdict(lambda: 0))
    for a in names:
        for b in names:
            ea = 1 / (1 + base**((elo_ratings[b] - elo_ratings[a]) / scale))
            wins[a][b] = ea
            wins[b][a] = 1 - ea

    data = {a: [wins[a][b] if a != b else np.NAN for b in names] for a in names}

    df = pd.DataFrame(data, index=names)
    df.index.name = 'model_a'
    df.columns.name = 'model_b'
    return df.T


def get_win_rate_column(df, column, baseline='gpt4-0314'):
    to_dict = df[['model', column]].set_index('model').to_dict()[column]
    win_rate_table = predict_win_rate(to_dict)
    return win_rate_table[baseline].fillna(0.5).apply(lambda x: round(x, 4))
