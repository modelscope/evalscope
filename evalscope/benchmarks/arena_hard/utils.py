import math
import numpy as np
import pandas as pd
import re
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from evalscope.utils.logger import get_logger

logger = get_logger()


def post_process_arenahard(completion):
    result = re.findall(r'\[\[([AB<>=]+)\]\]', completion)
    if result:
        return result[0]
    else:
        return None


def get_battles_from_row(row, first_game_only=False, multiplier=3):
    results = []
    output = {'model_a': row['model_a'], 'model_b': row['model_b']}

    game = row['games'][0]
    weight = 1
    if game['score'] == 'A=B':
        output['winner'] = 'tie'
    elif game['score'] == 'A>B':
        output['winner'] = 'model_a'
    elif game['score'] == 'A>>B':
        output['winner'] = 'model_a'
        weight = multiplier
    elif game['score'] == 'B>A':
        output['winner'] = 'model_b'
    elif game['score'] == 'B>>A':
        output['winner'] = 'model_b'
        weight = multiplier
    else:
        weight = 0

    if weight:
        results += [output] * weight

    if first_game_only:
        return pd.DataFrame(results)

    # game 2
    output = {'model_a': row['model_a'], 'model_b': row['model_b']}

    game = row['games'][1]

    weight = 1
    if game['score'] == 'A=B':
        output['winner'] = 'tie'
    elif game['score'] == 'A>B':
        output['winner'] = 'model_b'
    elif game['score'] == 'A>>B':
        output['winner'] = 'model_b'
        weight = multiplier
    elif game['score'] == 'B>A':
        output['winner'] = 'model_a'
    elif game['score'] == 'B>>A':
        output['winner'] = 'model_a'
        weight = multiplier
    else:
        weight = 0

    if weight:
        results += [output] * weight

    return pd.DataFrame(results)


def compute_mle_elo(df, SCALE=400, BASE=10, INIT_RATING=1000):
    models = pd.concat([df['model_a'], df['model_b']]).unique()
    models = pd.Series(np.arange(len(models)), index=models)

    # duplicate battles
    df = pd.concat([df, df], ignore_index=True)
    p = len(models.index)
    n = df.shape[0]

    X = np.zeros([n, p])
    X[np.arange(n), models[df['model_a']]] = +math.log(BASE)
    X[np.arange(n), models[df['model_b']]] = -math.log(BASE)

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
        elo_scores = pd.Series(INIT_RATING, index=models.index)
        if np.all(Y == 1.0):
            elo_scores[df['model_a'].iloc[0]] += SCALE  # Boost the winning model
        elif np.all(Y == 0.0):
            elo_scores[df['model_b'].iloc[0]] += SCALE  # Boost the winning model
        return elo_scores.sort_values(ascending=False)

    lr = LogisticRegression(
        fit_intercept=False, penalty=None, tol=1e-8)  # May need to set a small value when not use GPT4 as judge model
    lr.fit(X, Y)

    elo_scores = SCALE * lr.coef_[0] + INIT_RATING

    # set anchor as gpt4-0314 = 1000
    if 'gpt4-0314' in models.index:
        elo_scores += 1000 - elo_scores[models['gpt4-0314']]
    return pd.Series(elo_scores, index=models.index).sort_values(ascending=False)


def get_bootstrap_result(battles, func_compute_elo, num_round):
    rows = []
    for _ in tqdm(range(num_round), desc='bootstrap'):
        res = func_compute_elo(battles.sample(frac=1.0, replace=True))
        if res is not None:
            rows.append(res)
    df = pd.DataFrame(rows)
    return df[df.median().sort_values(ascending=False).index]


def preety_print_two_ratings(ratings_1, ratings_2, column_names):
    df = (
        pd.DataFrame(
            [[n, ratings_1[n], ratings_2[n]] for n in ratings_1.keys()],
            columns=['Model', column_names[0], column_names[1]],
        ).sort_values(column_names[0], ascending=False).reset_index(drop=True))
    df[column_names[0]] = (df[column_names[0]] + 0.5).astype(int)
    df[column_names[1]] = (df[column_names[1]] + 0.5).astype(int)
    df.index = df.index + 1
    return df


def predict_win_rate(elo_ratings, SCALE=400, BASE=10, INIT_RATING=1000):
    names = sorted(list(elo_ratings.keys()))
    wins = defaultdict(lambda: defaultdict(lambda: 0))
    for a in names:
        for b in names:
            ea = 1 / (1 + BASE**((elo_ratings[b] - elo_ratings[a]) / SCALE))
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
