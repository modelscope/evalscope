# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright (c) lmsys.org.

from collections import defaultdict


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

    for rd, model_a, model_b, win in battles[[
            col_model_a, col_model_b, col_win
    ]].itertuples():
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
