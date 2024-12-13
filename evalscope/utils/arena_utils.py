# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright (c) lmsys.org.

import numpy as np
import pandas as pd
import pyarrow as pa
import random
from collections import OrderedDict, defaultdict
from typing import List, Sequence, Union

from evalscope.utils.logger import get_logger

logger = get_logger()


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


def merge_ques_ans(answer_list_all, merge_key: str = 'question_id', merge_mode: str = 'inner') -> pd.DataFrame:
    """
    Merge question and answer list to unifiled data.

    Args:
        answer_list_all: list of answer list,
            e.g. [ans1_list, ans2_list, ...], an ans_list is predicted answers
            of a specific model, must contain following columns: 'question_id',
            'text', 'category', 'model_id', 'answer'
        merge_key: key for dataframe merging
        merge_mode: mode for dataframe merging,
            e.g. 'inner', 'left', 'right', 'outer'

    Returns:
        pandas DataFrame: merged dataframe, e.g. columns are
            ['question_id', 'gpt-3.5-turbo', 'llama2-7b']
    """
    ans_df = pd.DataFrame()
    for ans_list in answer_list_all:
        ans_list = [{'question_id': item['question_id'], item['model_id']: item} for item in ans_list]
        if ans_df.empty:
            ans_df = pa.Table.from_pylist(ans_list).to_pandas()
        else:
            ans_df = pd.merge(ans_df, pa.Table.from_pylist(ans_list).to_pandas(), on=merge_key, how=merge_mode)

    return ans_df


def get_battle_pairs(columns: List[str], baseline_idx: int = -1) -> List[tuple]:
    """
    Get battle pair names from columns.

    Args:
        columns: list of column names.

    Returns:
        list of battle pairs.

    Example:
        >>> columns = ['A', 'B', 'C']
        >>> res = get_battle_pairs(columns)
        >>> print(res)
        >>> [('B', 'A'), ('C', 'A'), ('C', 'B')]

        >>> columns = ['A', 'B', 'C']
        >>> res = get_battle_pairs(columns, 2)
        >>> print(res)
        >>> [('A', 'C'), ('B', 'C')]
    """
    res_list = []

    cols_num = len(columns)
    if cols_num <= 0:
        return res_list

    if baseline_idx != -1:
        n_column = columns[baseline_idx]
        res_list = [(column, n_column) for column in columns if column != n_column]
    else:
        mat = np.ones((cols_num, cols_num))
        mat_lower_tril = np.tril(mat, k=-1)
        x_ids, y_ids = np.where(mat_lower_tril == 1)
        res_list = [(columns[x_id], columns[y_id]) for x_id, y_id in zip(x_ids, y_ids)]

    return res_list


def get_battle_pairs_origin(columns: List[str], compare_base: bool = False, swap: bool = False):  # TODO: to refactor
    """
    Get battle pair names from columns.

    Args:
        columns: list of column names.

    Returns:
        list of battle pairs.

    Example:
        >>> columns = ['A', 'B', 'C']
        >>> res = get_battle_pairs(columns)
        >>> print(res)
        >>> [('B', 'A'), ('C', 'A'), ('C', 'B')]
    """
    res_list = []

    cols_num = len(columns)
    if cols_num <= 0:
        return res_list

    if not compare_base:
        mat = np.ones((cols_num, cols_num))
        mat_lower_tril = np.tril(mat, k=-1)
        x_ids, y_ids = np.where(mat_lower_tril == 1)
        res_list = [(columns[x_id], columns[y_id]) for x_id, y_id in zip(x_ids, y_ids)]
    else:
        for column in columns[1:]:
            res_list.append((columns[0], column))

    if swap:
        res_list.extend([(j, i) for i, j in res_list])
    return res_list


def shuffle_pairwise_preferences(df: pd.DataFrame, arr_is_shuffle: Sequence[int]) -> pd.DataFrame:
    """Shuffle the outputs of a pairwise preference dataframe.

    Examples
    --------
    >>> df = pd.DataFrame([dict(instruction='2+2', output_1='3', output_2='4', preference=2),
                           dict(instruction='2+3', output_1='5', output_2='4', preference=1)])
    >>> print(shuffle_pairwise_preferences(df, [True, False]))
        instruction output_1 output_2  preference
    0         2+2        4        3           1
    1         2+3        5        4           1
    """
    col_1 = df['output_1'].copy()
    col_2 = df['output_2'].copy()
    df['output_1'] = np.where(arr_is_shuffle, col_2, col_1)
    df['output_2'] = np.where(arr_is_shuffle, col_1, col_2)

    if 'preference' in df.columns:
        df['preference'] = np.where(arr_is_shuffle, 3 - df['preference'], df['preference'])

    return df


class BattlePairSelection:
    """
    Select battle pairs by specific strategy.

    Attributes:
        model_elo_map(dict): map of model_id--base_elo_score
    """

    DEFAULT_K = 5

    def __init__(self, model_elo_map: Union[dict, OrderedDict]):
        # Make sure model_elo_map to be ordered when compare_base is true.
        self.model_elo_map = model_elo_map

    def top_k(self, k: int = DEFAULT_K, compare_base: bool = False, swap: bool = False) -> list:
        if k <= 0:
            k = self.DEFAULT_K
        sorted_res = sorted(self.model_elo_map.items(), key=lambda x: x[1])[:k]
        sorted_res = list(dict(sorted_res).keys())
        return get_battle_pairs_origin(sorted_res, compare_base, swap)

    def random_k(self, k: int = DEFAULT_K, compare_base: bool = False, swap: bool = False) -> list:
        if k <= 0:
            k = self.DEFAULT_K
        if k > len(self.model_elo_map):
            k = len(self.model_elo_map)
        candidate_list = list(self.model_elo_map.items())
        k = len(candidate_list) if k > len(candidate_list) else k
        res = dict(random.sample(candidate_list, k=k))
        res = list(res.keys())
        return get_battle_pairs_origin(res, compare_base, swap)

    def volatility_index(self, frac: float = 0.2, compare_base: bool = False, swap: bool = False) -> list:
        res_list = []
        candidate_list = get_battle_pairs_origin(list(self.model_elo_map.keys()), compare_base, swap)
        for t in candidate_list:
            model_a = t[0]
            model_b = t[1]
            base_elo_a = self.model_elo_map.get(model_a)
            base_elo_b = self.model_elo_map.get(model_b)

            vol_frac = abs(base_elo_b - base_elo_a) / max(base_elo_a, base_elo_b)
            if vol_frac <= frac:
                res_list.append(t)

        return res_list
