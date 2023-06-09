# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright (c) lmsys.org.

from typing import List

import numpy as np
import pandas as pd
import pyarrow as pa
from evals.utils.logger import get_logger

logger = get_logger()


def merge_ques_ans(answer_list_all,
                   merge_key: str = 'question_id',
                   merge_mode: str = 'inner') -> pd.DataFrame:
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
            ['question_id', 'text', 'category', 'gpt-3.5-turbo', ...]
    """
    ans_df = pd.DataFrame()
    for ans_list in answer_list_all:
        ans_list = [{
            'question_id': item['question_id'],
            item['model_id']: item
        } for item in ans_list]
        if ans_df.empty:
            ans_df = pa.Table.from_pylist(ans_list).to_pandas()
        else:
            ans_df = pd.merge(
                ans_df,
                pa.Table.from_pylist(ans_list).to_pandas(),
                on=merge_key,
                how=merge_mode)

    return ans_df


def get_battle_pairs(columns: List[str]):
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
    mat = np.ones((cols_num, cols_num))
    mat_lower_tril = np.tril(mat, k=-1)
    x_ids, y_ids = np.where(mat_lower_tril == 1)
    res_list = [(columns[x_id], columns[y_id])
                for x_id, y_id in zip(x_ids, y_ids)]

    return res_list
