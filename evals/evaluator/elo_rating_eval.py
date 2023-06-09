# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import List, Union

import pandas as pd
import pyarrow as pa
from evals.constants import MetricMembers
from evals.evaluate import Evaluate
from evals.metrics.arena_metrics import compute_elo
from evals.utils.logger import get_logger
from evals.utils.utils import jsonl_to_list

logger = get_logger()

DEFAULT_COLUMNS_MAPPING = {
    'model_a': 'model_a',
    'model_b': 'model_b',
    'win': 'win',
    'tstamp': 'ts',
    'language': 'lang'
}


class EloRatingEvaluate(Evaluate):

    def __init__(self, metrics: list, **kwargs):
        super().__init__(metrics=metrics, **kwargs)

    def preprocess(self, raw_data_df: pd.DataFrame, **kwargs):

        # Get battles data
        raw_data_df = raw_data_df.sort_values(ascending=True, by=['tstamp'])
        battles = raw_data_df[raw_data_df['anony']].reset_index(drop=True)

        return battles

    def eval_samples(self, data_list: list):
        res_all = []

        raw_data: pd.DataFrame = None

        if len(data_list) > 0:
            raw_data = data_list[0]

        for metric in self.metrics:

            if metric == MetricMembers.ELO.value:
                # TODO: registry to be added
                battles = self.preprocess(raw_data_df=raw_data)
                elo_ratings = compute_elo(battles)
                col_model = 'Model'
                col_elo_rating = 'Elo_Rating'
                elo_ratings_res = pd.DataFrame(
                    [[n, elo_ratings[n]] for n in elo_ratings.keys()],
                    columns=[col_model, col_elo_rating]).sort_values(
                        col_elo_rating, ascending=False).reset_index(drop=True)
                elo_ratings_res = elo_ratings_res.round({col_elo_rating: 1})
                res_all.append(elo_ratings_res)

            else:
                raise ValueError(f'Unsupported metric: {metric}')

        return res_all

    def run(self, prompts: Union[str, list], **kwargs) -> List[pd.DataFrame]:
        """
        Load the predicted samples and evaluate them in arena mode.
        """
        # raw_data = pd.read_json(prompts)
        data_list = jsonl_to_list(prompts)
        data_df = pa.Table.from_pylist(data_list).to_pandas()
        res_list = self.eval_samples([data_df])

        return res_list
