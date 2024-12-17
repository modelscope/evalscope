# Copyright (c) Alibaba, Inc. and its affiliates.

import pandas as pd
import pyarrow as pa
from typing import List, Union

from evalscope.constants import MetricMembers
from evalscope.utils.arena_utils import compute_elo
from evalscope.utils.io_utils import jsonl_to_list
from evalscope.utils.logger import get_logger

logger = get_logger()

DEFAULT_COLUMNS_MAPPING = {'model_a': 'model_a', 'model_b': 'model_b', 'win': 'win', 'tstamp': 'ts', 'language': 'lang'}


class RatingEvaluate(object):

    def __init__(self, metrics: list, baseline_model: str = None, **kwargs):
        self.metrics = metrics
        self.baseline_model = baseline_model
        self.kwargs = kwargs

    def preprocess(self, raw_data_df: pd.DataFrame, **kwargs):

        # Get battles data
        raw_data_df = raw_data_df.sort_values(ascending=True, by=['tstamp'])
        battles = raw_data_df[raw_data_df['anony']].reset_index(drop=True)

        return battles

    def compute_elo_rating(self, raw_data):
        battles = self.preprocess(raw_data_df=raw_data)
        elo_ratings = compute_elo(battles)
        col_model = 'Model'
        col_elo_rating = 'Elo_Rating'
        elo_ratings_res = pd.DataFrame([[n, elo_ratings[n]] for n in elo_ratings.keys()],
                                       columns=[col_model, col_elo_rating]).sort_values(
                                           col_elo_rating, ascending=False).reset_index(drop=True)
        elo_ratings_res = elo_ratings_res.round({col_elo_rating: 1})
        return elo_ratings_res

    def get_single_pairwise_rating(self, row: pd.Series):
        tie = False
        if 'win' in row:
            win = row['win']
            if win == 'tie':
                tie = True
            else:
                if win == 'model_a':
                    winner = row['model_a']
                    loser = row['model_b']
                else:
                    winner = row['model_b']
                    loser = row['model_a']
        elif 'win_1' in row:
            win_1 = row['win_1']
            win_2 = row['win_2']
            if win_1 == 'tie' or win_1 != win_2:
                tie = True
            else:
                if win_1 == 'model_a':
                    winner = row['model_a']
                    loser = row['model_b']
                else:
                    winner = row['model_b']
                    loser = row['model_a']
        else:
            raise ValueError('Unsupported data format')

        if tie:
            return [{
                'model': row['model_a'],
                'win': 0,
                'loss': 0,
                'tie': 1
            }, {
                'model': row['model_b'],
                'win': 0,
                'loss': 0,
                'tie': 1
            }]
        else:
            return [{'model': winner, 'win': 1, 'loss': 0, 'tie': 0}, {'model': loser, 'win': 0, 'loss': 1, 'tie': 0}]

    def compute_pairwise_rating(self, raw_data):
        df_all = self.preprocess(raw_data_df=raw_data)
        model_list = (df_all['model_a'].unique().tolist() + df_all['model_b'].unique().tolist())
        model_list = list(set(model_list))

        list_res = []
        # traverse df row by row
        for index, row in df_all.iterrows():
            if self.baseline_model is not None:
                if self.baseline_model not in [row['model_a'], row['model_b']]:
                    logger.warning(
                        f'One of the models in the battle should be the baseline model: {self.baseline_model}')
                    continue
            rating = self.get_single_pairwise_rating(row)
            list_res = list_res + rating

        df = pd.DataFrame(list_res)
        df = df.groupby(['model']).sum()

        # remove baseline model
        if self.baseline_model is not None:
            df = df[df.index != self.baseline_model]
        # add win rate
        df['win_rate'] = df['win'] / (df['win'] + df['loss'] + df['tie'])
        df['loss_rate'] = df['loss'] / (df['win'] + df['loss'] + df['tie'])
        df['tie_rate'] = df['tie'] / (df['win'] + df['loss'] + df['tie'])
        return df.sort_values(by='win_rate', ascending=False)

    def compute_score_rating(self, raw_data):
        df_all = self.preprocess(raw_data_df=raw_data)
        df = df_all[['model', 'score']]

        df_score = df.groupby(['model']).mean()
        return df_score.sort_values(by='score', ascending=False)

    def eval_samples(self, data_list: list):
        res_all = []

        raw_data: pd.DataFrame = None

        if len(data_list) > 0:
            raw_data = data_list[0]

        for metric in self.metrics:

            if metric == MetricMembers.ELO:
                res = self.compute_elo_rating(raw_data)
                res_all.append(res)

            elif metric == MetricMembers.PAIRWISE:
                res = self.compute_pairwise_rating(raw_data)
                res_all.append(res)

            elif metric == MetricMembers.SCORE:
                res = self.compute_score_rating(raw_data)
                res_all.append(res)

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
