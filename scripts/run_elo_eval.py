# Copyright (c) Alibaba, Inc. and its affiliates.

import os

from evals.evaluator.elo_rating_eval import EloRatingEvaluate


def main():
    # Supported columns for model battles:
    # 'model_a', 'model_b', 'win', 'tstamp', 'language'
    review_data_path = os.path.join(os.getcwd(),
                                    '../evals/registry/data/arena/reviews',
                                    'review_gpt4.jsonl')

    metrics = ['elo']
    ae = EloRatingEvaluate(metrics=metrics)
    res_list = ae.run(review_data_path)

    print(res_list[0])


if __name__ == '__main__':
    main()
