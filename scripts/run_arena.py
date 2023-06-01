# Copyright (c) Alibaba, Inc. and its affiliates.

import os

from evals.evaluator.arena import ArenaEvaluate


def main():
    raw_data_path = os.path.join(os.getcwd(), '../evals/registry/data/arena', 'model_battles_examples.json')
    metrics = ['elo']
    ae = ArenaEvaluate(metrics=metrics)
    res_list = ae.run(raw_data_path)

    print(res_list[0])


if __name__ == '__main__':
    main()
