# Copyright (c) Alibaba, Inc. and its affiliates.

import re
from collections import defaultdict

from tqdm import tqdm

from evalscope.constants import MetricsConstant


def get_last_number(s):
    match = re.search(r'[-+]?\d*\.\d+|\d+', s[::-1])
    if match:
        last_digit = match.group()[::-1]
    else:
        last_digit = -100000
    return float(last_digit)


def compute_math_accuracy_one_sample(predict, reference):
    if isinstance(predict, list):
        predict = predict[0]
    if isinstance(reference, list):
        reference = reference[0]
    predict_number = get_last_number(predict)
    reference_number = get_last_number(reference)
    if abs(predict_number - reference_number) <= MetricsConstant.EPSILON:
        return 1
    else:
        return 0


def compute_math_accuracy(predict_l, reference_l):
    assert len(predict_l) == len(reference_l)
    if len(predict_l) == 0:
        return 0
    total_cnt = len(predict_l)
    correct_cnt = 0
    for predict, reference in zip(predict_l, reference_l):
        correct_cnt += compute_math_accuracy_one_sample(predict, reference)
    return {'math accuracy': correct_cnt / total_cnt}


def run_math_eval(data_l, md_level=2):
    print(f"{'#' * md_level} Math Eval(math accuracy)")
    for data in tqdm(data_l):
        data['math_accuracy'] = compute_math_accuracy_one_sample(
            data['gen'], data['target'])
    task_data_d = defaultdict(list)
    for data in data_l:
        for task in data['task_tags']:
            task_data_d[task].append(data)
    correct_cnt = sum([data['math_accuracy'] for data in data_l])
    print(f'[total], count: {len(data_l)}, math accuracy: '
          f'{correct_cnt / len(data_l) * 100:0.2f}%')
    for task in task_data_d.keys():
        correct_cnt = sum(
            [data['math_accuracy'] for data in task_data_d[task]])
        print(f'[{task}], count: {len(task_data_d[task])}, math accuracy: '
              f'{correct_cnt/len(task_data_d[task])*100:0.2f}%')
