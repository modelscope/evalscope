# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import json
import glob
from tabulate import tabulate

"""
Combine and generate table for reports of LLMs.
"""


def get_report(report_file: str):
    data_d: dict = json.load(open(report_file, 'r'))
    dataset_name = data_d['name']
    score = data_d['score']     # float or dict
    score_d = {}
    if isinstance(score, dict):
        # score_d = dict([(k, round(v, 4) * 100) for k, v in score.items()])
        score_d = score
    elif isinstance(score, float):
        # score_d['acc'] = round(score, 4) * 100
        score_d['acc'] = score
    else:
        raise ValueError(f'Unknown score type: {type(score)}')
    score_str = '\n'.join([str(v) + ' (' + k + ')' for k, v in score_d.items()])

    return {'dataset_name': dataset_name, 'score': score_str}


def get_model_reports(model_report_dir: str):
    model_report_dir = os.path.normpath(model_report_dir)
    model_report_dir = model_report_dir.rstrip('reports')
    model_info = os.path.basename(os.path.normpath(model_report_dir))
    model_name = '_'.join(model_info.split('_')[:-1][3:])
    report_files = glob.glob(os.path.join(model_report_dir, 'reports', '*.json'))

    model_reports_d = {model_name: []}
    for file_path in report_files:
        report_d = get_report(file_path)
        model_reports_d[model_name].append(report_d)

    return model_reports_d


def gen_table(reports_path_list: list):
    table_values = []
    headers = ['Model']
    is_headers_set = False

    for report_path in reports_path_list:
        model_reports_d = get_model_reports(report_path)
        for model_name, report_list in model_reports_d.items():
            # report_list: [{'dataset_name': 'CompetitionMath', 'score': '4.42 (acc)'},
            #               {'dataset_name': 'GSM8K', 'score': '28.51 (acc)'}]
            report_list = sorted(report_list, key=lambda x: x['dataset_name'])
            if not is_headers_set:
                headers.extend([x['dataset_name'] for x in report_list])
                is_headers_set = True
            single_row = []
            single_row.append(model_name)
            for single_report in report_list:
                # e.g. '28.51 (acc)'
                single_row.append(single_report['score'])
            table_values.append(single_row)

    report_table = tabulate(table_values, headers=headers, tablefmt='grid')
    return report_table


if __name__ == '__main__':
    report_dir_1 = '/to/path/20231129_020533_default_ZhipuAI_chatglm2-6b-base_none/reports'
    report_dir_2 = '/to/path/20231129_020533_default_ZhipuAI_chatglm2-6b_none/reports'

    report_table = gen_table([report_dir_1, report_dir_2])
    print(report_table)

    # ALL VALUES ONLY FOR EXAMPLE
    # +--------------------------+-------------------+-------------+
    # | Model                    | CompetitionMath   | GSM8K       |
    # +==========================+===================+=============+
    # | ZhipuAI_chatglm2-6b-base | 25.0 (acc)        | 30.50 (acc) |
    # +--------------------------+-------------------+-------------+
    # | ZhipuAI_chatglm2-6b      | 30.5 (acc)        | 40.50 (acc) |
    # +--------------------------+-------------------+-------------+
