# Copyright (c) Alibaba, Inc. and its affiliates.

import jieba
from collections import defaultdict
from rouge_chinese import Rouge
from statistics import mean
from tqdm import tqdm

from evalscope.constants import MetricsConstant
from evalscope.metrics.bundled_rouge_score import rouge_scorer
from evalscope.utils.logger import get_logger

logger = get_logger()


class DummyTokenizer:

    def tokenize(self, text: str):
        return text.split()


def is_contains_chinese(strs):
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            return True
    return False


def compute_rouge_score(predict_l, reference_l):
    assert len(predict_l) == len(reference_l)
    if len(predict_l) == 0:
        tmp_d = dict()
        for key in MetricsConstant.ROUGE_KEYS:
            tmp_d[key] = 0
        return tmp_d

    result = defaultdict(list)
    for p, r in tqdm(zip(predict_l, reference_l)):
        one_sample = compute_rouge_score_one_sample(p, r)
        for rouge_key in MetricsConstant.ROUGE_KEYS:
            result[rouge_key].append(one_sample[rouge_key])
    rlt = {}
    for rouge_key in MetricsConstant.ROUGE_KEYS:
        rlt[rouge_key] = (mean(result[rouge_key]) * 100 if rouge_key in result else MetricsConstant.INVALID_VALUE)
    return rlt


def compute_rouge_score_one_sample_zh(predict, reference):
    result = dict()
    zh_scorer = Rouge()
    for p, r in zip(predict, reference):
        p = ' '.join(jieba.cut(p)) if is_contains_chinese(p) else p
        r = ' '.join(jieba.cut(r)) if is_contains_chinese(r) else r

        try:
            score = zh_scorer.get_scores(p, r, ignore_empty=True)[0]
        except Exception as e:
            logger.warning(f'rouge score error: {p} {r} {e}')
            continue
        result['Rouge-1-R'] = score['rouge-1']['r']
        result['Rouge-1-P'] = score['rouge-1']['p']
        result['Rouge-1-F'] = score['rouge-1']['f']
        result['Rouge-2-R'] = score['rouge-2']['r']
        result['Rouge-2-P'] = score['rouge-2']['p']
        result['Rouge-2-F'] = score['rouge-2']['f']
        result['Rouge-L-R'] = score['rouge-l']['r']
        result['Rouge-L-P'] = score['rouge-l']['p']
        result['Rouge-L-F'] = score['rouge-l']['f']

    return result


def compute_rouge_score_one_sample(predict, reference):
    result = dict()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], tokenizer=DummyTokenizer())
    for p, r in zip(predict, reference):
        try:
            score = scorer.score(p, r)
        except Exception as e:
            logger.warning(f'rouge score error: {p} {r} {e}')
            continue
        result['rouge-1-r'] = score['rouge1'].recall
        result['rouge-1-p'] = score['rouge1'].precision
        result['rouge-1-f'] = score['rouge1'].fmeasure
        result['rouge-2-r'] = score['rouge2'].recall
        result['rouge-2-p'] = score['rouge2'].precision
        result['rouge-2-f'] = score['rouge2'].fmeasure
        result['rouge-l-r'] = score['rougeL'].recall
        result['rouge-l-p'] = score['rougeL'].precision
        result['rouge-l-f'] = score['rougeL'].fmeasure

    return result


def _to_table(final_result) -> str:
    table = []
    # step 1. table header
    all_tasks = ['', 'total']
    all_tasks.extend(final_result['all_tasks'].split(','))
    table.append('\t'.join(all_tasks))

    # step 2. table row
    for rouge_key in MetricsConstant.ROUGE_KEYS:
        row = [rouge_key]
        for task in all_tasks:
            if not task:
                continue
            elif task == 'total':
                row.append(f'{final_result["total"]["rouge"][rouge_key]:0.2f}')
            else:
                row.append(f'{final_result["tasks"][task]["rouge"][rouge_key]:0.2f}')
        table.append('\t'.join(row))

    return '\n'.join(table)


def run_rouge_eval(data_l, md_level=2, report_metric_key='rouge-l-f'):
    print(f"{'#' * md_level} Rouge Eval")
    for data in tqdm(data_l):
        data['rouge'] = compute_rouge_score_one_sample(data['gen_tok_str'], data['reference_tok_str'])
    task_data_d = defaultdict(list)
    for data in data_l:
        for task in data['task_tags']:
            task_data_d[task].append(data)

    total_rouge = mean([data['rouge'][report_metric_key] for data in data_l])
    print(f'[total], count: {len(data_l)}, {report_metric_key}: '
          f'{total_rouge * 100:0.2f}%')

    for task, task_data in task_data_d.items():
        task_rouge = mean([data['rouge'][report_metric_key] for data in task_data])
        print(f'[{task}], count: {len(task_data_d[task])}, {report_metric_key}: '
              f'{task_rouge * 100:0.2f}%')
