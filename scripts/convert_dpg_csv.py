#!/usr/bin/env python
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Convert the official DPG-Bench ``dpg_bench.csv`` into a prompt-level JSONL dataset.

Each output line carries the prompt and its atomic question metadata, which the
``dpg_bench`` adapter serializes into the sample target. Source CSV:
https://github.com/End2End-Diffusion/diffusionbench-evals/raw/main/dpgbench/dpg_evaluator/dpg_bench.csv
(matching the official TencentQQGYLab/ELLA DPG-Bench metadata).

Usage: python scripts/convert_dpg_csv.py dpg_bench.csv dpg_bench.jsonl
"""

import csv
import json
import sys


def main(csv_path: str, out_path: str) -> None:
    rows = list(csv.DictReader(open(csv_path, encoding='utf-8')))
    by_item: dict = {}
    order: list = []
    for row in rows:
        item_id = row['item_id']
        if item_id not in by_item:
            by_item[item_id] = {'item_id': item_id, 'prompt': row['text'], 'questions': []}
            order.append(item_id)
        dependency = [int(d) for d in str(row.get('dependency', '')).split(',') if d.strip().isdigit()]
        by_item[item_id]['questions'].append(
            {
                'proposition_id': int(row['proposition_id']),
                'question': row['question_natural_language'],
                'dependency': dependency,
                'category_broad': row['category_broad'],
                'category_detailed': row['category_detailed'],
            }
        )
    with open(out_path, 'w', encoding='utf-8', newline='\n') as f:
        for item_id in order:
            f.write(json.dumps(by_item[item_id], ensure_ascii=False) + '\n')
    print(f'{len(order)} prompts -> {out_path}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
