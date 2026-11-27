#!/usr/bin/env python
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Convert the official DPG-Bench ``dpg_bench.csv`` into a prompt-level JSONL dataset.

Each output line carries the prompt and its atomic question metadata. To use
the exact prompts, run against the CSV in the original ELLA checkout:
https://github.com/TencentQQGYLab/ELLA/tree/main/dpg_bench

Usage: python scripts/convert_dpg_csv.py dpg_bench.csv dpg_bench.jsonl
"""

import json
import sys

from evalscope.benchmarks.text2image.dpg_bench_adapter import convert_dpg_csv


def main(csv_path: str, out_path: str) -> None:
    """Write the canonical prompt-level records."""
    records = convert_dpg_csv(csv_path)
    with open(out_path, 'w', encoding='utf-8', newline='\n') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f'{len(records)} prompts -> {out_path}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
