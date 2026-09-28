"""Generate migration metadata/docs with unchanged statistics and offline translation.

Dataset statistics and existing Chinese prose are reused because only scoring changes.
No model API is called. Generated artifacts are written through repository generators.
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

from evalscope.utils.doc_utils import load_benchmark_data, save_benchmark_data
from evalscope.utils.doc_utils.generate_dataset_md import generate_docs, update_benchmark_data

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    """Refresh English metadata and append reviewed Chinese scoring changes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, help='Git revision containing the original metadata')
    parser.add_argument('--benchmarks', nargs='+', required=True, help='Benchmark names to regenerate')
    args = parser.parse_args()
    names = args.benchmarks
    baseline = args.baseline
    previous = {
        name: json.loads(
            subprocess.check_output(
                ['git', 'show', f'{baseline}:evalscope/benchmarks/_meta/{name}.json'],
                cwd=ROOT,
                text=True,
            )
        )
        for name in names
    }
    update_benchmark_data(benchmark_name=names, force=False, compute_stats=False, workers=4)
    data = load_benchmark_data()
    for name in names:
        original = previous[name]
        entry = data[name]
        assert entry['statistics'] == original['statistics']
        assert entry['sample_example'] == original['sample_example']
        chinese = original['readme']['zh']
        if name in ('aime25', 'aime26'):
            chinese = chinese.replace(
                '- 使用LLM-as-judge进行数学等价性检查',
                '- 使用LLM-as-judge进行数学等价性检查；规则评分要求整数常量。',
            )
        notes = ''
        if name == 'olympiad_bench':
            notes += (
                '- 数值比较保留数据声明的绝对误差及分项容差（默认 `1e-8`）；多答案不复用候选，不猜测 100 倍倍率。\n'
            )
        if name == 'docmath':
            notes += '- 规则回退保留 `0.0015` 的相对容差和布尔类型判断，不猜测倍率、不将解析失败补成 0。\n'
        if notes:
            match = re.search(r'^## .*?(?:评估|评测|评价).*$', chinese, re.MULTILINE)
            if match is None:
                raise ValueError(f'Missing translated evaluation heading: {name}')
            # Repeated invocations regenerate from the baseline translation, not from appended notes.
            chinese = chinese[: match.end()] + '\n\n' + notes + chinese[match.end() :].lstrip('\n')
        entry['readme']['zh'] = chinese
        entry['readme']['needs_translation'] = False
        if {key: value for key, value in entry.items() if key != 'updated_at'} == {
            key: value for key, value in original.items() if key != 'updated_at'
        }:
            entry['updated_at'] = original['updated_at']
        save_benchmark_data(entry, name)
        data[name] = entry
    generate_docs(data)


if __name__ == '__main__':
    main()
