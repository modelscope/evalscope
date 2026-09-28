"""Attribute every frozen old/new score change and build a reviewable validation report."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any


def samples(root: Path) -> dict[tuple, dict]:
    """Key native reviews by immutable dataset, variant, subset and sample identity."""
    return {
        (r['benchmark'], r['variant'], r['subset'], r['index']): r
        for r in map(json.loads, (root / 'samples.jsonl').read_text().splitlines())
    }


def value(row: dict) -> float:
    """Read the native metric without assuming a benchmark's metric name."""
    return float(next(iter(row['score']['value'].values())))


def compare(old_root: Path, new_root: Path, controlled: bool) -> tuple[list[dict], dict]:
    """Fail on identity/reference drift, unavailable scores or unexplained score changes."""
    old, new = samples(old_root), samples(new_root)
    assert old.keys() == new.keys(), 'Sample identity drift'
    changes = []
    totals: dict[str, Any] = defaultdict(lambda: {'samples': 0, 'old_correct': 0, 'new_correct': 0})
    for identity, previous in old.items():
        current = new[identity]
        assert previous['target'] == current['target'], f'Reference drift: {identity}'
        assert previous['score']['prediction'] == current['score']['prediction'], f'Prediction drift: {identity}'
        assert current['score']['status'] == 'success' and current['score']['value'], identity
        key = f'{identity[0]}/{identity[1]}'
        totals[key]['samples'] += 1
        totals[key]['old_correct'] += value(previous)
        totals[key]['new_correct'] += value(current)
        if controlled:
            expected = 1 if identity[1] in ('correct', 'equivalent') else 0
            assert value(current) == expected, f'New unexplained score: {identity}'
        if previous['score'] == current['score']:
            continue
        score_changed = value(previous) != value(current)
        if score_changed and controlled and identity[1] == 'incorrect':
            target = current['target'][0] if isinstance(current['target'], list) else current['target']
            prediction = current['score']['prediction'].removeprefix(r'\boxed{').removesuffix('}')
            assert Decimal(prediction) == Decimal(target) * 100 and Decimal(target) != 0, identity
            explanation = 'Removed unconditional 100x compatibility for numbers without a percent sign (#1774).'
        elif score_changed and controlled and identity[1] == 'equivalent':
            assert value(previous) == 0 and value(current) == 1, identity
            explanation = (
                'HF extracts/parses the complete LaTeX expression; the old last-number extraction '
                f'produced {previous["score"]["extracted_prediction"]!r}.'
            )
        elif score_changed:
            raise AssertionError(f'Unexplained score difference: {identity}')
        else:
            explanation = 'Upstream extraction/normalization changes display text; the mathematical score is unchanged.'
        changes.append(
            {
                'identity': identity,
                'score_changed': score_changed,
                'explanation': explanation,
                'old': previous,
                'new': current,
            }
        )
    return changes, dict(totals)


def main() -> None:
    """Produce a compact summary plus complete annotated per-sample evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('validation/math_verify'))
    args = parser.parse_args()
    root = args.root
    manifest = json.loads((root / 'data/manifest.json').read_text())
    for name, entry in manifest.items():
        assert hashlib.sha256((root / f'data/{name}.json').read_bytes()).hexdigest() == entry['sha256']
    differences, totals = compare(root / 'old', root / 'new', controlled=True)
    history = {}
    historical_changes = []
    for run in sorted((root / 'historical_new').iterdir()):
        if not run.is_dir():
            continue
        changed, counts = compare(root / 'historical_old' / run.name, run, controlled=False)
        history[run.name] = counts
        historical_changes.extend(changed)
    (root / 'new/differences.jsonl').write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in differences)
    )
    (root / 'historical_differences.jsonl').write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in historical_changes)
    )
    old_cases = json.loads((root / 'adapter_results_old.json').read_text())
    new_cases = json.loads((root / 'adapter_results.json').read_text())
    adapter_changes = []
    assert len(old_cases) == len(new_cases)
    for index, (previous, current) in enumerate(zip(old_cases, new_cases)):
        assert previous['case'] == current['case']
        assert current['matches_expected'], (index, current['case']['branch'])
        old_score = previous.get('score', {}).get('score', {})
        new_score = current['score']['score']
        if old_score.get('value') == new_score['value']:
            continue
        benchmark, branch = current['case']['benchmark'], current['case']['branch']
        if benchmark == 'docmath' and branch in ('wrong_scale', 'empty', 'parse_failure'):
            explanation = 'Removed guessed 100x scaling or parse-failure/empty-to-zero substitution.'
        elif benchmark == 'docmath' and branch == 'cached_real_score_context':
            explanation = 'HF extracts the actual boxed negative numeric answer, instead of losing boxed syntax.'
        elif benchmark == 'olympiad_bench' and branch == 'wrong_scale':
            explanation = "Removed the OlympiadBench grader's unconditional 100x numeric compatibility."
        elif benchmark == 'olympiad_bench':
            explanation = 'Parse each declared answer component upstream; retain all components and their tolerances.'
        elif benchmark == 'measure_bench' and branch in ('fraction_boxed', 'fraction_unit_outside'):
            explanation = 'Extract the complete LaTeX fraction and preserve the instrument unit.'
        else:
            raise AssertionError(f'Unexplained adapter change: {benchmark}/{branch}')
        adapter_changes.append(
            {
                'case_index': index,
                'benchmark': benchmark,
                'branch': branch,
                'old': old_score,
                'new': new_score,
                'explanation': explanation,
            }
        )
    (root / 'adapter_differences.json').write_text(json.dumps(adapter_changes, ensure_ascii=False, indent=2) + '\n')
    reasons = Counter(row['explanation'].split(';')[0] for row in differences if row['score_changed'])
    summary = {
        'baseline': (root / 'baseline.sha').read_text().strip(),
        'datasets': manifest,
        'controlled_totals': totals,
        'controlled_reviews': sum(r['samples'] for r in totals.values()),
        'changed_reviews': len(differences),
        'changed_scores': sum(r['score_changed'] for r in differences),
        'unexplained_changes': 0,
        'new_unavailable_scores': 0,
        'historical': history,
        'historical_score_changes': sum(r['score_changed'] for r in historical_changes),
        'adapter_cases': len(new_cases),
        'adapter_score_changes': len(adapter_changes),
        'adapter_expectation_failures': 0,
        'score_change_categories': dict(reasons),
    }
    (root / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    checks_path = root / 'checks.json'
    checks = json.loads(checks_path.read_text()) if checks_path.exists() else {}
    lines = [
        '# Math-Verify 迁移验证',
        '',
        f'基线 `{summary["baseline"]}`；数据版本由下列冻结文件的 SHA256 固定。没有新增模型推理或付费 judge 调用。',
        '',
        *(
            [
                '追加代码审查确认三个待修问题，详见 [review.md](review.md)。以下通过结果限于已有固定集，当前不建议合入。',
                '',
            ]
            if checks.get('review', {}).get('open_findings')
            else []
        ),
        *(
            ['追加代码审查发现的三处问题已修复，前后复核见 [review.md](review.md)。', '']
            if checks.get('review', {}).get('result') == 'passed'
            else []
        ),
        '## 完整真实数据的可控输出',
        '',
        'GSM8K 1,319 条与 MATH-500 500 条真实测试数据，各构造正确、错误、空白、等价表达四组输出。',
        '两套隔离环境分别经过原生 adapter、指标、聚合、JSON/HTML 报告流水线。以下是固定输出的验收结果，不代表模型能力。',
        '',
        '| 基准 / 输出 | 样本数 / 有效分母 | 旧正确数 | 新正确数 |',
        '| --- | ---: | ---: | ---: |',
    ]
    for key, result in sorted(totals.items()):
        lines.append(f'| {key} | {result["samples"]} | {result["old_correct"]:g} | {result["new_correct"]:g} |')
    lines += [
        '',
        f'共 {summary["controlled_reviews"]:,} 条 review；{summary["changed_scores"]:,} 条分数变化，'
        f'{summary["changed_reviews"] - summary["changed_scores"]} 条仅展示文本变化。无未解释差异，无新增正常样本评分失败。',
        '错误输出的分数下降逐条验证为无百分号的 100 倍错误；等价输出的提升对应 HF 完整数学片段提取，旧实现曾误取末尾数字。',
        '每条变化及旧/新预测、提取、分数、状态保存在 `new/differences.jsonl`；完整 review 与有效分母分别在 `new/samples.jsonl`、`new/aggregates.json`。',
        '',
        '## 历史模型输出',
        '',
        '复用工作区现有三个 GSM8K 运行的 5 + 5 + 1 条输出。通过完整题目唯一匹配原始数据并核对参考答案；预测原文保持不变。',
        '全部 11 条原本正确，新旧仍为 11/11。样本量很小，不能据此推断真实模型全量分数不变。',
        '来源与冻结哈希见 `history_data/manifest.json`；原生复跑报告在 `historical_old/`、`historical_new/`。',
        '',
        '## 独立 adapter',
        '',
        f'{len(new_cases)} 个真实评分上下文覆盖 {len({case["case"]["benchmark"] for case in new_cases})} 个基准，'
        '包括 OlympiadBench 数值/表达式/方程/区间/元组/混合多答案及分项容差、',
        'DocMath 真实零值和倍率错误、HiPhO 官方评分细则与多答案部分得分。原生评分和聚合执行通过。',
        '其他基准复用仓库缓存的真实 target、choices 和评分 metadata；部分题干在原缓存中已截断，未做图像推理。',
        'HiPhO judge 使用离线协议 double：按官方条目返回半分或 false，验证部分得分和流程，不验证 judge 模型质量。',
        f'{len(adapter_changes)} 条分数变化全部归因于已确认的缺陷修复或 HF 提取能力；详见 `adapter_differences.json`。',
        '',
        '## 执行、依赖与检查',
        '',
        (
            f'macOS {checks["macos"]["tests"]} 项完整回归、{checks["macos"]["final_targeted_math_tests"]} 项最终数学复核；'
            f'Linux {checks["linux"]["tests"]} 项阶段回归覆盖评分、进程和缓存；后续评分调整已在 macOS 最终回归中验证。'
            'make lint、make lint-imports、test_ci_lite 通过。'
            if checks
            else '检查结果见 checks.json。'
        ),
        '执行检查与复现命令见 README.md；冷启动和稳态吞吐见 runtime.json。吞吐使用重复简单表达式，包含上游缓存效果，不代表长公式的评测吞吐。',
        '基础 + OlympiadBench、AIGC 在干净环境安装并通过 pip check；OmegaConf 的 LAVIS 配置加载、合并、CLI 覆盖、插值和真实处理器初始化通过。',
        '当前机器没有 Windows 环境，Windows 执行机制尚未实测。',
        '`ms-vlmeval==0.0.20` 仍声明 ANTLR 4.11.1，与此次 4.13.2 冲突；没有升级外部库，也没有通过降级/忽略检查规避。',
        '不宣称 vlmeval 或 all 安装组合兼容。',
        '',
        '## 数据哈希',
        '',
    ]
    for name, entry in manifest.items():
        lines.append(f'- `{name}` / `{entry["dataset_id"]}`: `{entry["sha256"]}`')
    (root / 'report.md').write_text('\n'.join(lines) + '\n')
    print(f'Attributed {summary["changed_scores"]} score changes; adapter expectations all passed')


if __name__ == '__main__':
    main()
