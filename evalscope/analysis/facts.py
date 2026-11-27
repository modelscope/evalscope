"""Deterministic facts for ordinary evaluations and agents; no model calls."""

import math
from collections import Counter, defaultdict
from statistics import mean
from typing import List

from evalscope.api.analysis import AnalysisReportConfig, DiagnosisCoverage, Fact
from evalscope.api.analysis.trace import FindingCategory
from evalscope.constants import DEFAULT_LANGUAGE
from evalscope.metrics.semantics.ranking import bounded_quality_ratio

from .inputs import BenchmarkInput, SampleInput, is_choice_letter, unsupported_content
from .trace.reader import has_nontext_output

USABLE = {'success', 'fallback', 'degraded'}


def collect_facts(inputs: BenchmarkInput, config: AnalysisReportConfig, coverage: DiagnosisCoverage) -> List[Fact]:
    """Compute each denominator from available evidence; missing data is recorded, not zero-filled."""
    zh = config.language == 'zh' or (config.language == 'auto' and DEFAULT_LANGUAGE == 'zh')
    facts = []
    samples = inputs.samples
    report = inputs.report
    nontext = sum(has_nontext_output(s.record) for s in samples)
    if nontext:
        coverage.skipped_facts['nontext-output'] = (
            f'{nontext} 条样本包含非文本输出，未参与空文本、答案抽取和文本长度统计。'
            if zh
            else f'{nontext} nontext outputs excluded from text generation statistics.'
        )
    unsupported = Counter(kind for s in samples for kind in unsupported_content(s.record))
    if unsupported:
        coverage.skipped_facts['unsupported-content'] = (
            '媒体、内部数据和受保护推理不参与文本分析：'
            if zh
            else 'Media, internal data and protected reasoning are not analyzed: '
        ) + ', '.join(f'{kind}: {count}' for kind, count in sorted(unsupported.items()))

    def add(
        identity: str,
        kind: str,
        title: tuple,
        definition: tuple,
        value: float = None,
        numerator: int = None,
        denominator: int = None,
        examples: List[SampleInput] = None,
        data: dict = None,
        unit: str = '',
        warn: bool = False,
    ) -> None:
        facts.append(
            Fact(
                id=identity,
                kind=kind,
                title=title[0 if zh else 1],
                definition=definition[0 if zh else 1],
                value=value,
                numerator=numerator,
                denominator=denominator,
                unit=unit,
                severity='warn' if warn else 'info',
                samples=[s.key for s in (examples or [])[: config.max_examples]],
                data=data or {},
            )
        )

    def rate(identity: str, kind: str, title: tuple, definition: tuple, selected: list, available: list) -> None:
        if not available:
            coverage.skipped_facts[identity] = 'No eligible samples with the required evidence.'
            return
        value = len(selected) / len(available)
        add(
            identity,
            kind,
            title,
            definition,
            value,
            len(selected),
            len(available),
            selected,
            unit='fraction',
            warn=value >= config.warning_rate and bool(selected),
        )

    known = [s for s in samples if s.case.outcome.task_success is not None]
    rate(
        'F-task-failure',
        'generic',
        ('任务未通过', 'Task failure'),
        (
            '分母为有明确任务成败结果的样本；未评分不当作通过或失败。',
            'Samples with a known task outcome; unavailable scores are neither passes nor failures.',
        ),
        [s for s in known if s.case.outcome.task_success is False],
        known,
    )
    rate(
        'F-excluded',
        'generic',
        ('评分被排除', 'Excluded scores'),
        ('分母为全部可读取 review；excluded 不等于答错。', 'All readable reviews; excluded is not incorrect.'),
        [s for s in samples if s.case.score_status not in USABLE],
        samples,
    )
    excluded_reasons = Counter()
    missing_reasons = 0
    for sample in samples:
        if sample.case.score_status in USABLE:
            continue
        if recorded := sample.recorded_error:
            excluded_reasons[recorded[1]] += 1
        else:
            missing_reasons += 1
    if missing_reasons:
        coverage.skipped_facts['score-error-reasons'] = f'{missing_reasons} excluded scores have no recorded reason.'
    add(
        'F-score-status',
        'generic',
        ('评分状态', 'Score status'),
        (
            '按 review 状态计数，另列未产生 review 的执行错误。',
            'Review statuses, with execution errors lacking a review counted separately.',
        ),
        data={
            'statuses': dict(Counter(s.case.score_status for s in samples)),
            'errored_without_record': coverage.errored_without_record,
            'reasons': dict(excluded_reasons),
        },
    )
    if coverage.errored_without_record:
        coverage.skipped_facts['execution-errors'] = 'Execution error reasons were not persisted in review records.'
    metric = next((m for m in report.metrics if m.identity == report.primary_metric_identity), None) if report else None
    values = defaultdict(list)
    score_examples = []
    primary_name = metric.identity.name if metric else None
    direction = metric.semantics.direction.value if metric else 'none'
    for s in samples:
        if s.case.score_status in USABLE:
            for name, value in s.case.score_value.items():
                if math.isfinite(value):
                    values[name].append(value)
                    if name == primary_name and direction != 'none' and s.case.outcome.task_success is None:
                        score_examples.append((-value if direction == 'lower_is_better' else value, s))
    if values:
        distributions = []
        for name in sorted(values, key=lambda name: (name != primary_name, name)):
            items = values[name]
            low, high = min(items), max(items)
            bins = Counter(min(9, int((v - low) / (high - low) * 10)) if high > low else 0 for v in items)
            distributions.append(
                {
                    'metric': name,
                    'n': len(items),
                    'mean': mean(items),
                    'min': low,
                    'max': high,
                    'bins': [{'bin': b, 'count': count} for b, count in sorted(bins.items())],
                }
            )
        add(
            'F-scores',
            'generic',
            ('样本评分分布', 'Sample score distributions'),
            ('各指标仅包含有效评分；不同指标不混算。', 'Usable scores by metric; different metrics are never pooled.'),
            examples=[
                sample
                for _, sample in sorted(
                    score_examples, key=lambda item: (item[0], item[1].key.subset, item[1].key.index)
                )[: config.max_examples]
            ],
            data={'metrics': distributions},
        )
    else:
        coverage.skipped_facts['F-scores'] = 'No usable finite sample scores.'

    safety = [s for s in samples if s.case.outcome.safety_violation is not None]
    if safety:
        rate(
            'F-safety',
            'agent',
            ('Provider 安全信号', 'Provider safety signal'),
            (
                '按基准 provider 的分析约定判定；独立于任务成败，不等同于通用越权判决。',
                'Benchmark-provider analysis convention; separate from task success, not a general authorization verdict.',
            ),
            [s for s in safety if s.case.outcome.safety_violation],
            safety,
        )

    if report and report.judge_summary:
        judge = report.judge_summary
        add(
            'F-judge',
            'judge',
            ('评分模型覆盖率', 'Judge coverage'),
            (
                '来自报告的 judge_summary，与诊断模型成本分开。',
                'From report judge_summary; separate from diagnosis usage.',
            ),
            judge.coverage,
            judge.scored,
            judge.total,
            data={'failures': judge.failures},
            unit='fraction',
        )
    else:
        coverage.skipped_facts['F-judge'] = 'No judge_summary in the source report.'
    if report and report.perf_metrics:
        summary = report.perf_metrics.get('summary') or {}
        usage = summary.get('usage') or {}
        add(
            'F-perf',
            'perf',
            ('评测性能与用量', 'Evaluation performance and usage'),
            (
                '仅展示源报告中的评测性能；不推算缺失成本。',
                'Source evaluation performance; missing costs are not estimated.',
            ),
            data={
                'requests': summary.get('n_samples'),
                'input_tokens': usage.get('total_input_tokens'),
                'output_tokens': usage.get('total_output_tokens'),
                'latency_mean_s': (summary.get('latency') or {}).get('mean'),
                'coverage': report.perf_metrics.get('coverage'),
            },
        )
    else:
        coverage.skipped_facts['F-perf'] = 'No performance metrics in the source report.'
    if metric:
        overall = bounded_quality_ratio(metric.score, metric.semantics)
        weak = []
        for category in metric.categories:
            for subset in category.subsets:
                quality = bounded_quality_ratio(subset.score, metric.semantics)
                if overall is not None and quality is not None and subset.num >= config.weak_subset_min_samples:
                    gap = overall - quality
                    if gap >= config.weak_subset_gap or math.isclose(gap, config.weak_subset_gap):
                        weak.append(
                            {
                                'subset': subset.name,
                                'score': subset.score,
                                'n': subset.num,
                                'quality_gap': overall - quality,
                            }
                        )
        if overall is not None:
            add(
                'F-weak-subsets',
                'generic',
                ('较弱子集', 'Weaker subsets'),
                (
                    '按指标方向与声明区间归一化比较；只计样本数达标的子集。',
                    'Quality normalized by declared metric direction/range; only sufficiently large subsets.',
                ),
                data={'items': weak, 'min_samples': config.weak_subset_min_samples, 'gap': config.weak_subset_gap},
                warn=bool(weak),
            )
        else:
            coverage.skipped_facts['F-weak-subsets'] = 'Primary metric has no bounded quality range.'
    else:
        coverage.skipped_facts['F-weak-subsets'] = 'No declared primary metric in the source report.'

    generation = [
        s
        for s in samples
        if not s.case.tool_call_count
        and not s.record.agent_trace
        and not has_nontext_output(s.record)
        and any(m.role == 'assistant' for m in s.record.messages)
    ]
    rate(
        'F-empty',
        'generation',
        ('生成输出为空', 'Empty generation'),
        ('仅检查非 Agent 样本的可见助手输出。', 'Visible assistant output of non-agent samples only.'),
        [s for s in generation if not any(m.role == 'assistant' and (m.text or '').strip() for m in s.record.messages)],
        generation,
    )
    choice_task = inputs.is_multiple_choice
    extraction = [s for s in generation if s.record.sample_score.score.extracted_prediction is not None]
    rate(
        'F-extraction',
        'generation',
        ('未取得抽取答案', 'No extracted answer'),
        (
            '输出非空且评分协议提供 prediction，但没有可用抽取值；不证明答案正确或评分器有错。',
            'Nonempty output with a prediction field but no extracted value; not proof of a correct answer or judge bug.',
        ),
        [
            s
            for s in extraction
            if any(m.role == 'assistant' and (m.text or '').strip() for m in s.record.messages)
            and (
                not s.record.sample_score.score.extracted_prediction.strip()
                or (choice_task and not is_choice_letter(s.record.sample_score.score.extracted_prediction))
            )
        ],
        extraction,
    )
    recorded_endings = [s for s in generation if s.stop_reason and s.stop_reason != 'unknown']
    rate(
        'F-truncation',
        'generation',
        ('达到输出上限', 'Output limit reached'),
        (
            '仅统计有结束原因的文本样本；达到上限也可能已给出答案。',
            'Text samples with a recorded ending only. The answer may already be present when the limit is reached.',
        ),
        [s for s in recorded_endings if s.stop_reason in ('length', 'max_tokens', 'model_length')],
        recorded_endings,
    )
    rate(
        'F-refusal',
        'generation',
        ('检测到拒答文本', 'Refusal text detected'),
        (
            '文本规则，低置信度；不自动表示答错或安全违规。',
            'Low-confidence text rules; not automatically a task error or safety violation.',
        ),
        [s for s in generation if any(f.category == FindingCategory.REFUSAL for f in s.case.findings)],
        generation,
    )
    length_rows = [s for s in generation if s.case.outcome.task_success is not None]
    if length_rows:
        groups = defaultdict(list)
        for s in length_rows:
            size = sum(len(m.text or '') for m in s.record.messages if m.role == 'assistant')
            groups['short' if size < 500 else 'medium' if size < 2000 else 'long'].append(s.case.outcome.task_success)
        add(
            'F-length',
            'generation',
            ('输出长度与通过率', 'Output length and task success'),
            (
                '按可见字符数分组；只描述相关性，不推断因果。',
                'Visible-character bins; descriptive association, not causation.',
            ),
            data={'bins': [{'length': k, 'n': len(v), 'success_rate': mean(v)} for k, v in groups.items()]},
        )

    else:
        coverage.skipped_facts['F-length'] = 'No non-agent generations with known task outcomes.'

    pairs = []
    gold_choices = []
    for s in samples:
        gold = s.record.target or []
        pred = s.record.sample_score.score.extracted_prediction
        if pred is None:
            pred = s.record.sample_score.score.prediction
        if len(gold) == 1 and is_choice_letter(gold[0]):
            gold_choices.append(gold[0])
            if pred and is_choice_letter(pred):
                pairs.append((gold[0], pred.strip()))
    if choice_task:
        add(
            'F-mcq',
            'mcq',
            ('选项分布与混淆对', 'Choice distributions and confusion pairs'),
            (
                '至少90%的样本有单字母 gold；预测覆盖量单列，无效或缺失预测不补选项。',
                'At least 90% have single-letter gold; prediction coverage is separate and missing choices are not imputed.',
            ),
            data={
                'gold': dict(Counter(gold_choices)),
                'prediction': dict(Counter(p for g, p in pairs)),
                'confusions': [
                    {'gold': g, 'prediction': p, 'count': n}
                    for (g, p), n in Counter((g, p) for g, p in pairs if g != p).most_common(10)
                ],
                'covered': len(pairs),
                'total': len(samples),
            },
        )
    else:
        coverage.skipped_facts['F-mcq'] = 'Insufficient unambiguous single-letter gold/pred pairs.'
    if inputs.repeats > 1:
        repeated = defaultdict(list)
        for s in samples:
            score = s.record.sample_score
            if score.group_id is not None and score.generation_index is not None:
                repeated[(s.key.subset, str(score.group_id))].append(s)
        complete = [
            v
            for v in repeated.values()
            if len(v) == inputs.repeats
            and {s.record.sample_score.generation_index for s in v} == set(range(inputs.repeats))
            and all(s.case.outcome.task_success is not None for s in v)
        ]
        if complete:
            stable = sum(len({s.case.outcome.task_success for s in v}) == 1 for v in complete)
            add(
                'F-repeats',
                'repeats',
                ('重复采样稳定性', 'Repeated-sample consistency'),
                (
                    '仅统计所有计划轮次完整且成败已知的题组；缺轮次不补齐。',
                    'Complete planned generation groups with known outcomes only; missing attempts are not compacted.',
                ),
                stable / len(complete),
                stable,
                len(complete),
                unit='fraction',
                data={'unstable_groups': len(complete) - stable, 'incomplete_groups': len(repeated) - len(complete)},
            )
        else:
            coverage.skipped_facts['F-repeats'] = 'No complete repeated groups with known outcomes.'
    else:
        coverage.skipped_facts['F-repeats'] = 'repeats is not greater than one.'
    agent = [s for s in samples if s.case.tool_call_count or s.record.agent_trace]
    if agent:
        add(
            'F-agent',
            'agent',
            ('Agent 过程发现', 'Agent trace observations'),
            (
                '类别按样本去重；工具数据依赖和状态改变不等于越权，出现异常不证明失败根因。',
                'Categories counted once per sample; provenance and writes are not violations or proved failure causes.',
            ),
            data={
                'sample_categories': dict(
                    Counter(c for s in agent for c in {f.category.value for f in s.case.findings})
                ),
                'tool_calls': sum(s.case.tool_call_count for s in agent),
            },
        )
    else:
        coverage.skipped_facts['F-agent'] = 'No agent trace or tool calls.'
    return facts
