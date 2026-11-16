"""One offline/online diagnostic pipeline; disk inputs are the source of truth."""

import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

from filelock import FileLock

from evalscope.api.analysis import (
    AnalysisReportConfig,
    Attribution,
    DiagnosisCoverage,
    DiagnosisReport,
    FailurePattern,
    FindingCategory,
    SampleDiagnosis,
)
from evalscope.api.analysis.taxonomy import ISSUE_NAMES, TAXONOMY_VERSION, finding_type
from evalscope.constants import DEFAULT_LANGUAGE
from evalscope.report.diagnosis import data_fingerprint, diagnosis_dir, iter_review_files

from .facts import collect_facts
from .inputs import is_choice_letter, load_inputs
from .trace.attribution import primary_attribution
from .trace.export import _env_secrets, case_to_dict, redact
from .trace.reader import discover_runs, iter_report_pairs

ANALYZER_VERSION = '2.2.0'
_INFORMATIONAL = {FindingCategory.SIDE_EFFECT, FindingCategory.TAINTED_SIDE_EFFECT}


def config_fingerprint(config: AnalysisReportConfig) -> str:
    """Include analysis/taxonomy versions and effective language in cache identity."""
    payload = (
        json.dumps(config.model_dump(mode='json'), sort_keys=True)
        + ANALYZER_VERSION
        + TAXONOMY_VERSION
        + DEFAULT_LANGUAGE
    )
    if config.level != 'facts':
        from .llm import PROMPT, PROTOCOL_VERSION

        payload += PROMPT.read_text() + PROTOCOL_VERSION
    return hashlib.sha256(payload.encode()).hexdigest()


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.name}.')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _redact(value: object, secrets: List[str]) -> object:
    if isinstance(value, str):
        return redact(value, secrets)
    if isinstance(value, dict):
        return {
            redact(key, secrets) if isinstance(key, str) else key: _redact(item, secrets) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item, secrets) for item in value]
    return value


def render_markdown(report: DiagnosisReport, language: str = 'auto') -> str:
    """Render facts without implying LLM use or proved causation."""
    zh = language == 'zh' or (language == 'auto' and DEFAULT_LANGUAGE == 'zh')
    c = report.coverage
    lines = [
        ('## 自动诊断（规则）' if zh else '## Automatic analysis (rules)')
        if report.level == 'facts'
        else ('## 自动诊断（模型分析）' if zh else '## Automatic analysis (model proposals)'),
        '',
        (
            '事实与过程发现，不代表已证明的失败根因。'
            if zh
            else 'Observed facts and trace findings, not proved root causes.'
        ),
        '',
        (
            f'- 样本 {c.samples}；任务未通过 {c.task_failed}/{c.task_success_known}；成败未知 {c.outcome_unknown}。'
            if zh
            else f'- Samples {c.samples}; task failures {c.task_failed}/{c.task_success_known}; unknown outcomes {c.outcome_unknown}.'
        ),
        (
            f'- 未通过样本中暂无规则线索：{c.unexplained_failures}/{c.task_failed}；排除评分 {c.excluded}。'
            if zh
            else f'- Failed samples without rule clues: {c.unexplained_failures}/{c.task_failed}; excluded scores {c.excluded}.'
        ),
        '',
    ]
    if report.level != 'facts':
        estimate = c.failed_unexplained_estimate
        lines += [
            f'- Model proposals with valid citations: {c.diagnosed_ok}/{c.selected}; '
            f'review population: {c.diagnosis_population}. Citation existence is not semantic verification.'
        ]
        if estimate:
            lines += [
                f'- Failed samples without a grounded model proposal (weighted): {estimate.proportion:.1%}; '
                f'95% stratified interval [{estimate.lower:.1%}, {estimate.upper:.1%}].'
            ]
    if report.narrative:
        for item in report.narrative.findings + report.narrative.recommendations:
            lines.append(f'- {item.text} ' + ' '.join(f'[{ref}]' for ref in item.refs))
    for fact in report.facts:
        value = '' if fact.value is None else f' {fact.value:.1%}' if fact.unit == 'fraction' else f' {fact.value:g}'
        lines.append(f'- [{fact.id}] {fact.title}:{value} — {fact.definition}')
    if report.patterns:
        lines += ['', '| ID | 类型 / Type | 数量 / Count | 代表样本 / Examples |', '|---|---|---:|---|']
        for pattern in report.patterns[:10]:
            name = ISSUE_NAMES.get(pattern.failure_type, ISSUE_NAMES['other'])[1 if zh else 0]
            examples = ', '.join(f'{s.subset}:{s.index}' for s in pattern.examples)
            lines.append(f'| {pattern.id} | {name} | {pattern.count} | {examples} |')
    lines += ['', f'- Analysis calls: {report.cost.calls}; skipped facts: {len(c.skipped_facts)}.', '']
    return '\n'.join(lines)


def diagnose_benchmark(
    run_dir: str, model: str, benchmark: str, config: Optional[AnalysisReportConfig] = None, force: bool = False
) -> DiagnosisReport:
    """Analyze a benchmark and persist a readable failure state without changing scores."""
    config = config or AnalysisReportConfig()
    if config.level != 'facts' and config.model is None:
        raise ValueError('diagnose/full requires an explicit analysis model; facts makes no model calls.')
    started = datetime.now(timezone.utc).isoformat()
    try:
        return _diagnose_benchmark(run_dir, model, benchmark, config, force)
    except Exception as exc:
        target = diagnosis_dir(str(Path(run_dir).resolve()), model, benchmark)
        try:
            target.mkdir(parents=True, exist_ok=True)
            with FileLock(str(target / '.lock')):
                path = target / 'diagnosis.json'
                current = None
                if path.is_file():
                    try:
                        current = DiagnosisReport.model_validate_json(path.read_text())
                    except (ValueError, OSError):
                        pass
                if current is None or current.generated_at < started:
                    failure = DiagnosisReport(
                        analyzer_version=ANALYZER_VERSION,
                        run_dir=str(Path(run_dir).resolve()),
                        model=model,
                        benchmark=benchmark,
                        generated_at=datetime.now(timezone.utc).isoformat(),
                        config_fingerprint=config_fingerprint(config),
                        data_fingerprint='',
                        coverage=DiagnosisCoverage(
                            analysis_errors={type(exc).__name__: 1},
                            skipped_facts={'analysis': redact(str(exc), _env_secrets())},
                        ),
                    )
                    _atomic_write(
                        path,
                        json.dumps(
                            _redact(failure.model_dump(mode='json'), _env_secrets()), ensure_ascii=False, indent=2
                        )
                        + '\n',
                    )
        except Exception as persist_error:
            from evalscope.utils.logger import get_logger

            get_logger().warning(f'Cannot persist analysis failure: {type(persist_error).__name__}')
        raise


def _diagnose_benchmark(
    run_dir: str, model: str, benchmark: str, config: AnalysisReportConfig, force: bool
) -> DiagnosisReport:
    run_dir = str(Path(run_dir).resolve())
    target = diagnosis_dir(run_dir, model, benchmark)
    target.mkdir(parents=True, exist_ok=True)
    with FileLock(str(target / '.lock')):
        fingerprint = data_fingerprint(run_dir, model, benchmark)
        config_hash = config_fingerprint(config)
        path = target / 'diagnosis.json'
        samples_path = target / 'samples.jsonl'
        if not force and path.is_file() and samples_path.is_file():
            try:
                cached = DiagnosisReport.model_validate_json(path.read_text())
            except (ValueError, OSError):
                cached = None
            if (
                cached is not None
                and not cached.coverage.analysis_errors
                and not any(cached.coverage.diagnosis_statuses.get(k) for k in ('parse_error', 'transport_error'))
                and cached.narrative_status not in ('parse_error', 'transport_error')
                and cached.taxonomy_version == TAXONOMY_VERSION
                and cached.config_fingerprint == config_hash
                and cached.data_fingerprint == fingerprint
                and (cached.samples_sha256 == hashlib.sha256(samples_path.read_bytes()).hexdigest())
            ):
                _write_report_analysis(run_dir, model, benchmark, cached, config.language)
                return cached
        inputs = load_inputs(run_dir, model, benchmark, config.trace)
        coverage = DiagnosisCoverage(samples=len(inputs.samples), unreadable=inputs.unreadable)
        if inputs.unreadable_predictions:
            coverage.skipped_facts['predictions'] = (
                f'{inputs.unreadable_predictions} unreadable or ambiguous prediction rows.'
            )
        if inputs.report and inputs.report.execution_summary:
            coverage.errored_without_record = inputs.report.execution_summary.errored
        secrets = _env_secrets()
        samples = []
        decisive = {}
        choice_task = inputs.is_multiple_choice
        for item in inputs.samples:
            case = item.case
            relevant = [
                f
                for f in case.findings
                if f.category not in _INFORMATIONAL
                and f.category != FindingCategory.OUTCOME
                and not (f.category == FindingCategory.REFUSAL and case.labels.get('task_kind') == 'harmful')
            ]
            attribution, finding = primary_attribution(relevant)
            if finding is not None:
                decisive[(item.key.subset, item.key.index)] = finding
            issue = finding_type(finding) if finding else 'unexplained' if case.outcome.task_success is False else None
            if case.outcome.safety_violation is True and issue is None:
                issue = 'safety_violation'
            if not relevant and not case.tool_call_count and not item.record.agent_trace:
                if item.stop_reason in ('length', 'max_tokens', 'model_length'):
                    issue = 'truncated'
                elif any(m.role == 'assistant' for m in item.record.messages):
                    visible = any(m.role == 'assistant' and (m.text or '').strip() for m in item.record.messages)
                    if not visible:
                        issue = 'empty_output'
                    else:
                        prediction = item.record.sample_score.score.extracted_prediction
                        if prediction is not None and (
                            not prediction.strip() or (choice_task and not is_choice_letter(prediction))
                        ):
                            issue = 'extraction_failure'
            coverage.task_success_known += case.outcome.task_success is not None
            coverage.task_failed += case.outcome.task_success is False
            coverage.outcome_unknown += case.outcome.task_success is None
            coverage.excluded += case.score_status not in ('success', 'fallback', 'degraded')
            coverage.unexplained_failures += case.outcome.task_success is False and issue == 'unexplained'
            attention = bool(relevant) or case.failed or issue is not None
            coverage.needs_attention += attention
            exported = case_to_dict(case, config.trace, secrets)
            samples.append(
                SampleDiagnosis(
                    key=item.key,
                    task_success=case.outcome.task_success,
                    safety_violation=case.outcome.safety_violation,
                    score_status=case.score_status,
                    stop_reason=item.stop_reason,
                    score_value=case.score_value,
                    needs_attention=attention,
                    failure_type=issue,
                    rule_failure_type=issue,
                    source='none' if issue in (None, 'unexplained') else 'rule',
                    attribution=attribution or Attribution.UNKNOWN,
                    findings=exported['findings'],
                )
            )
        provider_notes = Counter(note for item in inputs.samples for note in item.case.notes)
        if provider_notes:
            coverage.skipped_facts['trace-provider'] = '; '.join(
                f'{note}: {count}' for note, count in provider_notes.items()
            )
        facts = collect_facts(inputs, config, coverage)
        report = DiagnosisReport(
            analyzer_version=ANALYZER_VERSION,
            run_dir=run_dir,
            model=model,
            benchmark=benchmark,
            generated_at=datetime.now(timezone.utc).isoformat(),
            config_fingerprint=config_hash,
            data_fingerprint=fingerprint,
            coverage=coverage,
            level=config.level,
            facts=facts,
            limitations=[
                'Rule findings are observations, not proved causes. Outcome unknown is not a pass.',
                'Pattern shares use samples needing review, not the official score denominator.',
                'Provenance and state changes alone do not establish policy violations.',
            ],
        )
        selected = []
        if config.level != 'facts':
            from .llm import enrich, estimate, summarize

            selected = enrich(report, samples, inputs, config, target, force, _atomic_write)
        grouped = defaultdict(list)
        for sample in samples:
            if sample.needs_attention and sample.failure_type:
                finding = decisive.get((sample.key.subset, sample.key.index))
                detail = finding.subtype if finding and sample.llm_status != 'ok' else ''
                if finding and sample.llm_status != 'ok' and finding.data.get('tool'):
                    detail += f' @ {finding.data["tool"]}'
                grouped[(sample.failure_type, sample.attribution, detail)].append(sample)

        def example_order(sample: SampleDiagnosis) -> tuple:
            finding = decisive.get((sample.key.subset, sample.key.index))
            rank = {'high': 0, 'medium': 1, 'low': 2}.get(finding.confidence, 3) if finding else 3
            return rank, sample.key.subset, sample.key.index

        patterns = []
        for (issue, attribution, detail), members in sorted(grouped.items(), key=lambda pair: (-len(pair[1]), pair[0])):
            identity = 'P-' + hashlib.sha256(f'{issue}/{attribution}/{detail}'.encode()).hexdigest()[:8]
            for sample in members:
                sample.pattern_id = identity
            sources = {s.source for s in members}
            source = (
                'rule+llm'
                if 'rule+llm' in sources or {'rule', 'llm'} <= sources
                else 'llm'
                if 'llm' in sources
                else 'none'
                if issue == 'unexplained'
                else 'rule'
            )
            patterns.append(
                FailurePattern(
                    id=identity,
                    failure_type=issue,
                    attribution=attribution,
                    detail=detail,
                    count=len(members),
                    denominator=coverage.needs_attention,
                    share=len(members) / coverage.needs_attention,
                    by_subset=dict(Counter(s.key.subset for s in members)),
                    examples=[s.key for s in sorted(members, key=example_order)[: min(config.max_examples, 5)]],
                    source=source,
                )
            )
        report.patterns = patterns
        if config.level != 'facts':
            for pattern in report.patterns:
                if pattern.source in ('llm', 'rule+llm'):
                    pattern.estimated_share = estimate(
                        coverage.strata, selected, lambda s: s.llm_status == 'ok' and s.pattern_id == pattern.id
                    )
            if config.level == 'full':
                summarize(report, config, target, force, _atomic_write)
            report.limitations += [
                'Valid citations establish provenance, not semantic support or root-cause accuracy.',
                'Weighted proportions concern model proposals in the frozen review population; failures retain their own denominator.',
                'Unselected, failed and low-confidence diagnoses remain unresolved for LLM coverage.',
            ]
        report = DiagnosisReport.model_validate(_redact(report.model_dump(mode='json'), secrets))
        if data_fingerprint(run_dir, model, benchmark) != fingerprint:
            raise RuntimeError('Analysis inputs changed during computation; retry on a stable run.')
        sample_text = ''.join(
            json.dumps(_redact(s.model_dump(mode='json'), secrets), ensure_ascii=False) + '\n'
            for s in samples
            if s.task_success is False or s.findings or s.needs_attention
        )
        report.samples_sha256 = hashlib.sha256(sample_text.encode()).hexdigest()
        _atomic_write(samples_path, sample_text)
        _atomic_write(
            path, json.dumps(_redact(report.model_dump(mode='json'), secrets), ensure_ascii=False, indent=2) + '\n'
        )
        _write_report_analysis(run_dir, model, benchmark, report, config.language)
        return report


def _write_report_analysis(run_dir: str, model: str, benchmark: str, report: DiagnosisReport, language: str) -> None:
    path = Path(run_dir) / 'reports' / model / f'{benchmark}.json'
    if path.is_file():
        data = json.loads(path.read_text())
        data['analysis'] = render_markdown(report, language)
        _atomic_write(path, json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def diagnose_run(
    run_dir: str,
    config: Optional[AnalysisReportConfig] = None,
    models: Optional[Sequence[str]] = None,
    benchmarks: Optional[Sequence[str]] = None,
    force: bool = False,
) -> List[DiagnosisReport]:
    """Run the same pipeline on every discovered run; filters refer to directory model IDs."""
    reports = []
    for run in discover_runs(run_dir):
        pairs = {(f.model, f.benchmark) for f in iter_review_files(run)}
        pairs.update(iter_report_pairs(run))
        analyzed = False
        for model, benchmark in sorted(pairs):
            if (models and model not in models) or (benchmarks and benchmark not in benchmarks):
                continue
            reports.append(diagnose_benchmark(run, model, benchmark, config, force))
            analyzed = True
        if analyzed and (Path(run) / 'reports').is_dir():
            from evalscope.report.renderer import gen_html_report_file

            gen_html_report_file(str(Path(run) / 'reports'))
    return reports
