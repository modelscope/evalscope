"""One offline/online diagnostic pipeline; disk inputs are the source of truth."""

import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

from filelock import FileLock

from evalscope.api.analysis import (
    AnalysisReportConfig,
    DiagnosisCoverage,
    DiagnosisReport,
    FailurePattern,
    FindingCategory,
    SampleDiagnosis,
)
from evalscope.api.analysis.schema import AnalysisFactsConfig
from evalscope.api.analysis.taxonomy import ISSUE_NAMES, TAXONOMY_VERSION, finding_type
from evalscope.constants import DEFAULT_LANGUAGE
from evalscope.report.diagnosis import (
    DiagnosisSnapshot,
    data_fingerprint,
    diagnosis_dir,
    iter_review_files,
    load_diagnosis,
    read_diagnosis_snapshot,
)
from evalscope.utils.io_utils import yaml_to_dict

from .facts import collect_facts
from .inputs import is_choice_letter, load_inputs
from .trace.export import _env_secrets, export_findings, redact, redact_data
from .trace.reader import discover_runs, has_nontext_output, iter_report_pairs

ANALYZER_VERSION = '3.4.7'
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


def _redacted_snapshot(report: DiagnosisSnapshot, secrets: List[str]) -> DiagnosisSnapshot:
    payload = report.model_dump(mode='json')
    sanitized = redact_data(payload, secrets)
    if sanitized['facts_config'] != payload['facts_config']:
        # Redaction can change a custom regex. Never reuse a modified rule.
        sanitized['facts_config'] = None
    return DiagnosisSnapshot.model_validate(sanitized)


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
        lines += [
            f'- Model hypotheses with locatable citations: {c.diagnosed_ok}/{c.selected}; '
            f'selected from {c.diagnosis_population} eligible samples. Causes remain unverified.'
        ]
    if report.narrative:
        for item in report.narrative.findings + report.narrative.recommendations:
            labels = []
            for ref in item.refs:
                sample_ref = report.narrative_samples.get(ref)
                if sample_ref:
                    label = f'{sample_ref.sample.subset} · {"样本" if zh else "Sample"} {sample_ref.sample.index}'
                    if sample_ref.step is not None:
                        label += f' · {"步骤" if zh else "Step"} {sample_ref.step}'
                    labels.append(f'[{label}]')
                else:
                    labels.append(f'[{ref}]')
            lines.append(f'- {item.text} ' + ' '.join(labels))
    for fact in report.facts:
        value = '' if fact.value is None else f' {fact.value:.1%}' if fact.unit == 'fraction' else f' {fact.value:g}'
        lines.append(f'- [{fact.id}] {fact.title}:{value} — {fact.definition}')
    if report.patterns:
        lines += [
            '',
            '| 类型 / Type | 未通过 / Failed | 无法评分 / Unscored | 代表样本 / Examples |',
            '|---|---:|---:|---|',
        ]
        for pattern in report.patterns[:10]:
            name = ISSUE_NAMES.get(pattern.failure_type, ISSUE_NAMES['other'])[1 if zh else 0]
            examples = ', '.join(f'{s.subset}:{s.index}' for s in pattern.examples)
            lines.append(f'| {name} | {pattern.failed} | {pattern.unscored} | {examples} |')
    lines += ['', f'- Analysis calls: {report.cost.calls}; skipped facts: {len(c.skipped_facts)}.', '']
    return '\n'.join(lines)


class MissingFactsSettingsError(ValueError):
    """Legacy analysis needs regeneration with its original settings before it can be inherited."""


def _saved_facts_settings(run_dir: str, model: str, benchmark: str, base: Path) -> AnalysisFactsConfig:
    _, previous = read_diagnosis_snapshot(run_dir, model, benchmark, base)
    if previous is None:
        _, previous = read_diagnosis_snapshot(run_dir, model, benchmark, base / 'diagnose')
    if previous is not None and previous.facts_config is not None:
        return previous.facts_config

    # Old artifacts saved a fingerprint but not the settings. Recover only settings we can verify.
    task_path = Path(run_dir) / 'configs' / 'task_config.yaml'
    task = yaml_to_dict(str(task_path)) if task_path.is_file() else None
    configured = task.get('analysis_report') if isinstance(task, dict) else None
    recorded = None
    if isinstance(configured, dict):
        try:
            recorded = AnalysisReportConfig.model_validate(configured)
        except ValueError:
            pass
    if previous is None:
        return recorded.facts_settings() if recorded is not None else AnalysisFactsConfig()

    candidates = [AnalysisReportConfig(language=language) for language in ('auto', 'zh', 'en')]
    if recorded is not None:
        candidates.append(recorded)
    for candidate in candidates:
        if candidate.level == previous.level and config_fingerprint(candidate) == previous.config_fingerprint:
            return candidate.facts_settings()
    raise MissingFactsSettingsError(
        'Regenerate the original analysis with evalscope analyze and its original configuration.'
    )


def diagnose_benchmark(
    run_dir: str,
    model: str,
    benchmark: str,
    config: Optional[AnalysisReportConfig] = None,
    force: bool = False,
    *,
    inherit_facts: bool = False,
) -> DiagnosisReport:
    """Analyze a benchmark; inherit saved fact settings only for callers requesting it."""
    config = config or AnalysisReportConfig()
    if config.level != 'facts' and config.model is None:
        raise ValueError('diagnose requires an explicit analysis model; facts makes no model calls.')
    run_dir = str(Path(run_dir).resolve())
    base = diagnosis_dir(run_dir, model, benchmark)
    target = diagnosis_dir(run_dir, model, benchmark, config.level)
    base.mkdir(parents=True, exist_ok=True)
    with FileLock(str(base / '.lock')):
        if inherit_facts:
            settings = _saved_facts_settings(run_dir, model, benchmark, base)
            config = AnalysisReportConfig.model_validate({**config.model_dump(), **settings.model_dump()})
        model_target = diagnosis_dir(run_dir, model, benchmark, 'diagnose')
        if not (model_target / 'diagnosis.json').exists():
            _, legacy = read_diagnosis_snapshot(run_dir, model, benchmark, base)
            if legacy is not None and legacy.level == 'diagnose':
                for name in ('samples.jsonl', 'diagnosis.json'):
                    if (base / name).is_file():
                        _atomic_write(model_target / name, (base / name).read_text(encoding='utf-8'))
        try:
            report = _diagnose_benchmark(run_dir, model, benchmark, config, force, target)
            _, active = load_diagnosis(run_dir, model, benchmark)
            _write_report_analysis(run_dir, model, benchmark, active or report, config.language)
            return report
        except Exception as exc:
            status, _ = read_diagnosis_snapshot(run_dir, model, benchmark, target)
            if status == 'ready':
                raise
            failure = DiagnosisSnapshot(
                analyzer_version=ANALYZER_VERSION,
                run_dir=run_dir,
                model=model,
                benchmark=benchmark,
                level=config.level,
                generated_at=datetime.now(timezone.utc).isoformat(),
                config_fingerprint=config_fingerprint(config),
                data_fingerprint='',
                facts_config=config.facts_settings(),
                coverage=DiagnosisCoverage(
                    analysis_errors={type(exc).__name__: 1},
                    skipped_facts={'analysis': redact(str(exc), _env_secrets())},
                ),
            )
            try:
                _atomic_write(
                    target / 'diagnosis.json',
                    _redacted_snapshot(failure, _env_secrets()).model_dump_json(indent=2) + '\n',
                )
            except OSError as persist_error:
                from evalscope.utils.logger import get_logger

                get_logger().warning(f'Cannot persist analysis failure: {type(persist_error).__name__}')
            raise


def _review_issues(samples: list[SampleDiagnosis], max_examples: int) -> list[FailurePattern]:
    issues = {}
    for sample in sorted(samples, key=lambda row: (row.key.subset, row.key.index)):
        failed = sample.task_success is False
        if not failed and not sample.is_unscored:
            continue
        for kind in sample.issue_types:
            issue = issues.setdefault(kind, FailurePattern(failure_type=kind))
            issue.failed += failed
            issue.unscored += sample.is_unscored
            if len(issue.examples) < max_examples:
                issue.examples.append(sample.key)
            if failed:
                for tool in {
                    str(f.data['tool']) for f in sample.findings if f.data.get('tool') and finding_type(f) == kind
                }:
                    issue.tools[tool] = issue.tools.get(tool, 0) + 1
    return sorted(issues.values(), key=lambda issue: (-issue.failed, -issue.unscored, issue.failure_type))


def _diagnose_benchmark(
    run_dir: str, model: str, benchmark: str, config: AnalysisReportConfig, force: bool, target: Path
) -> DiagnosisReport:
    fingerprint = data_fingerprint(run_dir, model, benchmark)
    config_hash = config_fingerprint(config)
    path = target / 'diagnosis.json'
    samples_path = target / 'samples.jsonl'
    if not force and path.is_file() and samples_path.is_file():
        try:
            cached = DiagnosisSnapshot.model_validate_json(path.read_text())
        except (ValueError, OSError):
            cached = None
        if (
            cached is not None
            and cached.facts_config is not None
            and not cached.coverage.analysis_errors
            and not any(
                cached.coverage.diagnosis_statuses.get(k)
                for k in ('parse_error', 'transport_error', 'invalid_evidence')
            )
            and cached.narrative_status not in ('parse_error', 'transport_error', 'invalid_references')
            and cached.schema_version == DiagnosisReport.model_fields['schema_version'].default
            and cached.taxonomy_version == TAXONOMY_VERSION
            and cached.config_fingerprint == config_hash
            and cached.data_fingerprint == fingerprint
            and (cached.samples_sha256 == hashlib.sha256(samples_path.read_bytes()).hexdigest())
        ):
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
    choice_task = inputs.is_multiple_choice
    for item in inputs.samples:
        case = item.case
        relevant = [
            f
            for f in case.findings
            if f.category not in _INFORMATIONAL
            and not (f.category == FindingCategory.REFUSAL and case.labels.get('task_kind') == 'harmful')
        ]
        observed = {finding_type(finding) for finding in relevant}
        failed = case.outcome.task_success is False
        additional_issue = (
            'safety_violation' if case.outcome.safety_violation is True and not observed and not failed else None
        )
        if (
            not relevant
            and not case.tool_call_count
            and not item.record.agent_trace
            and not has_nontext_output(item.record)
        ):
            if item.stop_reason in ('length', 'max_tokens', 'model_length'):
                additional_issue = 'truncated'
            elif any(m.role == 'assistant' for m in item.record.messages):
                visible = any(m.role == 'assistant' and (m.text or '').strip() for m in item.record.messages)
                if not visible:
                    additional_issue = 'empty_output'
                else:
                    prediction = item.record.sample_score.score.extracted_prediction
                    if prediction is not None and (
                        not prediction.strip() or (choice_task and not is_choice_letter(prediction))
                    ):
                        additional_issue = 'extraction_failure'
        recorded = item.recorded_error
        if recorded:
            additional_issue = recorded[0]
        if additional_issue:
            observed.add(additional_issue)
        coverage.task_success_known += case.outcome.task_success is not None
        coverage.task_failed += failed
        coverage.outcome_unknown += case.outcome.task_success is None
        coverage.excluded += case.score_status not in ('success', 'fallback', 'degraded')
        coverage.unexplained_failures += failed and not observed
        samples.append(
            SampleDiagnosis(
                key=item.key,
                task_success=case.outcome.task_success,
                safety_violation=case.outcome.safety_violation,
                score_status=case.score_status,
                stop_reason=item.stop_reason,
                needs_attention=failed or bool(observed),
                observed_types=sorted(observed),
                recorded_error=recorded[1] if recorded else None,
                findings=export_findings(case.findings, config.trace),
            )
        )
    provider_notes = Counter(note for item in inputs.samples for note in item.case.notes)
    if provider_notes:
        coverage.skipped_facts['trace-provider'] = '; '.join(
            f'{note}: {count}' for note, count in provider_notes.items()
        )
    facts = collect_facts(inputs, config, coverage)
    report = DiagnosisSnapshot(
        analyzer_version=ANALYZER_VERSION,
        run_dir=run_dir,
        model=model,
        benchmark=benchmark,
        generated_at=datetime.now(timezone.utc).isoformat(),
        config_fingerprint=config_hash,
        data_fingerprint=fingerprint,
        facts_config=config.facts_settings(),
        coverage=coverage,
        level=config.level,
        facts=facts,
        limitations=[
            'Rule findings are observations, not proved causes. Outcome unknown is not a pass.',
            'Observed issue groups may overlap; their counts must not be added.',
            'Provenance and state changes alone do not establish policy violations.',
        ],
    )
    if config.level != 'facts':
        from .llm import enrich, summarize

        enrich(report, samples, inputs, config, target.parent, force, _atomic_write)
    report.patterns = _review_issues(samples, config.max_examples)
    coverage.risk_samples = sum(sample.has_risk for sample in samples)
    coverage.proposals_without_rule_clues = sum(
        sample.task_success is False and sample.issue_types == ['unexplained'] and sample.llm_status == 'ok'
        for sample in samples
    )
    if config.level == 'diagnose':
        summarize(report, samples, config, target.parent, force, _atomic_write)
        report.limitations += [
            'Valid citations establish provenance, not semantic support or root-cause accuracy.',
            'Model hypotheses describe only the selected samples; no population cause estimates are made.',
        ]
    report = _redacted_snapshot(report, secrets)
    if data_fingerprint(run_dir, model, benchmark) != fingerprint:
        raise RuntimeError('Analysis inputs changed during computation; retry on a stable run.')
    sample_text = ''.join(
        json.dumps(redact_data(s.model_dump(mode='json'), secrets), ensure_ascii=False) + '\n'
        for s in samples
        if s.task_success is False
        or s.is_unscored
        or s.has_risk
        or s.findings
        or s.needs_attention
        or s.llm_status != 'not_selected'
    )
    report.samples_sha256 = hashlib.sha256(sample_text.encode()).hexdigest()
    _atomic_write(samples_path, sample_text)
    _atomic_write(path, json.dumps(report.model_dump(mode='json'), ensure_ascii=False, indent=2) + '\n')
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
