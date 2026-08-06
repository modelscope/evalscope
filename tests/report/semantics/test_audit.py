"""Tests for the metric audit.

* ``TestAuditIsReadOnly`` -- Property 25: the audit never modifies a scanned file.
* ``TestGroupingAndExitCode`` -- Property 26: the three buckets are mutually exclusive and the
  exit code is non-zero exactly when there is a finding.
* ``TestPrimaryMetricChecks`` -- the primary metric worklist and the stale declaration check.
* ``TestUndeclaredMetrics`` -- undeclared names name the metric and the entry location.
"""

import hashlib
import pytest
from pathlib import Path
from typing import Dict, List, Tuple

from evalscope.api.metric.semantics import MetricEntry
from evalscope.metrics.semantics.audit import checks as checks_module
from evalscope.metrics.semantics.audit.checks import (
    EXIT_AUDIT_ERRORS,
    EXIT_OK,
    AuditErrorCode,
    AuditReport,
    audit_primary_metric_counts,
    audit_stale_primary_metric,
    audit_undeclared_metrics,
    run_checks,
)
from evalscope.metrics.semantics.audit.cli import build_arg_parser, format_audit_report, run_audit
from evalscope.metrics.semantics.audit.collectors import (
    BenchmarkDeclaration,
    MetricGroup,
    MetricInventory,
    MetricRecord,
    adapter_source_files,
)

#: A name table covering the metrics the synthetic inventories below emit.
NAME_TABLE: Dict[str, MetricEntry] = {
    'f1_score': MetricEntry(baseline='quality.f1.ratio'),
    'precision': MetricEntry(baseline='quality.precision.ratio'),
    'recall': MetricEntry(baseline='quality.recall.ratio'),
    'mean_acc': MetricEntry(baseline='quality.accuracy.ratio'),
    'no_answer_num': MetricEntry(baseline='diagnostic.count.items'),
}


def _inventory(
    benchmark_name: str,
    metric_names: List[str],
    primary_metric: str = None,
    group: MetricGroup = MetricGroup.DEFAULT_AGGREGATION,
) -> MetricInventory:
    """Build a synthetic inventory for one benchmark."""
    records = [
        MetricRecord(benchmark_name=benchmark_name, metric_name=name, group=group, sources=['synthetic'])
        for name in metric_names
    ]
    declaration = BenchmarkDeclaration(
        benchmark_name=benchmark_name,
        declared_metric_names=metric_names,
        primary_metric=primary_metric,
    )
    bucket = {
        MetricGroup.DEFAULT_AGGREGATION: 'default_aggregation',
        MetricGroup.CUSTOM_AGGREGATION: 'custom_aggregation',
        MetricGroup.DYNAMIC: 'dynamic',
    }[group]
    return MetricInventory(declarations={benchmark_name: declaration}, **{bucket: records})


class TestAuditIsReadOnly:
    """Feature: metric-semantics-governance, Property 25: running the audit leaves the content
    hash of every scanned source file unchanged."""

    def test_scanned_sources_are_not_modified(self) -> None:
        paths = adapter_source_files()[:40]
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}

        run_audit(benchmarks=['gsm8k'])

        after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        assert before == after

    def test_default_run_does_not_read_workspace_outputs(self) -> None:
        report = run_audit(benchmarks=['gsm8k'])

        assert report.inventory.observed_paths == []

    def test_observed_paths_are_opt_in(self, tmp_path: Path) -> None:
        parser = build_arg_parser()

        args = parser.parse_args([])
        assert args.observed_paths is None

        args = parser.parse_args(['--observed-path', str(tmp_path)])
        assert args.observed_paths == [str(tmp_path)]


class TestGroupingAndExitCode:
    """Feature: metric-semantics-governance, Property 26: the three buckets partition the
    collected names and the exit code is non-zero exactly when a finding exists."""

    def test_buckets_are_mutually_exclusive(self) -> None:
        inventory = run_audit(benchmarks=['gsm8k']).inventory
        grouped = inventory.grouped()

        keys = [{record.key for record in records} for records in grouped.values()]
        assert keys[0] & keys[1] == set()
        assert keys[0] & keys[2] == set()
        assert keys[1] & keys[2] == set()

    def test_buckets_union_equals_all_records(self) -> None:
        inventory = run_audit(benchmarks=['gsm8k']).inventory
        grouped = inventory.grouped()

        union = {record.key for records in grouped.values() for record in records}
        assert union == {record.key for record in inventory.records()}

    def test_exit_code_is_zero_without_errors(self) -> None:
        report = AuditReport(inventory=MetricInventory(), errors=[])

        assert report.exit_code == EXIT_OK
        assert not report.has_errors

    def test_exit_code_is_non_zero_with_errors(self) -> None:
        inventory = _inventory('bench_x', ['undeclared_metric'])

        errors = audit_undeclared_metrics(inventory, name_table=NAME_TABLE, overrides={}, dynamic={})
        report = AuditReport(inventory=inventory, errors=errors)

        assert report.has_errors
        assert report.exit_code == EXIT_AUDIT_ERRORS

    def test_report_text_lists_every_bucket(self) -> None:
        text = format_audit_report(run_audit(benchmarks=['gsm8k']))

        for group in MetricGroup:
            assert group.value in text


class TestUndeclaredMetrics:
    """An undeclared name is reported with its metric name and catalog entry location."""

    def test_declared_name_is_not_reported(self) -> None:
        inventory = _inventory('bench_x', ['mean_acc'])

        errors = audit_undeclared_metrics(inventory, name_table=NAME_TABLE, overrides={}, dynamic={})

        assert errors == []

    def test_undeclared_name_is_reported_with_location(self) -> None:
        inventory = _inventory('bench_x', ['weird_metric'])

        errors = audit_undeclared_metrics(inventory, name_table=NAME_TABLE, overrides={}, dynamic={})

        assert len(errors) == 1
        assert errors[0].code is AuditErrorCode.UNDECLARED_METRIC
        assert errors[0].metric_name == 'weird_metric'
        assert "METRIC_NAME_SEMANTICS['weird_metric']" in errors[0].message

    def test_collision_override_counts_as_declared(self) -> None:
        inventory = _inventory('bench_x', ['weird_metric'])
        overrides: Dict[Tuple[str, str], MetricEntry] = {
            ('bench_x', 'weird_metric'): MetricEntry(baseline='diagnostic.unspecified')
        }

        errors = audit_undeclared_metrics(inventory, name_table=NAME_TABLE, overrides=overrides, dynamic={})

        assert errors == []

    def test_dynamic_allow_list_counts_as_declared(self) -> None:
        inventory = _inventory('bench_x', ['pass@1'], group=MetricGroup.DYNAMIC)

        errors = audit_undeclared_metrics(
            inventory, name_table=NAME_TABLE, overrides={}, dynamic={'bench_x': ('pass@1', )}
        )

        assert errors == []


class TestPrimaryMetricChecks:
    """Feature: metric-semantics-governance, Property 7: the audit lists exactly the benchmarks
    whose resolved primary metric count is not one, together with their candidates."""

    def test_single_primary_is_accepted(self) -> None:
        inventory = _inventory('bench_x', ['mean_acc', 'no_answer_num'])

        errors = audit_primary_metric_counts(inventory, name_table=NAME_TABLE, overrides={})

        assert errors == []

    def test_multiple_primaries_without_declaration_are_reported(self) -> None:
        inventory = _inventory('bench_x', ['f1_score', 'precision', 'recall'])

        errors = audit_primary_metric_counts(inventory, name_table=NAME_TABLE, overrides={})

        assert len(errors) == 1
        assert errors[0].code is AuditErrorCode.PRIMARY_COUNT
        assert 'BenchmarkMeta.primary_metric' in errors[0].message
        for candidate in ('f1_score', 'precision', 'recall'):
            assert candidate in errors[0].message

    def test_declaration_resolves_the_finding(self) -> None:
        inventory = _inventory('bench_x', ['f1_score', 'precision', 'recall'], primary_metric='f1_score')

        errors = audit_primary_metric_counts(inventory, name_table=NAME_TABLE, overrides={})

        assert errors == []

    def test_stale_declaration_is_reported(self) -> None:
        inventory = _inventory('bench_x', ['f1_score'], primary_metric='not_emitted')

        errors = audit_stale_primary_metric(inventory)

        assert len(errors) == 1
        assert errors[0].code is AuditErrorCode.STALE_PRIMARY_METRIC
        assert errors[0].metric_name == 'not_emitted'

    def test_aggregation_prefixed_name_matches_the_declaration(self) -> None:
        inventory = _inventory('bench_x', ['mean_acc'], primary_metric='acc')

        assert audit_stale_primary_metric(inventory) == []

    def test_run_checks_sorts_findings_by_code(self) -> None:
        inventory = _inventory('bench_x', ['f1_score', 'precision', 'weird_metric'])

        errors = run_checks(inventory, name_table=NAME_TABLE, overrides={}, dynamic={}, perf_fields={})

        codes = [error.code for error in errors]
        assert codes == sorted(codes, key=lambda code: checks_module.ERROR_CODE_ORDER.index(code))


class TestRemovedErrorCodes:
    """The two error codes of earlier revisions must not come back."""

    @pytest.mark.parametrize('name', ['DANGLING_SEMANTIC_ID', 'MISSING_BENCHMARK_SET'])
    def test_removed_codes_are_absent(self, name: str) -> None:
        assert not hasattr(AuditErrorCode, name)

    def test_expected_codes_are_present(self) -> None:
        assert {code.value for code in AuditErrorCode} == {
            'E_UNDECLARED_METRIC',
            'E_PRIMARY_COUNT',
            'E_STALE_PRIMARY_METRIC',
            'E_AGGREGATION_GROUP_CONFLICT',
            'E_UNDECLARED_PERF_FIELD',
        }
