"""Command line entry point of the metric audit.

Run as ``python -m evalscope.metrics.semantics.audit`` (or ``make metric-audit``). The whole run
is read-only: nothing is written, no adapter is instantiated and no report file is modified. A
default run never depends on the workspace ``outputs/`` directory; observing historical reports
is an explicit ``--observed-path`` opt-in.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Mapping, Optional, Sequence, Tuple, Union

from evalscope.api.metric.semantics import MetricEntry
from .checks import AuditReport, run_checks
from .collectors import AUDIT_LOG_PREFIX, GROUP_DISPLAY_ORDER, AdapterScan, collect_metric_inventory

if TYPE_CHECKING:
    from evalscope.api.benchmark.meta import BenchmarkMeta


def run_audit(
    benchmarks: Optional[Iterable[str]] = None,
    observed_paths: Iterable[Union[str, Path]] = (),
    registry: Optional[Mapping[str, 'BenchmarkMeta']] = None,
    scans: Optional[Mapping[Tuple[str, str], AdapterScan]] = None,
    name_table: Optional[Mapping[str, MetricEntry]] = None,
    overrides: Optional[Mapping[Tuple[str, str], MetricEntry]] = None,
    dynamic: Optional[Mapping[str, Sequence[str]]] = None,
    perf_fields: Optional[Mapping[str, MetricEntry]] = None,
) -> AuditReport:
    """Collect the inventory and run every check against it.

    Args:
        benchmarks: Restrict the audit to these benchmark names. ``None`` audits all of them.
        observed_paths: Explicit report paths to observe. Empty by default.
        registry: Benchmark registry override, for tests.
        scans: Adapter source scan override, for tests.
        name_table: Metric name table override, for tests.
        overrides: Collision override table override, for tests.
        dynamic: Dynamic allow-list override, for tests.
        perf_fields: Perf field semantics override, for tests.

    Returns:
        The report: the inventory plus the findings, sorted deterministically.
    """
    inventory = collect_metric_inventory(
        benchmarks=benchmarks,
        observed_paths=observed_paths,
        registry=registry,
        scans=scans,
    )
    errors = run_checks(
        inventory,
        name_table=name_table,
        overrides=overrides,
        dynamic=dynamic,
        perf_fields=perf_fields,
    )
    return AuditReport(inventory=inventory, errors=errors)


def format_audit_report(report: AuditReport) -> str:
    """Render the report as the plain text output of the entry point.

    Args:
        report: Report from :func:`run_audit`.

    Returns:
        The full text output: the three metric buckets, the public perf field keys and the
        findings with their counts.
    """
    inventory = report.inventory
    grouped = inventory.grouped()
    lines = [
        f'{AUDIT_LOG_PREFIX} audited {len(inventory.declarations)} benchmarks, '
        f'coverage base {len(inventory.coverage_base)}, '
        f'observed paths {len(inventory.observed_paths)}'
    ]

    for group in GROUP_DISPLAY_ORDER:
        records = grouped[group]
        lines.append(f'{AUDIT_LOG_PREFIX} {group.value} ({len(records)})')
        lines.extend(f'  {record.benchmark_name}  {record.metric_name}' for record in records)

    lines.append(f'{AUDIT_LOG_PREFIX} perf_field_keys ({len(inventory.perf_field_keys)})')
    lines.extend(
        f'  {record.field_key}  ({record.holder}.{record.constant_name})' for record in inventory.perf_field_keys
    )

    if not report.errors:
        lines.append(f'{AUDIT_LOG_PREFIX} no audit errors')
        return '\n'.join(lines)

    counts = ', '.join(f'{code}={count}' for code, count in report.error_counts().items())
    lines.append(f'{AUDIT_LOG_PREFIX} errors ({len(report.errors)}): {counts}')
    for error in report.errors:
        head, *rest = error.message.splitlines()
        lines.append(f'  {error.code.value} {head}')
        lines.extend(f'  {line}' for line in rest)
    return '\n'.join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser of the entry point."""
    parser = argparse.ArgumentParser(
        prog='python -m evalscope.metrics.semantics.audit',
        description=(
            'Read-only audit of the final report metric names EvalScope can emit and of their '
            'coverage in the metric semantics catalog. Exits non-zero when an audit error is found.'
        ),
    )
    parser.add_argument('--json', action='store_true', help='print the report as JSON instead of text')
    parser.add_argument(
        '--benchmark',
        action='append',
        default=None,
        metavar='NAME',
        dest='benchmarks',
        help='restrict the audit to this benchmark name, repeatable',
    )
    parser.add_argument(
        '--observed-path',
        action='append',
        default=None,
        metavar='PATH',
        dest='observed_paths',
        help='additionally collect metric names from report files under this path, repeatable',
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the audit from the command line.

    Args:
        argv: Argument list, defaults to ``sys.argv[1:]``.

    Returns:
        ``EXIT_OK`` when no audit error was found, ``EXIT_AUDIT_ERRORS`` otherwise.
    """
    args = build_arg_parser().parse_args(argv)
    report = run_audit(benchmarks=args.benchmarks, observed_paths=args.observed_paths or ())

    if args.json:
        print(json.dumps(report.to_json_dict(), indent=2, ensure_ascii=False, sort_keys=False))
    else:
        print(format_audit_report(report))
    return report.exit_code


if __name__ == '__main__':
    sys.exit(main())

__all__ = ['build_arg_parser', 'format_audit_report', 'main', 'run_audit']
