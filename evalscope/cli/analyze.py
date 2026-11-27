"""Analyze persisted evaluation outputs without rerunning evaluation."""

from argparse import Namespace

from evalscope.cli.base import ArgumentParserWithSubParsers, CLICommand


class AnalyzeCMD(CLICommand):
    """Unified entry point for facts, trace observations and explicit model diagnostics."""

    name = 'analyze'

    def __init__(self, args: Namespace) -> None:
        self.args = args

    @staticmethod
    def define_args(parsers: ArgumentParserWithSubParsers) -> None:
        parser = parsers.add_parser('analyze', help='Analyze evaluation results; facts make no model calls.')
        parser.add_argument('--run-dir', required=True)
        parser.add_argument('--model', action='append')
        parser.add_argument('--benchmark', action='append')
        parser.add_argument(
            '--level',
            choices=['facts', 'diagnose'],
            default=None,
            help='facts: basic analysis without model calls; diagnose: sample analysis and a referenced summary with an explicit model.',
        )
        parser.add_argument('--analysis-config', default=None, help='JSON configuration or JSON file path.')
        parser.add_argument(
            '--max-samples', help='Model-analysis sample limit per benchmark; a positive integer or all.'
        )
        parser.add_argument('--force', action='store_true')
        parser.set_defaults(func=AnalyzeCMD)

    def execute(self) -> None:
        from evalscope.analysis import diagnose_run
        from evalscope.api.analysis import AnalysisReportConfig
        from evalscope.report.diagnosis import diagnosis_dir

        config = AnalysisReportConfig.from_source(self.args.analysis_config or '{}')
        if self.args.level:
            config.level = self.args.level
        if self.args.max_samples is not None:
            limit = None if self.args.max_samples.lower() == 'all' else self.args.max_samples
            if limit is not None:
                limit = int(limit)
            config = AnalysisReportConfig.model_validate({**config.model_dump(), 'max_diagnosed_samples': limit})
        reports = diagnose_run(self.args.run_dir, config, self.args.model, self.args.benchmark, self.args.force)
        if not reports:
            raise ValueError('No matching evaluation runs found; check --run-dir, --model and --benchmark.')
        for report in reports:
            c = report.coverage
            patterns = (
                ', '.join(f'{p.failure_type}={p.failed} failed/{p.unscored} unscored' for p in report.patterns[:3])
                or 'none'
            )
            failure_rate = f'{c.task_failed / c.task_success_known:.1%}' if c.task_success_known else 'unknown'
            artifact = diagnosis_dir(report.run_dir, report.model, report.benchmark, report.level) / 'diagnosis.json'
            print(
                f'{report.run_dir} / {report.model}/{report.benchmark}: '
                f'{c.samples} reviews; level={report.level}; task failures={c.task_failed}/{c.task_success_known}; '
                f'failure rate={failure_rate}; rule clues={c.task_failed - c.unexplained_failures}/{c.task_failed}; '
                f'unknown outcomes={c.outcome_unknown}; without rule clues={c.unexplained_failures}/{c.task_failed}; '
                f'issue groups (all samples; may overlap): {patterns}; calls={report.cost.calls}; '
                f'output={artifact}'
            )
