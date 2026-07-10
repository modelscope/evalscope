from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
from datetime import datetime

from evalscope.perf.config.models import PerfConfig
from evalscope.perf.config.resolve import resolve_suite
from evalscope.perf.domain.errors import ResultAlreadyExistsError
from evalscope.perf.domain.result import ArtifactManifest, PerfSuiteResult
from evalscope.perf.engine.run_engine import RunEngine


class SuiteRunner:

    def __init__(self, config: PerfConfig) -> None:
        self.resolved = resolve_suite(config)
        self.config = config

    async def run(self) -> PerfSuiteResult:
        run_id = self.config.output.run_id or self._default_run_id()
        suite_dir = os.path.join(self.config.output.root, run_id)
        if os.path.exists(suite_dir):
            if not self.config.output.overwrite:
                raise ResultAlreadyExistsError(f'Output {suite_dir} already exists')
            shutil.rmtree(suite_dir)
        os.makedirs(os.path.join(suite_dir, 'runs'))

        sla_result = None
        if self.config.sla is not None:
            from evalscope.perf.sla import SLATuner

            runs, sla_result = await SLATuner(self.config, run_id, suite_dir).run()
        else:
            runs = []
            for index, spec in enumerate(self.resolved.runs):
                result = await RunEngine(self.config, run_id, spec, suite_dir).run()
                runs.append(result)
                if index < len(self.resolved.runs) - 1 and self.config.suite.sleep_between_runs:
                    await asyncio.sleep(self.config.suite.sleep_between_runs)

        artifacts = ArtifactManifest(
            root=suite_dir,
            files={
                'manifest': os.path.join(suite_dir, 'manifest.json'),
                'config': os.path.join(suite_dir, 'suite_config.json'),
                'summary': os.path.join(suite_dir, 'suite_summary.json'),
                'html': os.path.join(suite_dir, 'perf_report.html') if self.config.output.html_report else None,
            },
        )
        result = PerfSuiteResult(
            run_id=run_id,
            suite_config=self.config,
            runs=runs,
            sla_result=sla_result,
            artifacts=artifacts,
        )
        self._write_files(result)
        if self.config.output.console_report:
            from evalscope.perf.reporting.console import print_suite

            print_suite(result)
        if self.config.output.html_report:
            from evalscope.perf.reporting.html import write_html_report

            write_html_report(result, artifacts.files['html'])
        return result

    def _default_run_id(self) -> str:
        model = re.sub(r'[^A-Za-z0-9_.-]+', '-', self.config.target.model).strip('-') or 'model'
        return f'{datetime.now().strftime("%Y%m%d-%H%M%S-%f")}-{model}'

    def _write_files(self, result: PerfSuiteResult) -> None:
        files = result.artifacts.files
        with open(files['config'], 'w', encoding='utf-8') as file:
            json.dump(self.config.model_dump(mode='json'), file, ensure_ascii=False, indent=2)
        with open(files['summary'], 'w', encoding='utf-8') as file:
            json.dump(result.model_dump(mode='json'), file, ensure_ascii=False, indent=2)
        manifest = {
            'schema_version': 1,
            'run_id': result.run_id,
            'files': files,
            'runs': [run.artifacts.model_dump(mode='json') for run in result.runs],
        }
        with open(files['manifest'], 'w', encoding='utf-8') as file:
            json.dump(manifest, file, ensure_ascii=False, indent=2)
