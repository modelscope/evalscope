"""Offline MultiPL-E pipelines with official rows and native Java execution.

Only the model responses and execution backend are controlled. Dataset loading,
prompting, extraction, scoring, aggregation, reports, and cache reuse are real.
The native backend is a test harness for fixed code, not a security sandbox.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import uuid
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

pytest.importorskip('ms_enclave')

from evalscope import TaskConfig, run_task
from evalscope.api.messages import ChatMessage
from evalscope.api.mixin.code_execution_sandbox_mixin import CodeExecutionBackend, CodeExecutionSandboxMixin
from evalscope.api.model import GenerateConfig, ModelOutput
from evalscope.api.tool import ToolChoice, ToolInfo
from evalscope.models.mockllm import MockLLM
from evalscope.report import Report
from tests.benchmark.multipl_e_java_test_utils import java_major_version, run_java

_FIXTURES = Path(__file__).parent / 'fixtures' / 'multipl_e_java'
_CASES = {
    'multiple_humaneval': ('humaneval-java', 'humaneval_53_add.json'),
    'multiple_mbpp': ('mbpp-java', 'mbpp_3_is_not_prime.json'),
}
_SOLUTIONS = {
    'multiple_humaneval': [
        'class Problem { public static long add(long x, long y) { return x + y; } }',
        'class Problem {\n  public static long add(long x, long y) {\n    return x + y;\n  }\n}',
        'class Problem {\n    public static long add(long x, long y) {\n        return x - y;\n    }\n}',
    ],
    'multiple_mbpp': [
        'class Problem { public static boolean isNotPrime(long n) { '
        'if (n < 2) return true; for (long i = 2; i * i <= n; i++) if (n % i == 0) return true; '
        'return false; } }',
        'class Problem {\n  public static boolean isNotPrime(long n) {\n'
        '    if (n < 2) return true;\n'
        '    for (long i = 2; i * i <= n; i++) if (n % i == 0) return true;\n'
        '    return false;\n  }\n}',
        'class Problem {\n    public static boolean isNotPrime(long n) {\n        return false;\n    }\n}',
    ],
}


@pytest.fixture(autouse=True)
def _offline_dataset_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import datasets

    java_major_version()
    cache = str(tmp_path / 'hf-cache')
    monkeypatch.setenv('HF_DATASETS_CACHE', cache)
    monkeypatch.setenv('HF_DATASETS_OFFLINE', '1')
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    monkeypatch.setattr(datasets.config, 'HF_DATASETS_CACHE', cache)


class _ScriptedJavaModel:

    def __init__(self, repeats: int) -> None:
        self.repeats = repeats
        self.calls: Counter[str] = Counter()
        self.active = 0
        self.max_parallel = 0
        self.lock = threading.Lock()
        self.barriers = {benchmark: threading.Barrier(repeats) for benchmark in _CASES}

    def generate(self, model: str, messages: list[ChatMessage]) -> ModelOutput:
        prompt = messages[-1].content
        benchmark = 'multiple_mbpp' if 'isNotPrime(long n)' in prompt else 'multiple_humaneval'
        # Verify the real adapter prompt still asks for a complete fenced class.
        assert 'Write the complete code and wrap it in markdown fenced code.' in prompt
        assert 'The code should not contain `Main` function.' in prompt
        with self.lock:
            attempt = self.calls[benchmark]
            assert attempt < self.repeats, 'Cached runs must reuse predictions instead of calling the model'
            self.calls[benchmark] += 1
            self.active += 1
            self.max_parallel = max(self.max_parallel, self.active)
        try:
            self.barriers[benchmark].wait(timeout=30)
            # Each repeat group has one algorithmic failure; the other attempts
            # are complete correct classes with different brace indentation.
            solution = _SOLUTIONS[benchmark][-1 if attempt == self.repeats - 1 else attempt]
            return ModelOutput.from_content(model=model, content=f'```java\n{solution}\n```')
        finally:
            with self.lock:
                self.active -= 1


class _NativeJavaRuns:

    def __init__(self, repeats: int) -> None:
        self.results: list[dict[str, Any]] = []
        self.barrier = threading.Barrier(repeats)
        self.lock = threading.Lock()
        self.active = 0
        self.max_parallel = 0

    def __len__(self) -> int:
        return len(self.results)

    def run(self, code: str) -> dict[str, Any]:
        with self.lock:
            self.active += 1
            self.max_parallel = max(self.max_parallel, self.active)
        try:
            self.barrier.wait(timeout=30)
            result = run_java(code)
            with self.lock:
                self.results.append(result)
            return result
        finally:
            with self.lock:
                self.active -= 1


class _NativeJavaBackend(CodeExecutionBackend):

    def __init__(self, benchmark_meta: Any, task_config: TaskConfig, executions: _NativeJavaRuns) -> None:
        super().__init__(benchmark_meta, task_config)
        self.executions = executions
        self.ready = False

    def start(self) -> None:
        self.ready = True

    def is_ready(self) -> bool:
        return self.ready

    def execute(self, code: str | list[str], timeout: int, language: str) -> dict[str, Any]:
        assert self.ready
        assert language == 'java'
        assert timeout == 30
        assert isinstance(code, str)
        return self.executions.run(code)

    def stop(self) -> None:
        self.ready = False


@contextmanager
def _offline_runtime(repeats: int) -> Iterator[tuple[_ScriptedJavaModel, _NativeJavaRuns]]:
    scripted = _ScriptedJavaModel(repeats)
    executions = _NativeJavaRuns(repeats)

    def get_backend(adapter: CodeExecutionSandboxMixin) -> CodeExecutionBackend:
        if adapter._backend is None:
            adapter._backend = _NativeJavaBackend(adapter._benchmark_meta, adapter._task_config, executions)
        return adapter._backend

    def generate(
        model: MockLLM, input: list[ChatMessage], tools: list[ToolInfo],
        tool_choice: ToolChoice, config: GenerateConfig,
    ) -> ModelOutput:
        return scripted.generate(model.model_name, input)

    with patch.object(CodeExecutionSandboxMixin, '_get_backend', get_backend), patch.object(MockLLM, 'generate', generate):
        yield scripted, executions


def _local_dataset(tmp_path: Path, benchmark: str) -> Path:
    subset, fixture = _CASES[benchmark]
    data = tmp_path / benchmark
    data.mkdir()
    row = json.loads((_FIXTURES / fixture).read_text())
    (data / 'test.jsonl').write_text(json.dumps(row) + '\n')
    # This is a normal local Hugging Face mirror consumed by the standard loader.
    header = {'configs': [{'config_name': subset, 'data_files': [{'split': 'test', 'path': 'test.jsonl'}]}]}
    (data / 'README.md').write_text('---\n' + yaml.safe_dump(header) + '---\n')
    return data


def _task_config(tmp_path: Path, repeats: int) -> TaskConfig:
    return TaskConfig(
        model=f'java-pipeline-{uuid.uuid4().hex}',
        eval_type='mock_llm',
        datasets=list(_CASES),
        dataset_args={
            benchmark: {'local_path': str(_local_dataset(tmp_path, benchmark)), 'subset_list': [subset]}
            for benchmark, (subset, _) in _CASES.items()
        },
        sandbox={'enabled': True},
        limit=1,
        repeats=repeats,
        eval_batch_size=4,
        generation_config={'temperature': 0},
        analysis_report=False,
        enable_progress_tracker=False,
        work_dir=str(tmp_path / 'outputs'),
        no_timestamp=True,
    )


def _cache_path(output: Path, kind: str, benchmark: str) -> Path:
    subset, _ = _CASES[benchmark]
    files = list((output / kind).rglob(f'{benchmark}_{subset}.jsonl'))
    assert len(files) == 1
    return files[0]


def _cache_rows(output: Path, kind: str, benchmark: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in _cache_path(output, kind, benchmark).read_text().splitlines() if line.strip()]


def _assert_report(report: Report, repeats: int) -> None:
    expected = {('mean', None): (repeats - 1) / repeats, ('pass_at_k', 1): (repeats - 1) / repeats}
    expected.update({('pass_at_k', k): 1.0 for k in range(2, repeats + 1)})
    actual = {
        (metric.identity.aggregation, metric.identity.dimensions.get('k')): metric.score for metric in report.metrics
    }
    # Reports publish scores rounded to four decimal places.
    assert actual == pytest.approx(expected, abs=0.00005)
    for metric in report.metrics:
        assert metric.num == (repeats if metric.identity.aggregation == 'mean' else 1)


def _assert_artifacts(output: Path, model: str, repeats: int) -> None:
    snapshot = yaml.safe_load((output / 'configs' / 'task_config.yaml').read_text())
    for benchmark, (_, fixture) in _CASES.items():
        predictions = _cache_rows(output, 'predictions', benchmark)
        reviews = _cache_rows(output, 'reviews', benchmark)
        assert len(predictions) == len(reviews) == repeats
        task_id = json.loads((_FIXTURES / fixture).read_text())['name']
        assert {row['metadata']['task_id'] for row in predictions} == {task_id}
        scores = [row['sample_score']['score'] for row in reviews]
        assert Counter(bool(score['value']['acc']) for score in scores) == {True: repeats - 1, False: 1}
        for score in scores:
            assert score['metadata']['run_language'] == 'java'
            execution = score['metadata']['execution_result']
            assert execution['stage'] == 'run', execution
            assert (execution['status'] == 'success') == bool(score['value']['acc'])
            if not score['value']['acc']:
                assert 'AssertionError' in execution['stderr']
        reports = list((output / 'reports').rglob(f'{benchmark}.json'))
        assert len(reports) == 1
        report = Report.model_validate_json(reports[0].read_text())
        assert report.model_name == model
        _assert_report(report, repeats)
        assert snapshot['evaluation_identity']['benchmarks'][benchmark]['evaluation_version'] == 'v1.1'


@pytest.mark.parametrize('repeats', [2, 3])
def test_java_official_pipelines_and_versioned_rescore(tmp_path: Path, repeats: int) -> None:
    """Complete classes must score correctly through parallel runs and saved predictions."""
    with _offline_runtime(repeats) as (scripted, executions):
        config = _task_config(tmp_path, repeats)
        reports = run_task(config)
        for report in reports.values():
            _assert_report(report, repeats)
        assert scripted.calls == {benchmark: repeats for benchmark in _CASES}
        assert scripted.max_parallel >= repeats
        assert executions.max_parallel >= repeats
        assert len(executions) == repeats * len(_CASES)
        output = Path(config.work_dir)
        _assert_artifacts(output, config.model, repeats)
        predictions = {benchmark: _cache_path(output, 'predictions', benchmark).read_bytes() for benchmark in _CASES}

        cached = config.model_copy(deep=True, update={'use_cache': str(output)})
        for report in run_task(cached).values():
            _assert_report(report, repeats)
        assert len(executions) == repeats * len(_CASES)

        # Model an output saved before the assembly scoring change. Ordinary
        # resume must reject its old identity, while explicit review can reuse
        # its byte-identical raw predictions under the new semantics.
        snapshot_path = output / 'configs' / 'task_config.yaml'
        snapshot = yaml.safe_load(snapshot_path.read_text())
        for benchmark in _CASES:
            snapshot['evaluation_identity']['benchmarks'][benchmark].update(
                evaluation_version='v1.0', fingerprint='sha256:' + '0' * 64,
            )
        snapshot_path.write_text(yaml.safe_dump(snapshot, sort_keys=False))
        with pytest.raises(ValueError, match='rerun_review=True'):
            run_task(cached.model_copy(deep=True))
        assert len(executions) == repeats * len(_CASES)

        rerun = config.model_copy(deep=True, update={'use_cache': str(output), 'rerun_review': True})
        for report in run_task(rerun).values():
            _assert_report(report, repeats)
        assert len(executions) == repeats * len(_CASES) * 2
        assert scripted.calls == {benchmark: repeats for benchmark in _CASES}
        _assert_artifacts(output, config.model, repeats)
        snapshot = yaml.safe_load(snapshot_path.read_text())
        for benchmark in _CASES:
            assert _cache_path(output, 'predictions', benchmark).read_bytes() == predictions[benchmark]
            source = snapshot['evaluation_identity']['benchmarks'][benchmark]['cache_source']
            assert source['evaluation_version'] == 'v1.0'
            assert source['prediction_reused'] is True


def test_java_cli_official_pipeline_in_fresh_interpreter(tmp_path: Path) -> None:
    """The real CLI parser and dispatcher feed the same official Java pipeline."""
    repeats = 3
    config = _task_config(tmp_path, repeats)
    bootstrap = (
        'import runpy; from tests.benchmark.test_multipl_e_java_pipeline import _offline_runtime; '
        'runtime = _offline_runtime(3); runtime.__enter__(); '
        "runpy.run_module('evalscope.cli.cli', run_name='__main__'); runtime.__exit__(None, None, None)"
    )
    command = [
        sys.executable, '-c', bootstrap, 'eval', '--model', config.model, '--eval-type', 'mock_llm',
        '--datasets', *_CASES, '--dataset-args', json.dumps(config.dataset_args),
        '--sandbox', '{"enabled": true}', '--limit', '1', '--repeats', str(repeats),
        '--eval-batch-size', '4', '--generation-config', 'temperature=0',
        '--work-dir', config.work_dir, '--no-timestamp',
    ]
    environment = dict(os.environ, HF_DATASETS_OFFLINE='1', HF_HUB_OFFLINE='1')
    completed = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False, env=environment)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    _assert_artifacts(Path(config.work_dir), config.model, repeats)
