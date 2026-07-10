import asyncio
import pytest
from pydantic import ValidationError

from evalscope.perf import BenchmarkSuite, ClosedLoopLoad, OutputConfig, PerfConfig, TargetConfig, WorkloadConfig
from evalscope.perf.config.models import SLAConfig
from evalscope.perf.domain.result import ArtifactManifest, PercentileStats, RunResult, RunSummary
from evalscope.perf.sla.criteria import parse_criterion
from evalscope.perf.sla.tuner import SLATuner


def test_sla_requires_exactly_one_search_kind() -> None:
    with pytest.raises(ValidationError):
        SLAConfig(variable='concurrency')
    with pytest.raises(ValidationError):
        SLAConfig(variable='concurrency', criteria=[{'avg_latency': '<1'}], objective='max_rps')


def test_criterion_parsing() -> None:
    assert parse_criterion('<= 2.5').validate(2.5)
    assert not parse_criterion('< 2.5').validate(2.5)


def _config(tmp_path, sla):
    return PerfConfig(
        target=TargetConfig(model='fake'),
        workload=WorkloadConfig(name='prompt', prompt='hello'),
        suite=BenchmarkSuite(loads=[ClosedLoopLoad(concurrency=1, request_count=2)]),
        output=OutputConfig(root=str(tmp_path), run_id='sla'),
        sla=sla,
    )


async def _fake_run(self):
    value = self.spec.load.concurrency
    rps = 10 - (value - 4)**2
    return RunResult(
        run_id=self.run_id,
        run_spec=self.spec,
        summary=RunSummary(
            total=2,
            succeeded=2,
            success_rate=100,
            request_throughput=rps,
            averages={'latency': float(value)},
        ),
        percentiles={'latency': PercentileStats(p99=float(value))},
        artifacts=ArtifactManifest(root=self.run_dir),
    )


def test_constraint_search_finds_highest_passing_load(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr('evalscope.perf.engine.run_engine.RunEngine.run', _fake_run)
    config = _config(
        tmp_path,
        SLAConfig(
            variable='concurrency',
            criteria=[{
                'avg_latency': '<=3'
            }],
            lower_bound=1,
            upper_bound=8,
            repetitions=2,
        ),
    )
    _, result = asyncio.run(SLATuner(config, 'run', str(tmp_path)).run())
    assert result.best_value == 3


def test_objective_search_scans_peak_neighborhood(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr('evalscope.perf.engine.run_engine.RunEngine.run', _fake_run)
    config = _config(
        tmp_path,
        SLAConfig(
            variable='concurrency',
            objective='max_rps',
            lower_bound=1,
            upper_bound=8,
            repetitions=1,
        ),
    )
    _, result = asyncio.run(SLATuner(config, 'run', str(tmp_path)).run())
    assert result.best_value == 4
