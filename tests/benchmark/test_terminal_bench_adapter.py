import pytest

from evalscope.benchmarks.terminal_bench.terminal_bench_adapter import _validate_environment_requirements


def test_terminal_bench_docker_requires_cli(monkeypatch):
    monkeypatch.setattr(
        'evalscope.benchmarks.terminal_bench.terminal_bench_adapter.shutil.which',
        lambda command: None,
    )

    with pytest.raises(RuntimeError, match='requires the Docker CLI'):
        _validate_environment_requirements('docker')


def test_terminal_bench_docker_allows_installed_cli(monkeypatch):
    monkeypatch.setattr(
        'evalscope.benchmarks.terminal_bench.terminal_bench_adapter.shutil.which',
        lambda command: '/usr/bin/docker',
    )

    _validate_environment_requirements('docker')


def test_terminal_bench_skips_docker_cli_check_for_remote_environments(monkeypatch):
    monkeypatch.setattr(
        'evalscope.benchmarks.terminal_bench.terminal_bench_adapter.shutil.which',
        lambda command: None,
    )

    _validate_environment_requirements('e2b')
