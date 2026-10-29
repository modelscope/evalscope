"""--dry-run parses and prints the TaskConfig, then returns without running inference."""

from unittest.mock import patch

from evalscope.config import TaskConfig
from evalscope.constants import EvalType
from evalscope.run import run_task


def _cfg(dry_run: bool) -> TaskConfig:
    return TaskConfig(
        model='mock',
        eval_type=EvalType.MOCK_LLM,
        datasets=['gsm8k'],
        limit=1,
        dry_run=dry_run,
    )


def test_dry_run_returns_without_inference() -> None:
    with patch('evalscope.run.run_single_task') as run_single, patch('evalscope.run.setup_work_directory') as setup:
        result = run_task(_cfg(dry_run=True))

    assert result == {}
    run_single.assert_not_called()
    setup.assert_not_called()


def test_dry_run_false_proceeds_to_single_task() -> None:
    with patch('evalscope.run.run_single_task', return_value={}) as run_single:
        result = run_task(_cfg(dry_run=False))

    assert result == {}
    run_single.assert_called_once()


def test_dry_run_list_runs_each_config() -> None:
    with patch('evalscope.run.run_single_task') as run_single:
        result = run_task([_cfg(dry_run=True), _cfg(dry_run=True)])

    assert result == [{}, {}]
    run_single.assert_not_called()
