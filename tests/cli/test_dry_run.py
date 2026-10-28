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


def test_dry_run_log_hides_api_url_credentials() -> None:
    api_url = 'https://user:pw-secret@example.com:8443/v1?api-key=query-secret'
    cfg = TaskConfig(model='m', api_url=api_url, datasets=['gsm8k'], dry_run=True)

    with patch('evalscope.run.logger') as logger:
        run_task(cfg)

    logged = ''.join(str(call.args[0]) for call in logger.info.call_args_list)
    assert 'pw-secret' not in logged
    assert 'query-secret' not in logged
    assert '"api_url": "example.com:8443"' in logged
    assert cfg.api_url == api_url


def test_logged_api_url_does_not_change_request_url() -> None:
    api_url = 'https://user:pw-secret@example.com/v1'
    cfg = TaskConfig(model='m', api_url=api_url, datasets=['gsm8k'])

    with patch('evalscope.run.run_single_task', return_value={}) as run_single:
        run_task(cfg)

    assert run_single.call_args.args[0].api_url == api_url
