import csv
import shutil
import sys
import tarfile
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.benchmark.adapters.browsergym_adapter import BrowserGymSession
from evalscope.benchmarks.miniwob import utils as miniwob_utils
from evalscope.benchmarks.miniwob.miniwob_adapter import MiniWobAdapter
from evalscope.config import TaskConfig


def test_browsergym_package_contract() -> None:
    pytest.importorskip('browsergym.miniwob')
    from browsergym.core.action.highlevel import HighLevelActionSet

    action_mapping = HighLevelActionSet(
        subsets=['miniwob_all'],
        multiaction=False,
        strict=False,
        retry_with_force=True,
        demo_mode='off',
    ).to_python_code
    assert callable(action_mapping)


def test_adapter_checks_browsergym_dependency_during_init() -> None:
    meta = BenchmarkMeta(name='miniwob', pretty_name='MiniWoB', dataset_id='browsergym')
    config = TaskConfig(datasets=['miniwob'])

    with patch('evalscope.api.benchmark.adapters.browsergym_adapter.check_import') as check_import:
        MiniWobAdapter(benchmark_meta=meta, task_config=config)

    check_import.assert_called_once_with(
        'browsergym.miniwob',
        extra='miniwob',
        raise_error=True,
        feature_name='MiniWoB',
    )


def test_load_miniwob_records_builds_a_deterministic_repeat_schedule(tmp_path: Path) -> None:
    metadata_path = tmp_path / 'miniwob.csv'
    with metadata_path.open('w', encoding='utf-8', newline='') as metadata_file:
        writer = csv.DictWriter(metadata_file, fieldnames=miniwob_utils._EXPECTED_FIELDS)
        writer.writeheader()
        for task_name in ('miniwob.a', 'miniwob.b'):
            writer.writerow({
                'task_name': task_name,
                'miniwob_category': 'test',
                'comment': '',
                'webgum_subset': 'False',
                'similarity_group': '0',
                'browsergym_split': 'test',
            })

    with patch.object(miniwob_utils, 'ensure_miniwob_metadata', return_value=metadata_path):
        first, first_path = miniwob_utils.load_miniwob_records(repeats=2)
        second, second_path = miniwob_utils.load_miniwob_records(repeats=2)

    assert first_path == second_path == metadata_path
    assert first == second
    assert [record['task_id'] for record in first] == ['miniwob.a', 'miniwob.b']
    assert all(len(record['_episode_seeds']) == 2 for record in first)
    assert all(0 <= seed < miniwob_utils.MINIWOB_SEED_MAX for record in first for seed in record['_episode_seeds'])


def test_ensure_miniwob_assets_extracts_the_verified_archive(tmp_path: Path) -> None:
    archive_source = tmp_path / 'source.tar.gz'
    source_root = tmp_path / 'archive-root'
    html_dir = source_root / 'miniwob' / 'html' / 'miniwob'
    html_dir.mkdir(parents=True)
    (html_dir / 'click-dialog.html').write_text('<html></html>', encoding='utf-8')
    with tarfile.open(archive_source, 'w:gz') as archive:
        archive.add(source_root, arcname='miniwob-source')

    def copy_archive(url: str, save_path: str, **kwargs: Any) -> None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(archive_source, save_path)

    with patch.object(miniwob_utils, 'download_url', side_effect=copy_archive) as download:
        extracted = miniwob_utils.ensure_miniwob_assets(cache_root=tmp_path / 'cache')
        reused = miniwob_utils.ensure_miniwob_assets(cache_root=tmp_path / 'cache')

    assert extracted == reused
    assert (extracted / 'click-dialog.html').read_text(encoding='utf-8') == '<html></html>'
    download.assert_called_once()


def test_browsergym_session_uses_the_expected_gym_contract() -> None:
    environment = MagicMock()
    environment.reset.return_value = ({'axtree_txt': 'initial'}, {'ignored': True})
    environment.step.return_value = ({'axtree_txt': 'next'}, 1.0, True, False, {'ignored': True})
    gymnasium = ModuleType('gymnasium')
    gymnasium.make = MagicMock(return_value=environment)
    session = BrowserGymSession(
        task_name='miniwob.click-dialog',
        seed=42,
        action_mapping=MagicMock(),
        max_steps=10,
        task_kwargs={'base_url': 'file:///tmp/miniwob/'},
    )

    with patch.dict(sys.modules, {'gymnasium': gymnasium}):
        reset_result = session._reset()
        step_result = session._step('click("1")')
        session._close()

    gymnasium.make.assert_called_once_with(
        'browsergym/miniwob.click-dialog',
        disable_env_checker=True,
        max_episode_steps=10,
        headless=True,
        action_mapping=session.action_mapping,
        task_kwargs={'base_url': 'file:///tmp/miniwob/'},
    )
    environment.reset.assert_called_once_with(seed=42)
    environment.step.assert_called_once_with('click("1")')
    environment.close.assert_called_once_with()
    assert reset_result.observation == {'axtree_txt': 'initial'}
    assert step_result.reward == 1.0
    assert step_result.done is True


@pytest.mark.parametrize(
    'action',
    [
        'click(1)',
        'page.click("1")',
        'click(str(1))',
        'click("1"); noop()',
    ],
)
def test_validate_browser_action_rejects_unsupported_shapes(action: str) -> None:
    with pytest.raises(ValueError):
        miniwob_utils.validate_browser_action(action)
