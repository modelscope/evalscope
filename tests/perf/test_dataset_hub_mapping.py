from typing import Any, Dict

import pytest

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets import base
from evalscope.perf.plugin.datasets.longalpaca import LongAlpacaDatasetPlugin
from evalscope.perf.plugin.datasets.mmmu_multi_image import MMMUMultiImageDatasetPlugin
from evalscope.perf.plugin.datasets.openqa import OpenqaDatasetPlugin
from evalscope.perf.plugin.datasets.share_gpt import ShareGPTEnDatasetPlugin, ShareGPTZhDatasetPlugin
from evalscope.perf.plugin.datasets.swe_smith import SweSmithDatasetPlugin


def _args(**kwargs: Any) -> Arguments:
    return Arguments(model='test-model', url='http://localhost:8080/v1/chat/completions', **kwargs)


def test_openqa_maps_repo_id_for_huggingface(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: Dict[str, Any] = {}
    monkeypatch.setattr(base, 'download_dataset_file', lambda **kwargs: calls.update(kwargs) or '/tmp/open_qa.jsonl')

    plugin = OpenqaDatasetPlugin(_args(dataset='openqa', data_source='huggingface'))

    assert plugin.download_hub_file(file_name='open_qa.jsonl') == '/tmp/open_qa.jsonl'
    assert calls == {
        'data_id_or_path': 'Hello-SimpleAI/HC3-Chinese',
        'file_path': 'open_qa.jsonl',
        'data_source': 'huggingface',
        'revision': None,
    }


@pytest.mark.parametrize(
    ('plugin_cls', 'file_name', 'hf_file_path'),
    [
        (ShareGPTZhDatasetPlugin, 'common_zh_70k.jsonl', 'sharegpt_jsonl/common_zh_70k.jsonl'),
        (ShareGPTEnDatasetPlugin, 'common_en_70k.jsonl', 'sharegpt_jsonl/common_en_70k.jsonl'),
    ],
)
def test_sharegpt_maps_repo_and_file_path_for_huggingface(
    monkeypatch: pytest.MonkeyPatch,
    plugin_cls: type,
    file_name: str,
    hf_file_path: str,
) -> None:
    calls: Dict[str, Any] = {}
    monkeypatch.setattr(base, 'download_dataset_file', lambda **kwargs: calls.update(kwargs) or '/tmp/sharegpt.jsonl')

    plugin = plugin_cls(_args(dataset='share_gpt_zh', data_source='huggingface'))
    plugin.download_hub_file(file_name=file_name)

    assert calls['data_id_or_path'] == 'shareAI/ShareGPT-Chinese-English-90k'
    assert calls['file_path'] == hf_file_path
    assert calls['data_source'] == 'huggingface'


def test_longalpaca_maps_repo_for_huggingface(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: Dict[str, Any] = {}
    monkeypatch.setattr(base, 'load_dataset_from_hub', lambda **kwargs: calls.update(kwargs) or [])

    plugin = LongAlpacaDatasetPlugin(_args(dataset='longalpaca', data_source='huggingface'))
    plugin.load_hub_dataset()

    assert calls['data_id_or_path'] == 'Yukang/LongAlpaca-12k'
    assert calls['data_source'] == 'huggingface'
    assert calls['version'] is None


def test_same_repo_id_loads_from_huggingface_without_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: Dict[str, Any] = {}
    monkeypatch.setattr(base, 'load_dataset_from_hub', lambda **kwargs: calls.update(kwargs) or [])

    plugin = SweSmithDatasetPlugin(_args(dataset='swe_smith', data_source='huggingface'))
    plugin.load_hub_dataset(split='tool')

    assert calls['data_id_or_path'] == 'SWE-bench/SWE-smith-trajectories'
    assert calls['data_source'] == 'huggingface'
    assert calls['version'] is None


def test_mmmu_maps_matching_hf_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: Dict[str, Any] = {}
    monkeypatch.setattr(base, 'load_dataset_from_hub', lambda **kwargs: calls.update(kwargs) or [])

    plugin = MMMUMultiImageDatasetPlugin(_args(dataset='mmmu_multi_image', data_source='huggingface'))
    plugin.load_hub_dataset(split='validation', subset='Accounting')

    assert calls['data_id_or_path'] == 'MMMU/MMMU'
    assert calls['version'] == '21d1d90a93c7450d30bddb579d7b510c00b8a9ab'
    assert calls['split'] == 'validation'
    assert calls['subset'] == 'Accounting'


def test_modelscope_keeps_original_repo_and_file_path(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: Dict[str, Any] = {}
    monkeypatch.setattr(base, 'download_dataset_file', lambda **kwargs: calls.update(kwargs) or '/tmp/sharegpt.jsonl')

    plugin = ShareGPTZhDatasetPlugin(_args(dataset='share_gpt_zh', data_source='modelscope'))
    plugin.download_hub_file(file_name='common_zh_70k.jsonl')

    assert calls['data_id_or_path'] == 'swift/sharegpt'
    assert calls['file_path'] == 'common_zh_70k.jsonl'
    assert calls['data_source'] == 'modelscope'


def test_local_file_keeps_highest_precedence(tmp_path) -> None:
    dataset_path = tmp_path / 'open_qa.jsonl'
    dataset_path.write_text('{}\n', encoding='utf-8')
    plugin = OpenqaDatasetPlugin(
        _args(dataset='openqa', dataset_path=str(dataset_path), data_source='huggingface')
    )

    assert plugin.download_hub_file(file_name='open_qa.jsonl') == str(dataset_path)
