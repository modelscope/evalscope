from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from evalscope.api.dataset import DatasetDict, MemoryDataset, Sample
from evalscope.api.registry import get_benchmark
from evalscope.utils.doc_utils import benchmark_stats, generate_dataset_md
from evalscope.utils.doc_utils.readme_generator import generate_readme_from_dict


def test_suppresses_prompt_statistics_for_runtime_generated_inputs(monkeypatch) -> None:
    statistics = SimpleNamespace(
        to_dict=lambda: {
            'total_samples': 89,
            'prompt_length': {'mean': 0, 'min': 0, 'max': 0, 'std': None},
            'subset_stats': [
                {
                    'name': 'test',
                    'sample_count': 89,
                    'prompt_length_mean': 0,
                    'prompt_length_min': 0,
                    'prompt_length_max': 0,
                    'prompt_length_std': None,
                }
            ],
        }
    )
    monkeypatch.setattr(benchmark_stats, 'compute_benchmark_statistics', lambda adapter: statistics)
    adapter = SimpleNamespace(_suppress_doc_prompt_statistics=True)

    result = generate_dataset_md.compute_adapter_statistics(adapter)

    assert result['total_samples'] == 89
    assert 'prompt_length' not in result
    assert result['subset_stats'] == [{'name': 'test', 'sample_count': 89}]


def test_suppresses_sample_example_for_runtime_generated_inputs(monkeypatch) -> None:
    get_sample_example = Mock()
    monkeypatch.setattr(benchmark_stats, 'get_sample_example', get_sample_example)
    adapter = SimpleNamespace(_suppress_doc_sample_example=True)

    result = generate_dataset_md.get_adapter_sample_example(adapter)

    assert result == {}
    get_sample_example.assert_not_called()


@pytest.mark.parametrize('name,hub', [('graphwalks', 'huggingface'), ('gsm8k', 'modelscope')])
def test_generated_docs_select_dataset_hub(name: str, hub: str) -> None:
    adapter = get_benchmark(name)
    metadata = generate_dataset_md.extract_adapter_meta(adapter)
    readme = generate_readme_from_dict(name, metadata)
    if hub == 'huggingface':
        assert 'https://huggingface.co/datasets/openai/graphwalks' in readme
        assert '--dataset-hub huggingface' in readme
        assert "dataset_hub='huggingface'" in readme
        assert 'https://modelscope.cn/datasets/openai/graphwalks' not in readme
    else:
        assert 'https://modelscope.cn/datasets/' in readme
        assert '--dataset-hub' not in readme
        assert 'dataset_hub=' not in readme


@pytest.mark.parametrize('name,hub', [('graphwalks', 'huggingface'), ('gsm8k', 'modelscope')])
@pytest.mark.parametrize('operation', ['compute_benchmark_statistics', 'get_sample_example'])
def test_doc_data_loading_selects_dataset_hub(
    monkeypatch: pytest.MonkeyPatch, name: str, hub: str, operation: str
) -> None:
    adapter = get_benchmark(name)
    samples = DatasetDict({'default': MemoryDataset([Sample(input='Prompt', target='Answer')])})

    def load_dataset() -> DatasetDict:
        assert adapter.dataset_hub == hub
        return samples

    monkeypatch.setattr(adapter, 'load_dataset', load_dataset)
    assert getattr(benchmark_stats, operation)(adapter) is not None
