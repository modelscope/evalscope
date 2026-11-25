from types import SimpleNamespace
from unittest.mock import Mock

from evalscope.utils.doc_utils import benchmark_stats, generate_dataset_md


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
