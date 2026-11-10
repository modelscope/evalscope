"""Local-file benchmarks must honor the configured dataset shuffle seed."""

import json
import random
from collections.abc import Iterator
from pathlib import Path

import pytest

from evalscope.api.registry import BENCHMARK_REGISTRY, get_benchmark
from evalscope.config import TaskConfig


@pytest.fixture(autouse=True)
def preserve_random_state() -> Iterator[None]:
    state = random.getstate()
    try:
        yield
    finally:
        random.setstate(state)


@pytest.fixture
def frames_directory(tmp_path: Path) -> Path:
    rows = [
        {
            'Prompt': f'question-{index}',
            'Answer': f'answer-{index}',
            'wiki_items': [{'title': f'title-{index}', 'text': f'context-{index}'}],
        }
        for index in range(20)
    ]
    (tmp_path / 'test.jsonl').write_text(
        '\n'.join(json.dumps(row) for row in rows) + '\n', encoding='utf-8'
    )
    return tmp_path


def load_targets(directory: Path, seed: int | None, shuffle: bool = True) -> list[str]:
    config = TaskConfig(
        datasets=['frames'],
        dataset_args={'frames': {'dataset_id': str(directory), 'shuffle': shuffle}},
        seed=seed,
        limit=5,
    )
    dataset, _ = get_benchmark('frames', config).load()
    return [str(sample.target) for sample in dataset['test']]


@pytest.mark.parametrize('seed', [0, 17, 42])
def test_local_file_shuffle_uses_configured_seed(frames_directory: Path, seed: int) -> None:
    expected = [f'answer-{index}' for index in range(20)]
    random.Random(seed).shuffle(expected)

    random.seed(123)
    first = load_targets(frames_directory, seed)
    random.seed(456)
    second = load_targets(frames_directory, seed)

    assert first == second == expected[:5]


@pytest.mark.parametrize('seed', [None, 42])
def test_local_file_without_shuffle_preserves_file_order(frames_directory: Path, seed: int | None) -> None:
    random.seed(123)
    first = load_targets(frames_directory, seed, shuffle=False)
    random.seed(456)
    second = load_targets(frames_directory, seed, shuffle=False)

    assert first == second == [f'answer-{index}' for index in range(5)]


def test_local_file_none_seed_preserves_global_random_behavior(frames_directory: Path) -> None:
    expected = [f'answer-{index}' for index in range(20)]
    random.Random(123).shuffle(expected)
    random.seed(123)

    assert load_targets(frames_directory, None) == expected[:5]


@pytest.mark.parametrize(
    'benchmark, version',
    [
        ('frames', 'v1.2'),
        ('maritime_ocr_bench', 'v1.2'),
        ('data_collection', 'v1.1'),
        ('health_bench', 'v1.1'),
        ('pmc_vqa', 'v1.1'),
        ('wide_search', 'v1.1'),
    ],
)
def test_local_file_shuffle_versions_track_sample_selection(benchmark: str, version: str) -> None:
    assert BENCHMARK_REGISTRY[benchmark].evaluation_version == version
