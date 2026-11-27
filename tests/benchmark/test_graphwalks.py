"""GraphWalks reference parity, offline pipeline, and opt-in official-data verification."""

import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import datasets
import pytest

from evalscope import TaskConfig, run_task
from evalscope.api.dataset import DatasetHub
from evalscope.api.evaluator import TaskState
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.graphwalks.graphwalks_adapter import DATASET_REVISION, GraphWalksAdapter
from evalscope.models.mockllm import MockLLM
from evalscope.report import Report


def _adapter(tmp_path: Path, **kwargs: Any) -> GraphWalksAdapter:
    return get_benchmark(
        'graphwalks',
        TaskConfig(
            model='mock',
            eval_type='mock_llm',
            datasets=['graphwalks'],
            dataset_hub='huggingface',
            dataset_dir=str(tmp_path / 'cache'),
            **kwargs,
        ),
    )


def _record(problem_type: str, answer: list[str], prompt: str = ' Graph {edges}\n') -> dict[str, Any]:
    return {
        'prompt': prompt,
        'answer_nodes': answer,
        'prompt_chars': len(prompt),
        'problem_type': problem_type,
        'date_added': '2026-02-27',
    }


def _reference_score(response: str, answer_nodes: list[str]) -> tuple[dict[str, float], bool]:
    # Independent transcription of OpenAI's MIT-licensed README grader, pinned to:
    # https://huggingface.co/datasets/openai/graphwalks/blob/be6cc6ecf9b4d495b07d1ff53d2a16598e90fed7/README.md
    def get_list(response: str) -> tuple[list[str], bool]:
        line = response.split('\n')[-1]
        if 'Final Answer:' not in line:
            return [], True
        list_part = re.search(r'\[.*\]', line)
        if list_part:
            result_list = list_part.group(0).strip('[]').split(',')
            result_list = [item.strip() for item in result_list if item.strip()]
            return result_list, False
        return [], True

    sampled_list, failed_to_parse = get_list(response)
    sampled_set = set(sampled_list)
    truth_set = set(answer_nodes)
    n_golden = len(truth_set)
    n_sampled = len(sampled_set)
    if failed_to_parse:
        recall = precision = f1 = 0.0
    elif n_golden == n_sampled == 0:
        recall = precision = f1 = 1.0
    else:
        n_overlap = len(sampled_set & truth_set)
        recall = n_overlap / n_golden if n_golden > 0 else 0
        precision = n_overlap / n_sampled if n_sampled > 0 else 0
        f1 = 2 * (recall * precision) / (recall + precision) if recall + precision > 0 else 0
    return {'precision': precision, 'recall': recall, 'f1': f1}, failed_to_parse


RESPONSES = [
    'Final Answer: [a, b]',
    'Final Answer: [b, a, a]',
    'Final Answer: [a, c]',
    'Final Answer: [x]',
    'Final Answer: []',
    'Final Answer: [ , , ]',
    'Reasoning\nFinal Answer: [ a , b ]',
    'Final Answer: [a]\nFinal Answer: [b]',
    'Final Answer: [a]\n',
    'Final Answer: [a]\nExplanation',
    'Final Answer: [a]\r\n',
    'Final Answer: [a]\r',
    'Final Answer: [a] trailing text',
    'prefix Final Answer: [a]',
    'final answer: [a]',
    'Final Answer: a',
    'Final Answer: [a',
    'Final Answer: [a] [b]',
    '[a] Final Answer: [b]',
    'Final Answer: [[a]]',
    'Final Answer: ["a", "b"]',
    "Final Answer: ['a', 'b']",
    'Final Answer: [A, B]',
    'Final Answer: [a,\nb]',
    '```\nFinal Answer: [a]\n```',
    '[]',
    '',
]


@pytest.mark.parametrize('response', RESPONSES)
@pytest.mark.parametrize('target', [[], ['a'], ['a', 'b'], ['a', 'a', 'b']])
def test_reference_parity(tmp_path: Path, response: str, target: list[str]) -> None:
    adapter = _adapter(tmp_path)
    sample = adapter.record_to_sample(_record('bfs', target))
    sample.id = 0
    state = TaskState(
        sample=sample, output=ModelOutput.from_content('mock', response), model='mock', completed=True
    )
    score = adapter.calculate_metrics(state).score
    expected, failed = _reference_score(response, target)
    assert score.value == pytest.approx(expected)
    assert score.metadata['failed_to_parse'] is failed
    assert score.main_score_name == 'f1'
    assert score.prediction == response


def test_prompt_and_target_preserved(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    record = _record('parents', [])
    sample = adapter.record_to_sample(record)
    assert adapter.process_sample_str_input(sample, 'parents')[0].content == record['prompt']
    assert json.loads(sample.target) == []
    assert sample.subset_key == 'parents'
    assert sample.metadata == {'problem_type': 'parents', 'prompt_chars': len(record['prompt'])}
    assert adapter.benchmark_meta.evaluation_version == 'v1.0'
    with pytest.raises(ValueError, match='does not support few-shot'):
        _adapter(tmp_path, dataset_args={'graphwalks': {'few_shot_num': 1}})


def test_remote_loader_pin_and_subset_controls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    records = [_record(task, [str(i)], f'{task}-{i}') for task in ('bfs', 'parents') for i in range(4)]
    calls = []

    def load_dataset(**kwargs: Any) -> datasets.Dataset:
        calls.append(kwargs)
        return datasets.Dataset.from_list(records)

    monkeypatch.setattr(datasets, 'load_dataset', load_dataset)
    adapter = _adapter(
        tmp_path,
        limit=0.5,
        repeats=2,
        dataset_args={'graphwalks': {'subset_list': ['parents'], 'shuffle': True}},
    )
    loaded = adapter.load_dataset()
    assert list(loaded.keys()) == ['parents']
    assert len(loaded['parents']) == 4
    assert len({s.group_id for s in loaded['parents']}) == 2
    assert all(s.metadata['problem_type'] == 'parents' for s in loaded['parents'])
    assert len(calls) == 1
    assert calls[0]['path'] == 'openai/graphwalks'
    assert calls[0]['revision'] == DATASET_REVISION
    assert calls[0]['split'] == 'train'
    assert calls[0]['name'] is None


def test_mock_evaluation_both_tasks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(datasets.config, 'HF_DATASETS_CACHE', str(tmp_path / 'hf-cache'))
    records = [
        _record('bfs', ['a'], ' BFS exact {graph}\n'),
        _record('bfs', ['a'], 'BFS partial'),
        _record('bfs', [], 'BFS malformed'),
        _record('parents', [], 'Parents empty'),
        _record('parents', ['a'], 'Parents trailing newline'),
    ]
    responses = ['Final Answer: [a, a]', 'Final Answer: [a, c]', '[]', 'Final Answer: []', 'Final Answer: [a]\n']
    scripted = dict(zip([r['prompt'] for r in records], responses))
    seen = []

    def generate(self: MockLLM, input: list, **kwargs: Any) -> ModelOutput:
        assert len(input) == 1 and input[0].role == 'user'
        prompt = input[0].content
        seen.append(prompt)
        return ModelOutput.from_content('mock', scripted[prompt])

    monkeypatch.setattr(MockLLM, 'generate', generate)
    source = tmp_path / 'data'
    source.mkdir()
    # Mirror the official two-file layout; the ordinary loader must discover both.
    datasets.Dataset.from_list(records[:2]).to_parquet(source / 'graphwalks_128k_and_shorter.parquet')
    datasets.Dataset.from_list(records[2:]).to_parquet(source / 'graphwalks_256k_to_1mil.parquet')
    output = tmp_path / 'output'
    run_task(
        TaskConfig(
            model='graphwalks-mock',
            eval_type='mock_llm',
            datasets=['graphwalks'],
            dataset_hub='huggingface',
            dataset_args={'graphwalks': {'local_path': str(source)}},
            dataset_dir=str(tmp_path / 'cache'),
            work_dir=str(output),
            no_timestamp=True,
            eval_batch_size=1,
        )
    )
    assert Counter(seen) == Counter(scripted.keys())
    report = Report.from_dict(json.loads((output / 'reports/graphwalks-mock/graphwalks.json').read_text()))
    assert report.primary_metric_identity.name == 'f1'
    assert report.num == 5
    assert report.score == pytest.approx(8 / 15, abs=0.0001)
    metrics = {metric.identity.name: metric for metric in report.metrics}
    assert {name: metric.score for name, metric in metrics.items()} == pytest.approx(
        {'f1': 8 / 15, 'precision': 0.5, 'recall': 0.6}, abs=0.0001
    )
    subsets = {s.name: s for c in metrics['f1'].categories for s in c.subsets if not s.is_aggregate}
    assert {name: s.num for name, s in subsets.items()} == {'bfs': 3, 'parents': 2}
    assert subsets['bfs'].score == pytest.approx(5 / 9, abs=0.0001)
    assert subsets['parents'].score == 0.5
    reviews = [
        json.loads(line)
        for path in (output / 'reviews/graphwalks-mock').glob('*.jsonl')
        for line in path.read_text().splitlines()
    ]
    assert len(reviews) == 5
    assert sum(r['sample_score']['score']['metadata']['failed_to_parse'] for r in reviews) == 2


@pytest.mark.skipif(
    os.environ.get('EVALSCOPE_GRAPHWALKS_DOWNLOAD') != '1',
    reason='Set EVALSCOPE_GRAPHWALKS_DOWNLOAD=1 to verify the complete official dataset (about 319 MB).',
)
def test_official_dataset(tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    adapter = _adapter(tmp_path)
    loaded = adapter.load_dataset()
    assert {task: len(rows) for task, rows in loaded.items()} == {'bfs': 550, 'parents': 600}
    actual = Counter(
        (task, hashlib.sha256(sample.input[0].content.encode()).hexdigest(), sample.target)
        for task, rows in loaded.items()
        for sample in rows
    )
    hub = DatasetHub('openai/graphwalks', data_source='huggingface', revision=DATASET_REVISION)
    expected = Counter()
    for filename, counts in [
        ('graphwalks_128k_and_shorter.parquet', {'bfs': 350, 'parents': 400}),
        ('graphwalks_256k_to_1mil.parquet', {'bfs': 200, 'parents': 200}),
    ]:
        parquet = pq.ParquetFile(hub.download_file(filename))
        task_counts = Counter()
        for batch in parquet.iter_batches(batch_size=16):
            for row in batch.to_pylist():
                task_counts[row['problem_type']] += 1
                expected[(row['problem_type'], hashlib.sha256(row['prompt'].encode()).hexdigest(), json.dumps(row['answer_nodes']))] += 1
        assert task_counts == counts
    assert actual == expected
    assert sum(actual.values()) == 1150
