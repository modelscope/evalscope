"""Serialized primary selectors survive native evaluation and cached YAML replays."""

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from evalscope import run_task
from evalscope.api.metric.semantics import MetricSelector
from evalscope.api.model.model import ModelCache
from evalscope.api.registry import get_benchmark
from evalscope.arguments import add_argument
from evalscope.config import parse_task_config
from evalscope.models.mockllm import MockLLM


def test_cli_dataset_args_preserve_primary_selector() -> None:
    parser = argparse.ArgumentParser()
    add_argument(parser)
    config = parse_task_config(parser.parse_args([
        '--model', 'primary-config-test', '--eval-type', 'mock_llm', '--datasets', 'general_mcq',
        '--repeats', '2', '--dataset-args', json.dumps({'general_mcq': {
            'aggregation': 'mean_and_vote_at_k',
            'primary_metric': {'name': 'accuracy', 'aggregation': 'vote_at_k', 'dimensions': {'k': 2}},
        }}),
    ]))

    assert get_benchmark('general_mcq', config).primary_metric == MetricSelector(
        name='accuracy', aggregation='vote_at_k', dimensions={'k': 2})


@pytest.mark.parametrize('aggregation', ['vote_at_k', 'pass_at_k', 'pass_hat_k'])
@pytest.mark.parametrize('select_mean', [False, True])
def test_json_evaluation_and_saved_yaml_cache_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, aggregation: str, select_mean: bool,
) -> None:
    monkeypatch.setattr(ModelCache, '_models', {})
    monkeypatch.setattr(MockLLM, 'default_output', 'A')
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'tiny.jsonl').write_text(''.join(
        json.dumps({'question': f'Question {i}', 'A': 'first', 'B': 'second', 'answer': answer}) + '\n'
        for i, answer in enumerate(['A', 'B'])), encoding='utf-8')
    output = tmp_path / 'output'
    selector = {'name': 'accuracy', 'aggregation': 'mean' if select_mean else aggregation,
                'dimensions': {} if select_mean else {'k': 2}}
    config_file = tmp_path / 'input.json'
    config_file.write_text(json.dumps({
        'model': 'primary-config-test', 'eval_type': 'mock_llm', 'datasets': ['general_mcq'],
        'dataset_args': {'general_mcq': {
            'dataset_id': str(data), 'subset_list': ['tiny'], 'few_shot_num': 0,
            'aggregation': 'mean_and_' + aggregation, 'primary_metric': selector,
        }},
        'repeats': 2, 'eval_batch_size': 2, 'work_dir': str(output), 'no_timestamp': True,
    }), encoding='utf-8')

    run_task(str(config_file))

    report_path = next((output / 'reports').rglob('general_mcq.json'))
    report = json.loads(report_path.read_text())
    assert report['primary_metric_identity'] == selector
    assert len(report['metrics']) == 3
    primary = next(m for m in report['metrics'] if m['identity'] == selector)
    assert primary['score'] == 0.5
    assert primary['num'] == (4 if select_mean else 2)
    assert (output / 'reports' / 'report.html').is_file()
    predictions_path = next((output / 'predictions').rglob('*.jsonl'))
    predictions = predictions_path.read_bytes()
    assert len(predictions.splitlines()) == 4
    reviews_path = next((output / 'reviews').rglob('*.jsonl'))
    assert len(reviews_path.read_text().splitlines()) == 4

    restored = parse_task_config(str(output / 'configs' / 'task_config.yaml'))
    assert restored.dataset_args['general_mcq']['primary_metric'] == selector
    restored.use_cache = str(output)

    def unexpected_generate(*args: Any, **kwargs: Any) -> Any:
        pytest.fail('YAML cache replay must not generate new predictions.')

    monkeypatch.setattr(MockLLM, 'generate', unexpected_generate)
    for rerun_review in [False, True]:
        restored.rerun_review = rerun_review
        restored.dump_yaml(str(tmp_path / 'replay'), strict=True)
        run_task(str(tmp_path / 'replay' / 'task_config.yaml'))
        reloaded_report = json.loads(report_path.read_text())
        assert reloaded_report['primary_metric_identity'] == selector
        assert reloaded_report['metrics'] == report['metrics']
        assert predictions_path.read_bytes() == predictions
        assert len(reviews_path.read_text().splitlines()) == 4
