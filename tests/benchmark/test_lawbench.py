"""Offline native LawBench contract tests."""

import json
import math
from pathlib import Path
from typing import Any

import pytest

from evalscope.api.benchmark import DataAdapter
from evalscope.api.metric import SampleScore, Score
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.lawbench.scoring import score_response
from evalscope.config import TaskConfig


def _adapter(path: Path, subsets: list[str], **kwargs: Any) -> DataAdapter:
    return get_benchmark('lawbench', TaskConfig(
        datasets=['lawbench'], dataset_args={'lawbench': {'local_path': str(path), 'subset_list': subsets}}, **kwargs,
    ))


def test_native_loader_preserves_prompt_and_standard_repeats(tmp_path: Path) -> None:
    records = [{'instruction': '选择答案', 'question': f'题目{i}', 'answer': '正确答案：A'} for i in range(4)]
    (tmp_path / '1-2.json').write_text(json.dumps(records))
    adapter = _adapter(tmp_path, ['1-2'], repeats=2, limit=2)
    samples = list(adapter.load_dataset()['1-2'])
    assert len(samples) == 4
    assert [s.id for s in samples] == [0, 1, 2, 3]
    assert [s.group_id for s in samples] == [0, 0, 1, 1]
    assert samples[0].input[0].content == '选择答案\n题目0'
    assert samples[0].target == '正确答案：A'
    assert samples[0].metadata['task'] == '1-2'
    assert adapter.benchmark_meta.evaluation_version == 'v1.0'


@pytest.mark.parametrize('task', ['1-2', '2-5', None])
def test_hub_loader_preserves_official_rows_and_task_context(
    task: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datasets import Dataset as HFDataset

    from evalscope.api.dataset import DatasetHub

    records = [{'instruction': '任务指令', 'question': f'句子：\n原句{i}', 'answer': f'答案{i}'} for i in range(4)]
    seen = []

    def load(hub: DatasetHub, split: str, subset: str, **kwargs: Any) -> HFDataset:
        seen.append((hub.data_id_or_path, hub.data_source, hub.revision, split, subset))
        return HFDataset.from_list(records)

    monkeypatch.setattr(DatasetHub, 'load', load)
    options: dict[str, Any] = {'dataset_revision': 'test-revision', 'shuffle': True}
    if task is not None:
        options['subset_list'] = [task]
    adapter = get_benchmark('lawbench', TaskConfig(
        datasets=['lawbench'], dataset_dir=str(tmp_path), limit=2, repeats=2,
        dataset_args={'lawbench': options},
    ))
    datasets = adapter.load_dataset()
    expected = [task] if task is not None else [
        '1-1', '1-2', *(f'2-{i}' for i in range(2, 11)), *(f'3-{i}' for i in range(1, 9)),
    ]
    assert list(datasets.keys()) == expected
    assert seen == [('evalscope/LawBench', 'modelscope', 'test-revision', 'zero_shot', name) for name in expected]
    for name, samples in datasets.items():
        assert len(samples) == 4
        assert [sample.group_id for sample in samples] == [0, 0, 1, 1]
        assert len({sample.target for sample in samples}) == 2
        for sample in samples:
            source = next(record for record in records if record['answer'] == sample.target)
            assert sample.input[0].content == f'{source["instruction"]}\n{source["question"]}'
            assert sample.metadata['task'] == name


def test_native_aggregation_keeps_official_denominators(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path, ['2-2'])
    scores = [SampleScore(sample_id=i, score=Score(value=value), sample_metadata={'task': '2-2'})
              for i, value in enumerate([
                  {'lawbench_score': 1.0, 'abstention_rate': 0.0},
                  {'lawbench_score': 0.0, 'abstention_rate': 1.0}, {'abstention_rate': 0.0},
              ])]
    aggregates = {a.metric_name: a for a in adapter.aggregate_scores(scores)}
    assert aggregates['lawbench_score'].score == 0.5
    assert aggregates['lawbench_score'].num == 2
    assert aggregates['abstention_rate'].score == pytest.approx(1 / 3)
    assert aggregates['abstention_rate'].num == 3


def test_native_report_weights_tasks_equally(tmp_path: Path) -> None:
    from evalscope.api.metric import AggScore
    adapter = _adapter(tmp_path, ['1-1', '2-2', '2-3'])
    scores = {task: [AggScore(metric_name='lawbench_score', aggregation='mean', score=value, num=count)]
              for task, value, count in [('1-1', 1, 1), ('2-2', 0, 8), ('2-3', 0.5, 2)]}
    report = adapter.generate_report(scores, 'scripted', '')
    assert report.score == 0.5
    assert {c.name[0]: c.score for c in report.metrics[0].categories} == {
        'memorization': 1.0, 'understanding': 0.25,
    }
    from evalscope.report import Report
    restored = Report.model_validate_json(report.model_dump_json())
    assert restored.score == 0.5
    assert {c.name[0]: c.score for c in restored.metrics[0].categories} == {
        'memorization': 1.0, 'understanding': 0.25,
    }


def test_native_loader_rejects_unsupported_task_and_fewshot(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='subset'):
        _adapter(tmp_path, ['../1-2'])
    with pytest.raises(ValueError, match='few_shot|shot'):
        get_benchmark('lawbench', TaskConfig(datasets=['lawbench'], dataset_args={'lawbench': {'few_shot_num': 1}}))


@pytest.mark.parametrize('subsets', [['2-1'], ['1-2', '2-1']])
def test_proofreading_is_rejected_before_dataset_loading(
    subsets: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evalscope.api.dataset import DatasetHub

    monkeypatch.setattr(DatasetHub, 'load', lambda *a, **k: pytest.fail('unsupported tasks must not load data'))
    with pytest.raises(ValueError, match='2-1.*not supported'):
        _adapter(tmp_path, subsets)


def test_native_run_and_cache_replay_are_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from evalscope import run_task
    from evalscope.models.mockllm import MockLLM

    data = tmp_path / 'data'
    data.mkdir()
    (data / '1-2.json').write_text(json.dumps([
        {'instruction': '选择', 'question': '题目一', 'answer': '正确答案：A'},
        {'instruction': '选择', 'question': '题目二', 'answer': '正确答案：B'},
    ]))
    monkeypatch.setattr(MockLLM, 'default_output', 'A')
    work = tmp_path / 'run'
    base = dict(model='offline-lawbench', eval_type='mock_llm', datasets=['lawbench'],
                repeats=2, eval_batch_size=4, no_timestamp=True, work_dir=str(work),
                dataset_args={'lawbench': {'local_path': str(data), 'subset_list': ['1-2']}})
    run_task(TaskConfig(**base))
    predictions = work / 'predictions/offline-lawbench/lawbench_1-2.jsonl'
    before = predictions.read_bytes()
    monkeypatch.setattr(MockLLM, 'generate', lambda *a, **k: pytest.fail('cache replay must not generate'))
    run_task(TaskConfig(**base, use_cache=str(work)))
    run_task(TaskConfig(**base, use_cache=str(work), rerun_review=True))
    assert predictions.read_bytes() == before
    report = json.loads((work / 'reports/offline-lawbench/lawbench.json').read_text())
    metric = next(m for m in report['metrics'] if m['identity']['name'] == 'lawbench_score')
    assert metric['score'] == 0.5
    assert metric['num'] == 4


def test_only_official_mean_aggregation_is_supported(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='aggregation'):
        get_benchmark('lawbench', TaskConfig(datasets=['lawbench'], dataset_args={'lawbench': {'aggregation': 'pass_at_k'}}))


def test_scoring_uses_configured_prediction_filters() -> None:
    from evalscope.api.evaluator import TaskState
    from evalscope.api.model import ModelOutput

    task = '1-2'
    adapter = get_benchmark('lawbench', TaskConfig(datasets=['lawbench'], dataset_args={'lawbench': {
        'subset_list': [task], 'filters': {'remove_until': {'delimiter': '|FINAL|'}},
    }}))
    sample = adapter.record_to_sample({'task': task, 'instruction': '选择', 'question': '句子：\n原句',
                                       'answer': '正确答案：A'})
    raw = 'B|FINAL|A'
    scored = adapter.calculate_metrics(TaskState(
        model='scripted', sample=sample, completed=True, output=ModelOutput.from_content('scripted', raw),
    ))
    assert scored.score.prediction == raw
    assert scored.score.extracted_prediction == 'A'
    assert scored.score.value['lawbench_score'] == 1.0


@pytest.mark.parametrize(('task', 'prediction', 'reference'), [
    ('1-1', '依法治国', '答案:依法治国'),
    ('1-2', 'A', '正确答案：A'),
    ('2-2', '合同效力', '争议焦点类别：合同效力。'),
    ('2-3', '婚后有子女', '类别:婚后有子女。'),
    ('2-4', '交通事故', '交通事故'),
    ('2-5', '张三', '回答:张三'),
    ('2-6', '受害人:张三', '受害人:张三'),
    ('2-7', '依法治国', '依法治国'),
    ('2-8', 'E', '[正确答案]E'),
    ('2-9', '欺骗;拘捕', '欺骗;拘捕'),
    ('2-10', '欺骗;拘捕', '欺骗;拘捕'),
    ('3-1', '刑法第二百六十四条', '法条:刑法第264条'),
    ('3-2', '依法治国', '依法治国'),
    ('3-3', '盗窃', '罪名:盗窃'),
    ('3-4', '一年', '刑期:12个月'),
    ('3-5', '12个月', '刑期:12个月'),
    ('3-6', 'D', '正确答案:D'),
    ('3-7', '共计2000元', '上文涉及到的犯罪金额:2000元。'),
    ('3-8', '依法治国', '依法治国'),
])
def test_all_row_scoring_tasks(task: str, prediction: str, reference: str) -> None:
    assert score_response(task, prediction, reference)['score'] == pytest.approx(1.0)


@pytest.mark.parametrize(('prediction', 'score', 'abstention'), [
    ('A', 1, 0), ('AA', 1, 0), ('AB', 0, 0), ('B', 0, 0), ('a', 0, 1), ('', 0, 1),
])
def test_choice_uses_official_substring_rule(prediction: str, score: float, abstention: float) -> None:
    assert score_response('1-2', prediction, '正确答案：A') == {'score': score, 'abstention_rate': abstention}


def test_official_exclusions_do_not_become_wrong_answers() -> None:
    assert score_response('2-2', '', '争议焦点类别：赔偿。') == {'abstention_rate': 0.0}
    assert score_response('3-4', '', '刑期:无期徒刑') == {'abstention_rate': 0.0}
    assert score_response('3-5', '', '刑期:死刑') == {'abstention_rate': 0.0}


def test_imprisonment_prefers_months_and_preserves_unbounded_log_distance() -> None:
    expected = 1 - abs(math.log(13) - math.log(4)) / math.log(216)
    assert score_response('3-4', '一年三个月', '刑期:12个月')['score'] == pytest.approx(expected)
    assert score_response('3-4', '', '刑期:12个月') == {'score': 0.0, 'abstention_rate': 1.0}
    assert score_response('3-4', '999999个月', '刑期:1个月')['score'] < 0


def test_extraction_keeps_official_partial_credit_and_order() -> None:
    assert score_response('2-5', '张三。', '回答:张三')['score'] == 1.0
    assert score_response('2-6', '', '受害人:张三') == {'score': 0.0, 'abstention_rate': 1.0}
    assert score_response('2-10', '甲;乙', '乙;甲')['score'] == 0.0
    assert score_response('2-9', '欺骗', '欺骗;拘捕')['score'] == pytest.approx(2 / 3)
    assert score_response('3-7', '10元、2000元', '上文涉及到的犯罪金额:2000元。')['score'] == 1


@pytest.mark.parametrize('task', ['4-1', '2-1'])
def test_unknown_task_cannot_silently_score_zero(task: str) -> None:
    with pytest.raises(ValueError, match='task'):
        score_response(task, '', '')
