# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

import pytest

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import SampleScore
from evalscope.api.metric.semantics import MetricIdentity, MetricSelector
from evalscope.api.registry import BENCHMARK_REGISTRY
from evalscope.benchmarks.general_qa.general_qa_adapter import GeneralQAAdapter
from evalscope.benchmarks.general_qa_vqa_metrics import METRIC_SCORE_KEYS
from evalscope.config import TaskConfig

ROUGE_KEYS = METRIC_SCORE_KEYS['Rouge']
BLEU_KEYS = METRIC_SCORE_KEYS['BLEU']


def _adapter(metric_list: list[str]) -> GeneralQAAdapter:
    return GeneralQAAdapter(
        benchmark_meta=BenchmarkMeta(
            name='general_qa',
            dataset_id='dummy',
            eval_split='test',
            pretty_name='General-QA',
            description='General QA test adapter.',
            metric_list=metric_list,
            primary_metric=MetricSelector(
                name='rouge', aggregation='mean', dimensions={'variant': 'l', 'statistic': 'recall'}
            ),
        ),
        task_config=TaskConfig(datasets=['general_qa']),
    )


def _task_state() -> TaskState:
    return TaskState(model='mock-model', sample=Sample(input='question', target='answer'))


def _rouge_values(value: float) -> Dict[str, float]:
    return dict.fromkeys(ROUGE_KEYS, value)


def _raise_metric_error(*args: Any, **kwargs: Any) -> Dict[str, float]:
    raise LookupError('metric exploded')


class TestGeneralQAAdapterMatchScore:

    def test_rouge_error_returns_real_zero_score_schema(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter = _adapter(['Rouge'])
        monkeypatch.setattr(
            'evalscope.metrics.utils.rouge.compute_rouge_score_one_sample_zh', _raise_metric_error
        )

        score = adapter.match_score('pred', 'pred', 'answer', _task_state())

        assert score is not None
        assert score.value == dict.fromkeys(ROUGE_KEYS, 0.0)
        assert score.main_score_name == 'Rouge-L-R'
        assert score.metadata['metric_errors']['Rouge'] == 'LookupError: metric exploded'

    def test_rouge_error_is_counted_in_aggregation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter = _adapter(['Rouge'])
        call_count = 0

        def succeed_then_fail(*args: Any, **kwargs: Any) -> Dict[str, float]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _rouge_values(1.0)
            raise LookupError('metric exploded')

        monkeypatch.setattr(
            'evalscope.metrics.utils.rouge.compute_rouge_score_one_sample_zh', succeed_then_fail
        )
        scores = [
            SampleScore(score=adapter.match_score('answer', 'answer', 'answer', _task_state()), sample_id=0),
            SampleScore(score=adapter.match_score('pred', 'pred', 'answer', _task_state()), sample_id=1),
        ]

        rouge_l_recall = next(
            item
            for item in adapter.aggregate_scores(scores)
            if item.identity
            == MetricIdentity(
                name='rouge', aggregation='mean', dimensions={'variant': 'l', 'statistic': 'recall'}
            )
        )

        assert rouge_l_recall.score == 0.5
        assert rouge_l_recall.num == 2

    def test_all_metric_errors_keep_zero_primary_in_report(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter = _adapter(['Rouge', 'BLEU'])
        monkeypatch.setattr(
            'evalscope.metrics.utils.rouge.compute_rouge_score_one_sample_zh', _raise_metric_error
        )
        monkeypatch.setattr('evalscope.metrics.bleu_ngram_one_sample', _raise_metric_error)
        sample_score = SampleScore(score=adapter.match_score('pred', 'pred', 'answer', _task_state()), sample_id=0)

        report = adapter.generate_report(
            {'test': adapter.aggregate_scores([sample_score])}, model_name='mock-model', output_dir=''
        )

        assert report.primary_metric is not None
        assert report.primary_metric.identity.name == 'rouge'
        assert report.primary_metric.score == 0.0
        assert report.primary_metric.num == 1

    def test_bleu_error_keeps_successful_rouge_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter = _adapter(['Rouge', 'BLEU'])
        monkeypatch.setattr(
            'evalscope.metrics.utils.rouge.compute_rouge_score_one_sample_zh',
            lambda *args, **kwargs: _rouge_values(1.0),
        )
        monkeypatch.setattr('evalscope.metrics.bleu_ngram_one_sample', _raise_metric_error)

        score = adapter.match_score('answer', 'answer', 'answer', _task_state())

        assert score.value['Rouge-L-R'] == 1.0
        assert {key: score.value[key] for key in BLEU_KEYS} == dict.fromkeys(BLEU_KEYS, 0.0)
        assert score.metadata['metric_errors']['BLEU'] == 'LookupError: metric exploded'


class TestGeneralQAAdapterMetadata:

    def test_evaluation_version(self) -> None:
        assert BENCHMARK_REGISTRY['general_qa'].evaluation_version == 'v1.1'


class TestGeneralQAAdapterRecordToSample:

    def test_query_answer_record(self) -> None:
        sample = _adapter(['Rouge']).record_to_sample({'question': 'Q?', 'answer': 'A.'})

        assert sample.input[-1].text == 'Q?'
        assert sample.target == 'A.'
